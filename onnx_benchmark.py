import json
import modal

app = modal.App("onnx-llm-inference")
image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "onnxruntime-genai-cuda",
    "onnx", "onnx-ir", "safetensors", "gguf", "peft", "tqdm",
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
models = modal.Volume.from_name("onnx-models", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/models": models}

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TOKENS = 512
NEW_TOKENS = 64
TRIALS = 5


@app.function(gpu="T4", image=image, timeout=3600, memory=32768, volumes=VOLUMES)
def export(precision: str):
    """Convert the PyTorch model into an ONNX model (runs once per precision)."""
    import os, subprocess, sys

    out = f"/models/qwen-{precision}"
    if os.path.exists(os.path.join(out, "genai_config.json")):
        print("Already exported:", out)
        return
    subprocess.run(
        [sys.executable, "-m", "onnxruntime_genai.models.builder",
         "-m", MODEL, "-o", out, "-p", precision, "-e", "cuda",
         "-c", "/root/.cache/huggingface/builder",
         "--extra_options", "hf_token=false"],      
        check=True,
    )
    models.commit()   # save the exported model to the volume


@app.function(gpu="T4", image=image, timeout=1800, volumes=VOLUMES)
def bench(precision: str):
    import os, time, statistics, subprocess
    import numpy as np
    import torch                      # loads the CUDA libraries ONNX Runtime needs
    import onnxruntime_genai as og

    models.reload()
    path = f"/models/qwen-{precision}"
    size_gb = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1e9

    def gpu_mem_gb():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout
            return round(int(out.strip().splitlines()[0]) / 1024, 2)
        except Exception:
            return None

    model = og.Model(path)
    tok = og.Tokenizer(model)
    text = "The history of computing is a story of making machines faster. " * 200
    ids = np.asarray(tok.encode(text))[:PROMPT_TOKENS]

    def run_once():
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=PROMPT_TOKENS + NEW_TOKENS + 8, do_sample=False)
        gen = og.Generator(model, params)

        # PREFILL: whole prompt -> first token
        t0 = time.perf_counter()
        gen.append_tokens(ids)
        gen.generate_next_token()
        gen.get_next_tokens()             # copies the token to the CPU, so it waits for the GPU
        ttft = time.perf_counter() - t0

        # DECODE: one token at a time (the KV cache is managed inside the generator)
        tbts = []
        for _ in range(NEW_TOKENS):
            if gen.is_done():
                break
            t0 = time.perf_counter()
            gen.generate_next_token()
            gen.get_next_tokens()
            tbts.append(time.perf_counter() - t0)
        del gen
        return ttft, statistics.median(tbts)

    for _ in range(2):                    # warmup
        run_once()
    runs = [run_once() for _ in range(TRIALS)]
    ttft = statistics.median(r[0] for r in runs)
    tbt = statistics.median(r[1] for r in runs)

    return {
        "model": MODEL, "backend": "onnxruntime-genai", "precision": precision,
        "gpu": torch.cuda.get_device_name(0),
        "prompt_tokens": int(len(ids)), "new_tokens": NEW_TOKENS,
        "ttft_ms": round(ttft * 1000, 2),
        "tbt_ms": round(tbt * 1000, 2),
        "decode_tokens_per_sec": round(1 / tbt, 1),
        "model_size_on_disk_gb": round(size_gb, 2),
        "gpu_mem_used_gb_nvidia_smi": gpu_mem_gb(),
    }


@app.local_entrypoint()
def main(precision: str = "fp16"):
    export.remote(precision)
    result = bench.remote(precision)
    print(json.dumps(result, indent=2))
    with open(f"results_onnx_{precision}.json", "w") as f:
        json.dump(result, f, indent=2)