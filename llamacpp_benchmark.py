import json
import modal

app = modal.App("onnx-llm-inference")

# llama.cpp must be compiled with GPU (CUDA) support, so we start from NVIDIA's
# CUDA development image. The first build compiles llama.cpp: expect 10-20 minutes.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "cmake", "git")
    .env({
        "CC": "gcc",          # use the C compiler that's installed
        "CXX": "g++",         # and the C++ compiler
        "CMAKE_ARGS": "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 "
                      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined",
    })
    .pip_install("llama-cpp-python", "huggingface_hub", "numpy")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Ready-made GGUF files: one file per precision, already quantized by the publisher.
MODELS = {
    "qwen": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF",
             {"f16": "*fp16.gguf", "q8_0": "*q8_0.gguf", "q4_k_m": "*q4_k_m.gguf"}),
    "llama": ("bartowski/Llama-3.2-1B-Instruct-GGUF",
              {"f16": "*f16.gguf", "q8_0": "*Q8_0.gguf", "q4_k_m": "*Q4_K_M.gguf"}),
}
PROMPT_TOKENS = 512
NEW_TOKENS = 64
TRIALS = 5


@app.function(gpu="T4", image=image, timeout=1800,
              volumes={"/root/.cache/huggingface": hf_cache})
def bench(model: str, precision: str):
    import os, time, statistics, subprocess
    import numpy as np
    import llama_cpp
    from llama_cpp import Llama

    repo, files = MODELS[model]
    llm = Llama.from_pretrained(
        repo_id=repo, filename=files[precision],
        n_gpu_layers=-1,          # put every layer on the GPU
        n_ctx=1024, n_batch=512,  # 512-token prompt goes through in one batch
        verbose=False,
    )
    hf_cache.commit()
    n_vocab = llm.n_vocab()

    text = "The history of computing is a story of making machines faster. " * 200
    ids = llm.tokenize(text.encode("utf-8"), add_bos=False)[:PROMPT_TOKENS]

    def next_token():
        # Reading the logits makes the CPU wait until the GPU has finished.
        ptr = llama_cpp.llama_get_logits_ith(llm.ctx, -1)
        return int(np.argmax(np.ctypeslib.as_array(ptr, shape=(n_vocab,))))

    def run_once():
        llm.reset()                          # empty the KV cache

        # PREFILL: whole prompt -> first token
        t0 = time.perf_counter()
        llm.eval(ids)
        tok = next_token()
        ttft = time.perf_counter() - t0

        # DECODE: one token at a time
        tbts = []
        for _ in range(NEW_TOKENS):
            t0 = time.perf_counter()
            llm.eval([tok])
            tok = next_token()
            tbts.append(time.perf_counter() - t0)
        return ttft, statistics.median(tbts)

    for _ in range(2):                       # warmup
        run_once()
    runs = [run_once() for _ in range(TRIALS)]
    ttft = statistics.median(r[0] for r in runs)
    tbt = statistics.median(r[1] for r in runs)

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().split(", ")
        gpu, mem_gb = out[0], round(int(out[1]) / 1024, 2)
    except Exception:
        gpu, mem_gb = "unknown", None

    return {
        "model": repo, "backend": "llama.cpp", "precision": precision,
        "gguf_file": os.path.basename(llm.model_path), "gpu": gpu,
        "prompt_tokens": len(ids), "new_tokens": NEW_TOKENS,
        "ttft_ms": round(ttft * 1000, 2),
        "tbt_ms": round(tbt * 1000, 2),
        "decode_tokens_per_sec": round(1 / tbt, 1),
        "model_size_on_disk_gb": round(os.path.getsize(llm.model_path) / 1e9, 2),
        "gpu_mem_used_gb_nvidia_smi": mem_gb,
    }


@app.local_entrypoint()
def main(model: str = "qwen", precision: str = "q4_k_m"):
    result = bench.remote(model, precision)
    print(json.dumps(result, indent=2))
    with open(f"results_llamacpp_{model}_{precision}.json", "w") as f:
        json.dump(result, f, indent=2)