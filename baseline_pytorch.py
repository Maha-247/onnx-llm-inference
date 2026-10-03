import json
import modal

app = modal.App("onnx-llm-inference")
image = modal.Image.debian_slim().pip_install("torch", "transformers")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TOKENS = 512   # prompt length (prefill size)
NEW_TOKENS = 64       # tokens to generate (decode steps)
TRIALS = 5


@app.function(gpu="T4", image=image, timeout=1800,
              volumes={"/root/.cache/huggingface": hf_cache})
def baseline():
    import time, statistics, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).half().to("cuda").eval()  # FP16

    text = "The history of computing is a story of making machines faster. " * 200
    ids = tok(text, return_tensors="pt").input_ids[:, :PROMPT_TOKENS].to("cuda")

    def now():
        torch.cuda.synchronize()   # wait for the GPU to finish before reading the clock
        return time.perf_counter()

    @torch.no_grad()
    def run_once():
        # PREFILL: whole prompt at once -> first token
        t0 = now()
        out = model(ids, use_cache=True)
        next_tok = out.logits[:, -1:].argmax(-1)
        ttft = now() - t0
        past = out.past_key_values          # the KV cache

        # DECODE: one token at a time, reusing the KV cache
        tbts = []
        for _ in range(NEW_TOKENS):
            t0 = now()
            out = model(next_tok, past_key_values=past, use_cache=True)
            next_tok = out.logits[:, -1:].argmax(-1)
            past = out.past_key_values
            tbts.append(now() - t0)
        return ttft, statistics.median(tbts)

    for _ in range(2):                      # warmup runs (not counted)
        run_once()

    torch.cuda.reset_peak_memory_stats()
    runs = [run_once() for _ in range(TRIALS)]
    ttft = statistics.median(r[0] for r in runs)
    tbt = statistics.median(r[1] for r in runs)

    return {
        "model": MODEL, "backend": "pytorch", "precision": "fp16",
        "gpu": torch.cuda.get_device_name(0),
        "prompt_tokens": int(ids.shape[1]), "new_tokens": NEW_TOKENS,
        "ttft_ms": round(ttft * 1000, 2),
        "tbt_ms": round(tbt * 1000, 2),
        "decode_tokens_per_sec": round(1 / tbt, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


@app.local_entrypoint()
def main():
    result = baseline.remote()
    print(json.dumps(result, indent=2))
    with open("results_pytorch_fp16.json", "w") as f:
        json.dump(result, f, indent=2)