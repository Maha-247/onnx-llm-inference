import json
import modal

app = modal.App("onnx-llm-inference")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "datasets")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK = 512
N_CHUNKS = 40      # 40 x 512 = about 20,000 tokens of real text


@app.function(gpu="T4", image=image, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def run():
    import math, torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).half().to("cuda").eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])[:200000]
    ids = tok(text, return_tensors="pt").input_ids[0][: CHUNK * N_CHUNKS]
    chunks = [ids[i * CHUNK:(i + 1) * CHUNK] for i in range(len(ids) // CHUNK)]

    @torch.no_grad()
    def perplexity():
        total, count = 0.0, 0
        for c in chunks:
            c = c.to("cuda")
            logits = model(c.unsqueeze(0)).logits[0]
            logp = torch.log_softmax(logits[:-1].float(), dim=-1)
            total += -logp.gather(1, c[1:].unsqueeze(1)).sum().item()
            count += len(c) - 1
        return math.exp(total / count)

    def fake_int4(w, block=32):
        """Round weights to 16 levels (4 bits), in groups of 32, then convert back."""
        w32 = w.float()
        rows, cols = w32.shape
        pad = (-cols) % block
        padded = F.pad(w32, (0, pad))
        groups = padded.view(rows, -1, block)
        scale = (groups.abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-8)
        q = torch.clamp(torch.round(groups / scale), -8, 7)      # quantize: divide by scale
        restored = (q * scale).view(rows, -1)[:, :cols]          # dequantize: multiply by scale
        return restored.to(w.dtype)

    # Find the weight matrices inside the 28 layers, split into two groups
    attention, feedforward = [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ".layers." in name:
            if ".self_attn." in name:
                attention.append(module)
            elif ".mlp." in name:
                feedforward.append(module)
    originals = {m: m.weight.data.clone() for m in attention + feedforward}

    def measure(label, modules):
        for m in modules:
            m.weight.data = fake_int4(originals[m])
        ppl = perplexity()
        for m in modules:                      # put the original weights back
            m.weight.data = originals[m].clone()
        print(f"{label}: perplexity {ppl:.3f}")
        return ppl

    base = measure("FP16 (nothing quantized)", [])
    results = [{"config": "fp16_baseline", "perplexity": round(base, 3), "increase_pct": 0.0}]
    for label, modules in [
        ("int4_attention_only", attention),
        ("int4_feedforward_only", feedforward),
        ("int4_attention_and_feedforward", attention + feedforward),
    ]:
        ppl = measure(label, modules)
        results.append({"config": label, "perplexity": round(ppl, 3),
                        "increase_pct": round((ppl / base - 1) * 100, 2)})
    return {"model": MODEL, "tokens_scored": len(chunks) * (CHUNK - 1),
            "attention_matrices": len(attention), "feedforward_matrices": len(feedforward),
            "results": results}


@app.local_entrypoint()
def main():
    out = run.remote()
    print(json.dumps(out, indent=2))
    with open("results_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)