import json
import modal

app = modal.App("onnx-llm-inference")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "datasets")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK = 512
N_CHUNKS = 40


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
    # Choose the sensitive layers on one half of the text, test the result on the other half
    calib, heldout = chunks[::2], chunks[1::2]

    @torch.no_grad()
    def perplexity(chunk_list):
        total, count = 0.0, 0
        for c in chunk_list:
            c = c.to("cuda")
            logits = model(c.unsqueeze(0)).logits[0]
            logp = torch.log_softmax(logits[:-1].float(), dim=-1)
            total += -logp.gather(1, c[1:].unsqueeze(1)).sum().item()
            count += len(c) - 1
        return math.exp(total / count)

    def fake_quant(w, bits, block=32):
        """Round weights to 2**bits levels, in groups of 32, then convert back."""
        qmax = 2 ** (bits - 1) - 1                    # 7 for INT4, 127 for INT8
        w32 = w.float()
        rows, cols = w32.shape
        padded = F.pad(w32, (0, (-cols) % block))
        groups = padded.view(rows, -1, block)
        scale = (groups.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-8)
        q = torch.clamp(torch.round(groups / scale), -qmax - 1, qmax)
        return (q * scale).view(rows, -1)[:, :cols].to(w.dtype)

    # Group the weight matrices by layer number (0 to 27)
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ".layers." in name:
            idx = int(name.split(".layers.")[1].split(".")[0])
            layers.setdefault(idx, []).append(module)
    originals = {m: m.weight.data.clone() for ms in layers.values() for m in ms}
    n = len(layers)

    def measure(bits_per_layer, chunk_list):
        """bits_per_layer = {layer number: 4 or 8}. Layers not listed stay FP16."""
        for idx, bits in bits_per_layer.items():
            for m in layers[idx]:
                m.weight.data = fake_quant(originals[m], bits)
        ppl = perplexity(chunk_list)
        for idx in bits_per_layer:                    # put the original weights back
            for m in layers[idx]:
                m.weight.data = originals[m].clone()
        return ppl

    # PART 1: quantize ONE layer at a time to INT4 (on the first half of the text)
    base_calib = measure({}, calib)
    per_layer = []
    for i in range(n):
        ppl = measure({i: 4}, calib)
        inc = (ppl / base_calib - 1) * 100
        per_layer.append({"layer": i, "perplexity": round(ppl, 3), "increase_pct": round(inc, 3)})
        print(f"layer {i:2d} at INT4: +{inc:.3f}%")
    ranked = [p["layer"] for p in sorted(per_layer, key=lambda p: -p["increase_pct"])]
    print("Most sensitive layers first:", ranked)

    # PART 2: mixed precision, tested on the OTHER half of the text
    base_held = measure({}, heldout)
    mixed = []

    def try_config(label, protected):
        cfg = {i: (8 if i in protected else 4) for i in range(n)}
        ppl = measure(cfg, heldout)
        inc = (ppl / base_held - 1) * 100
        avg_bits = (8 * len(protected) + 4 * (n - len(protected))) / n
        mixed.append({"config": label, "layers_at_int8": sorted(protected),
                      "avg_bits_per_weight": round(avg_bits, 2),
                      "perplexity": round(ppl, 3), "increase_pct": round(inc, 2)})
        print(f"{label}: +{inc:.2f}% at {avg_bits:.2f} bits/weight")

    try_config("all_int4", [])
    try_config("protect_top_2", ranked[:2])
    try_config("protect_top_4", ranked[:4])
    try_config("protect_top_8", ranked[:8])
    try_config("protect_least_sensitive_4 (control)", ranked[-4:])
    try_config("all_int8", list(range(n)))

    return {"model": MODEL, "layers": n,
            "calibration_tokens": len(calib) * (CHUNK - 1),
            "heldout_tokens": len(heldout) * (CHUNK - 1),
            "baseline_perplexity_calibration": round(base_calib, 3),
            "baseline_perplexity_heldout": round(base_held, 3),
            "per_layer_int4": per_layer, "sensitivity_ranking": ranked,
            "mixed_precision": mixed}


@app.local_entrypoint()
def main():
    out = run.remote()
    print(json.dumps(out, indent=2))
    with open("results_sensitivity_layers.json", "w") as f:
        json.dump(out, f, indent=2)