import json
import modal

app = modal.App("onnx-llm-inference")
image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "onnxruntime-genai-cuda",
    "onnx", "onnx-ir", "safetensors", "gguf", "peft", "tqdm", "datasets",
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
models = modal.Volume.from_name("onnx-models", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/models": models}

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK = 512        # tokens per chunk
N_CHUNKS = 8       # 8 x 512 = 4096 tokens of real text


def load_chunks():
    """Real English text (WikiText-2 test set), cut into equal token chunks."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])[:40000]
    ids = tok(text, return_tensors="pt").input_ids[0][: CHUNK * N_CHUNKS]
    return [ids[i * CHUNK:(i + 1) * CHUNK] for i in range(N_CHUNKS)]


def nll_from_logits(logits, ids):
    """How surprised was the model by each real next token? (lower = better)"""
    import torch
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    return -logp.gather(1, ids[1:].unsqueeze(1)).sum().item(), len(ids) - 1


@app.function(gpu="T4", image=image, timeout=1800, volumes=VOLUMES)
def ppl_pytorch():
    import math, torch
    from transformers import AutoModelForCausalLM

    chunks = load_chunks()
    model = AutoModelForCausalLM.from_pretrained(MODEL).half().to("cuda").eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for ids in chunks:
            logits = model(ids.unsqueeze(0).to("cuda")).logits[0]
            nll, n = nll_from_logits(logits.cpu(), ids)
            total, count = total + nll, count + n
    return {"backend": "pytorch", "precision": "fp16",
            "perplexity": round(math.exp(total / count), 3), "tokens_scored": count}


@app.function(gpu="T4", image=image, timeout=1800, volumes=VOLUMES)
def ppl_onnx(precision: str):
    import math
    import numpy as np
    import torch                      # loads the CUDA libraries ONNX Runtime needs
    import onnxruntime_genai as og

    models.reload()
    chunks = load_chunks()
    model = og.Model(f"/models/qwen-{precision}")
    total, count = 0.0, 0
    for ids in chunks:
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=CHUNK + 8, do_sample=False)
        gen = og.Generator(model, params)
        gen.append_tokens(ids.numpy().astype(np.int32))
        logits = torch.from_numpy(np.asarray(gen.get_output("logits")))[0]
        assert logits.shape[0] == len(ids), f"unexpected logits shape {tuple(logits.shape)}"
        nll, n = nll_from_logits(logits, ids)
        total, count = total + nll, count + n
        del gen
    return {"backend": "onnxruntime-genai", "precision": precision,
            "perplexity": round(math.exp(total / count), 3), "tokens_scored": count}


@app.local_entrypoint()
def main():
    results = [ppl_pytorch.remote()]
    for precision in ["fp16", "int8", "int4"]:
        results.append(ppl_onnx.remote(precision))
    print(json.dumps(results, indent=2))
    with open("results_quality.json", "w") as f:
        json.dump(results, f, indent=2)