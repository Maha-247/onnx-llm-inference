import modal

app = modal.App("onnx-llm-inference")
image = modal.Image.debian_slim().pip_install("torch", "transformers")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@app.function(gpu="T4", image=image, timeout=600,
              volumes={"/root/.cache/huggingface": hf_cache})
def chat(question: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).half().to("cuda").eval()

    # Wrap the question in the chat format Qwen was trained on
    messages = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    n_prompt = inputs.input_ids.shape[1]
    print(f"Prompt is {n_prompt} tokens (this is the prefill)")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)

    new_ids = out[0, n_prompt:]
    print(f"Generated {len(new_ids)} tokens (each one was a decode step)")
    print("First 10 tokens, one by one:", [tok.decode(t) for t in new_ids[:10]])
    print("\nANSWER:\n" + tok.decode(new_ids, skip_special_tokens=True))


@app.local_entrypoint()
def main(question: str = "Explain what a KV cache is in two sentences."):
    chat.remote(question)