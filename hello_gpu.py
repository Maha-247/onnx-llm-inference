import modal

app = modal.App("onnx-llm-inference")
image = modal.Image.debian_slim().pip_install("torch", "transformers")

@app.function(gpu="T4", image=image)
def check_gpu():
    import torch
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA available:", torch.cuda.is_available())

@app.local_entrypoint()
def main():
    check_gpu.remote()