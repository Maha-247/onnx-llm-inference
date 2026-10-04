"""Draw the results chart from the saved JSON files. Run:  python plot.py"""
import json
import matplotlib.pyplot as plt

CONFIGS = [  # (label, speed results file, backend, precision)
    ("PyTorch FP16", "results_pytorch_fp16.json", "pytorch", "fp16"),
    ("ONNX Runtime FP16", "results_onnx_fp16.json", "onnxruntime-genai", "fp16"),
    ("ONNX Runtime INT8", "results_onnx_int8.json", "onnxruntime-genai", "int8"),
    ("ONNX Runtime INT4", "results_onnx_int4.json", "onnxruntime-genai", "int4"),
]
quality = json.load(open("results_quality.json"))

labels, ttft, tbt, ppl = [], [], [], []
for label, path, backend, precision in CONFIGS:
    r = json.load(open(path))
    labels.append(label)
    ttft.append(r["ttft_ms"])
    tbt.append(r["tbt_ms"])
    ppl.append(next(q["perplexity"] for q in quality
                    if q["backend"] == backend and q["precision"] == precision))

PANELS = [
    ("Prefill: time to first token (ms)", ttft, "{:.0f} ms"),
    ("Decode: time per token (ms)", tbt, "{:.1f} ms"),
    ("Quality: perplexity on WikiText-2", ppl, "{:.2f}"),
]
BLUE, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True, facecolor=SURFACE)
for ax, (title, values, fmt) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.barh(labels, values, height=0.45, color=BLUE)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.02, i, fmt.format(v), va="center", color=INK, fontsize=10)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlim(0, max(values) * 1.25)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=INK)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUTED, color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0cfc9")

fig.suptitle("Qwen2.5-1.5B on NVIDIA T4: 512-token prompt, lower is better in every panel",
             x=0.01, ha="left", fontsize=12, color=INK, y=1.04)
fig.tight_layout()
fig.savefig("results.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
print("Saved results.png")