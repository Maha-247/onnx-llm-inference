"""Draw the sensitivity chart from the saved JSON file. Run:  python plot_sensitivity.py"""
import json
import matplotlib.pyplot as plt

r = json.load(open("results_sensitivity_layers.json"))
layers = [p["layer"] for p in r["per_layer_int4"]]
incs = [p["increase_pct"] for p in r["per_layer_int4"]]

NAMES = {
    "all_int4": "All layers INT4",
    "protect_top_2": "2 most sensitive layers at INT8",
    "protect_top_4": "4 most sensitive layers at INT8",
    "protect_top_8": "8 most sensitive layers at INT8",
    "protect_least_sensitive_4 (control)": "Control: 4 least sensitive at INT8",
    "all_int8": "All layers INT8",
}
mixed = r["mixed_precision"]
labels = [f'{NAMES.get(m["config"], m["config"])}  ({m["avg_bits_per_weight"]:.2f} bits)' for m in mixed]
values = [m["increase_pct"] for m in mixed]

BLUE, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.2), facecolor=SURFACE,
                           gridspec_kw={"width_ratios": [1.25, 1]})

a.bar(layers, incs, width=0.6, color=BLUE)
a.set_title("Perplexity increase when ONE layer is quantized to INT4 (%)",
            loc="left", fontsize=11, color=INK, pad=10)
a.set_xlabel("Layer number (0 = first, 27 = last)", fontsize=10, color=MUTED)
a.set_xticks(range(0, 28, 3))

b.barh(labels, values, height=0.45, color=BLUE)
for i, v in enumerate(values):
    b.text(max(v, 0) + 0.25, i, f"{v:+.1f}%", va="center", color=INK, fontsize=10)
b.set_title("Mixed precision on held-out text: perplexity increase vs. FP16 (%)",
            loc="left", fontsize=11, color=INK, pad=10)
b.set_xlim(min(0, min(values)) - 0.3, max(values) * 1.18)
b.invert_yaxis()

for ax in (a, b):
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="both", labelsize=9, labelcolor=MUTED, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d0cfc9")
b.tick_params(axis="y", length=0, labelsize=10, labelcolor=INK)
b.spines["left"].set_visible(False)

fig.suptitle("Qwen2.5-1.5B: where INT4 quantization costs quality (simulated in PyTorch, lower is better)",
             x=0.01, ha="left", fontsize=12, color=INK, y=1.03)
fig.tight_layout()
fig.savefig("results_sensitivity.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
print("Saved results_sensitivity.png")