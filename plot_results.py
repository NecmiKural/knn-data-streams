"""Draws figures/accuracy.png from results.csv: accuracy of the three windowing strategies.

    pip install matplotlib && python3 plot_results.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = {"knn": "Base kNN", "knn-cw": "Class-weighted", "knn-asw": "Age-aware"}
COLOURS = {"knn": "#9ca3af", "knn-cw": "#2563eb", "knn-asw": "#f59e0b"}
SETTINGS = [(100, 3), (100, 21), (500, 3), (500, 21)]
DATASETS = {"electricity": "electricity", "rbf": "RBF with drift"}

with open("results.csv", newline="") as f:
    rows = list(csv.DictReader(f))
accuracy = {(r["dataset"], r["knn_identifier"], int(r["window_size"]), int(r["k"])): float(r["accuracy"]) for r in rows}

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
width = 0.26
for ax, (dataset, title) in zip(axes, DATASETS.items()):
    for j, variant in enumerate(VARIANTS):
        xs = [i + (j - 1) * width for i in range(len(SETTINGS))]
        ys = [accuracy[(dataset, variant, w, k)] for w, k in SETTINGS]
        bars = ax.bar(xs, ys, width, color=COLOURS[variant], label=VARIANTS[variant])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(SETTINGS)), [f"w={w}, k={k}" for w, k in SETTINGS], fontsize=9)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Accuracy (%)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.91))
fig.suptitle("Accuracy by windowing strategy, first 50,000 instances of each stream", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig("figures/accuracy.png", dpi=180)
