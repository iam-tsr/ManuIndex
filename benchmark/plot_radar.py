"""
Radar chart: holistic quality profile across RAGAS + RAGChecker metrics.
Metrics where lower = better (hallucination, noise sensitivity) are inverted to [0,1].
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPORT = "benchmark/report.json"

with open(REPORT) as f:
    r = json.load(f)

# --- metrics (all scaled to [0,1], higher = better) ---
metrics = {
    "Faithfulness\n(RAGAS)":         r["ragas"]["faithfulness"],
    "Answer\nRelevancy":             r["ragas"]["answer_relevancy"],
    "Context\nRecall":               r["ragas"]["context_recall"],
    "Claim\nRecall":                 r["ragchecker"]["claim_recall"] / 100,
    "Context\nPrecision":            r["ragchecker"]["context_precision"] / 100,
    "Context\nUtilization":          r["ragchecker"]["context_utilization"] / 100,
    "Faithfulness\n(RAGChecker)":    r["ragchecker"]["faithfulness"] / 100,
    "Anti-Hallucin.":                1 - r["ragchecker"]["hallucination"] / 100,
    "Anti-Noise\nSensitivity":       1 - r["ragchecker"]["noise_sensitivity_in_relevant"] / 100,
}

labels = list(metrics.keys())
values = list(metrics.values())
N = len(labels)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values_plot = values + [values[0]]
angles_plot = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")

# grid rings
for r_val in [0.25, 0.5, 0.75, 1.0]:
    ax.plot(angles_plot, [r_val] * (N + 1), color="grey", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.text(angles[0], r_val + 0.03, f"{r_val:.2f}", ha="center", va="bottom",
            fontsize=7, color="grey")

# filled polygon
ax.fill(angles_plot, values_plot, alpha=0.20, color="#1f77b4")
ax.plot(angles_plot, values_plot, color="#1f77b4", linewidth=2)

# data point annotations
for angle, val in zip(angles, values):
    ax.plot(angle, val, "o", color="#1f77b4", markersize=6, zorder=5)
    offset = 0.13
    ax.text(angle, val + offset, f"{val:.2f}", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#1f77b4")

# axis labels
ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticklabels([])
ax.set_ylim(0, 1.2)
ax.spines["polar"].set_visible(False)

ax.set_title("GRAG System — Holistic Quality Profile", fontsize=13,
             fontweight="bold", pad=25)

note = mpatches.Patch(color="none",
    label="Anti-Hallucin. & Anti-Noise = 1 − raw score (higher is better)")
ax.legend(handles=[note], loc="lower center", bbox_to_anchor=(0.5, -0.12),
          fontsize=8, frameon=False)

plt.tight_layout()
out = "benchmark/plot_radar.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
