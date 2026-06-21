"""
Grouped bar chart: mean BLEU & ROUGE-L per document set.
Dashed lines mark the corpus-wide averages. Error bars show ±1 std within each set.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

REPORT = "benchmark/report.json"

with open(REPORT) as f:
    r = json.load(f)

# --- aggregate per document set ---
sets = defaultdict(lambda: {"bleu": [], "rougeL": []})
for q in r["per_question"]:
    doc_set = q["id"].rsplit("-", 1)[0]
    sets[doc_set]["bleu"].append(q["bleu"])
    sets[doc_set]["rougeL"].append(q["rougeL"])

DOC_LABELS = {
    "BUD":      "Job Description",
    "MARSHALL": "COVID Safety Plan",
    "ADOP":     "Privacy Policy",
    "MARCOPA":  "Bus Schedule",
    "PAIN":     "Clinical Case",
    "CHURCHILL":"Biography",
    "CORRUPT":  "Water Sector",
    "GROWTH":    "Medical Study",
    "INFERENCE": "Research Paper",
}

doc_sets = list(sets.keys())
bleu_means  = [np.mean(sets[d]["bleu"])   for d in doc_sets]
bleu_stds   = [np.std(sets[d]["bleu"])    for d in doc_sets]
rouge_means = [np.mean(sets[d]["rougeL"]) for d in doc_sets]
rouge_stds  = [np.std(sets[d]["rougeL"])  for d in doc_sets]

x = np.arange(len(doc_sets))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")

bars_bleu  = ax.bar(x - width/2, bleu_means,  width, yerr=bleu_stds,
                    label="BLEU",   color="#4c72b0", capsize=4, alpha=0.88)
bars_rouge = ax.bar(x + width/2, rouge_means, width, yerr=rouge_stds,
                    label="ROUGE-L", color="#dd8452", capsize=4, alpha=0.88)

# corpus-wide average lines
ax.axhline(r["traditional"]["bleu_avg"],    color="#4c72b0", linewidth=1.4,
           linestyle="--", alpha=0.7, label=f"BLEU avg ({r['traditional']['bleu_avg']:.3f})")
ax.axhline(r["traditional"]["rouge_l_avg"], color="#dd8452", linewidth=1.4,
           linestyle="--", alpha=0.7, label=f"ROUGE-L avg ({r['traditional']['rouge_l_avg']:.3f})")

# value labels on bars
for bar in [*bars_bleu, *bars_rouge]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([DOC_LABELS.get(d, d) for d in doc_sets], fontsize=7)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("BLEU & ROUGE-L per Document Set", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
out = "benchmark/plot_per_doc.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
