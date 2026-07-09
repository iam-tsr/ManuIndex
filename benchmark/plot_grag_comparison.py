import json
import math
import os
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BENCHMARK_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPORT_PATH = BENCHMARK_DIR / "final_report.json"
OUTPUT_DIR = BENCHMARK_DIR / "plots"

METHOD_ORDER = [
    "grag",
    "naive_rag",
    "flat_hybrid_rag",
    "hierarchical_rag",
    "parent_child_rag",
    "query_rewrite_rag",
    "graph_rag",
]

METHOD_LABELS = {
    "grag": "GRAG",
    "naive_rag": "Naive",
    "flat_hybrid_rag": "Flat Hybrid",
    "hierarchical_rag": "Hierarchical",
    "parent_child_rag": "Parent-Child",
    "query_rewrite_rag": "Query Rewrite",
    "graph_rag": "Graph RAG",
}

METHOD_MARKERS = {
    "grag": "*",
    "naive_rag": "o",
    "flat_hybrid_rag": "s",
    "hierarchical_rag": "^",
    "parent_child_rag": "D",
    "query_rewrite_rag": "P",
    "graph_rag": "X",
}

BASELINE_COLOR = "#5B6472"
GRAG_COLOR = "#C2410C"
GRID_COLOR = "#D4D8DD"
AXIS_COLOR = "#2B2F36"
PAPER_FACE = "#FBFCFD"


def load_records(report_path: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    if not report_path.exists():
        raise FileNotFoundError(f"Could not find benchmark report: {report_path}")

    with report_path.open("r", encoding="utf-8") as handle:
        report_data = json.load(handle)

    records: list[dict] = []
    panel_order: list[tuple[str, str]] = []

    for embedding_name, llm_groups in report_data.items():
        for llm_name, method_groups in llm_groups.items():
            panel_key = (embedding_name, llm_name)
            panel_order.append(panel_key)

            for method_name in METHOD_ORDER:
                method_metrics = method_groups.get(method_name)
                if method_metrics is None:
                    continue

                runtime = method_metrics["runtime"]
                cost = method_metrics["cost"]
                ragas = method_metrics["ragas"]

                records.append(
                    {
                        "embedding": embedding_name,
                        "llm": llm_name,
                        "panel_key": panel_key,
                        "method": method_name,
                        "label": METHOD_LABELS[method_name],
                        "is_grag": method_name == "grag",
                        "f1": float(ragas["F1"]),
                        "context_recall": float(ragas["Context Recall"]),
                        "total_tokens": float(cost["average_total_tokens"]),
                        "total_time_seconds": float(
                            runtime["average_retrieval_time_seconds"]
                            + runtime["average_answer_time_seconds"]
                        ),
                    }
                )

    if not records:
        raise ValueError(f"No benchmark entries were found in {report_path}")

    return records, panel_order


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PAPER_FACE,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": AXIS_COLOR,
            "axes.titlecolor": AXIS_COLOR,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.8,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )


def metric_limits(records: list[dict], x_key: str, y_key: str) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = [row[x_key] for row in records]
    y_values = [row[y_key] for row in records]

    def padded_range(values: list[float], hard_min: float | None = None, hard_max: float | None = None) -> tuple[float, float]:
        minimum = min(values)
        maximum = max(values)
        span = maximum - minimum
        padding = span * 0.12 if span else max(abs(maximum) * 0.1, 0.05)
        lower = minimum - padding
        upper = maximum + padding

        if hard_min is not None:
            lower = max(hard_min, lower)
        if hard_max is not None:
            upper = min(hard_max, upper)
        return lower, upper

    if x_key == "context_recall":
        x_range = padded_range(x_values, hard_min=0.0, hard_max=1.0)
    else:
        x_range = padded_range(x_values)

    if y_key == "f1":
        y_range = padded_range(y_values, hard_min=0.0, hard_max=1.0)
    else:
        y_range = padded_range(y_values)

    return x_range, y_range


def plot_comparison(
    records: list[dict],
    panel_order: list[tuple[str, str]],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
    output_stem: str,
) -> None:
    x_limits, y_limits = metric_limits(records, x_key, y_key)
    n_panels = len(panel_order)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(16, 5.4 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for index, panel_key in enumerate(panel_order):
        axis = axes_list[index]
        panel_records = [row for row in records if row["panel_key"] == panel_key]
        panel_records.sort(key=lambda row: METHOD_ORDER.index(row["method"]))

        grag_record = next(row for row in panel_records if row["is_grag"])

        axis.axvline(grag_record[x_key], color=GRAG_COLOR, linestyle="--", linewidth=1.1, alpha=0.45, zorder=1)
        axis.axhline(grag_record[y_key], color=GRAG_COLOR, linestyle="--", linewidth=1.1, alpha=0.45, zorder=1)

        for row in panel_records:
            marker = METHOD_MARKERS[row["method"]]
            size = 260 if row["is_grag"] else 80
            color = GRAG_COLOR if row["is_grag"] else BASELINE_COLOR
            edge = "#7C2D12" if row["is_grag"] else "white"

            axis.scatter(
                row[x_key],
                row[y_key],
                s=size,
                marker=marker,
                color=color,
                edgecolors=edge,
                linewidths=0.9,
                alpha=0.98,
                zorder=3,
            )

        axis.set_title(f"{panel_key[0]}\n{panel_key[1]}", pad=10)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(True, which="major")

    for axis in axes_list[n_panels:]:
        axis.set_visible(False)

    for row_index in range(nrows):
        axes_list[row_index * ncols].set_ylabel(y_label)
    for col_index in range(min(ncols, n_panels)):
        axes_list[(nrows - 1) * ncols + col_index].set_xlabel(x_label)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method_name],
            color="none",
            markerfacecolor=GRAG_COLOR if method_name == "grag" else BASELINE_COLOR,
            markeredgecolor="#7C2D12" if method_name == "grag" else "white",
            markersize=14 if method_name == "grag" else 9,
            label=METHOD_LABELS[method_name],
        )
        for method_name in METHOD_ORDER
    ]
    legend_handles.append(
        Line2D([0], [0], color=GRAG_COLOR, linestyle="--", linewidth=1.1, alpha=0.45, label="GRAG reference lines")
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.08, ha="left", y=0.97)
    fig.text(0.08, 0.935, subtitle, fontsize=10, color=AXIS_COLOR, ha="left")
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2, columnspacing=1.2, handletextpad=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_metric_bars(
    records: list[dict],
    *,
    metric_key: str,
    x_label: str,
    title: str,
    subtitle: str,
    output_stem: str,
) -> None:
    averages: list[dict] = []
    for method_name in METHOD_ORDER:
        method_rows = [row for row in records if row["method"] == method_name]
        if not method_rows:
            continue
        averages.append(
            {
                "method": method_name,
                "label": METHOD_LABELS[method_name],
                "is_grag": method_name == "grag",
                "value": sum(row[metric_key] for row in method_rows) / len(method_rows),
            }
        )

    max_metric = max(row["value"] for row in averages)
    fig, axis = plt.subplots(figsize=(11.5, 6.8))

    y_positions = list(range(len(averages)))
    y_labels = [row["label"] for row in averages]

    for position, row in zip(y_positions, averages):
        color = GRAG_COLOR if row["is_grag"] else BASELINE_COLOR
        alpha = 0.95 if row["is_grag"] else 0.85
        axis.barh(position, row["value"], color=color, alpha=alpha, height=0.62, zorder=2)

    axis.set_xlim(0, max_metric * 1.12)
    axis.set_xlabel(x_label)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(y_labels)
    axis.invert_yaxis()
    axis.grid(True, axis="x", which="major")
    axis.set_axisbelow(True)

    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.08, ha="left", y=0.98)
    fig.text(0.08, 0.925, subtitle, fontsize=10, color=AXIS_COLOR, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    configure_style()
    records, panel_order = load_records(REPORT_PATH)

    plot_metric_bars(
        records,
        metric_key="total_time_seconds",
        x_label="Average End-to-End Time per Question (s)",
        title="RAG Family vs GRAG: Overall Efficiency Comparison",
        subtitle="Shorter bars are better. Values are averaged across all benchmark settings.",
        output_stem="rag_vs_grag_efficiency",
    )

    plot_comparison(
        records,
        panel_order,
        x_key="total_tokens",
        y_key="f1",
        x_label="Average Total Tokens per Question",
        y_label="F1 Score",
        title="RAG Family vs GRAG: Cost-Quality Tradeoff",
        subtitle="Upper-left is better. This view combines token cost and answer quality in one plot.",
        output_stem="rag_vs_grag_cost_quality",
    )

if __name__ == "__main__":
    main()
