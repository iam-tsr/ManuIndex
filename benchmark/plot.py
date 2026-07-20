import json
import math
import os
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BENCHMARK_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPORT_PATH = BENCHMARK_DIR / "ragmix_report.json"  # neural_bridge_report.json # ragmix_report.json
OUTPUT_DIR = BENCHMARK_DIR / "plots/ragmix"  # plots/neural_bridge # plots/ragmix

METHOD_ORDER = [
    "grag",
    "naive_rag",
    "flat_hybrid_rag",
    "hierarchical_rag",
    "parent_child_rag",
    "query_rewrite_rag",
]

METHOD_LABELS = {
    "grag": "GRAG",
    "naive_rag": "Naive",
    "flat_hybrid_rag": "Flat Hybrid",
    "hierarchical_rag": "Hierarchical",
    "parent_child_rag": "Parent-Child",
    "query_rewrite_rag": "Query Rewrite",
}

METHOD_MARKERS = {
    "grag": "*",
    "naive_rag": "o",
    "flat_hybrid_rag": "s",
    "hierarchical_rag": "^",
    "parent_child_rag": "D",
    "query_rewrite_rag": "P",
}

BASELINE_COLOR = "#5B6472"
GRAG_COLOR = "#C2410C"
GRID_COLOR = "#D4D8DD"
AXIS_COLOR = "#2B2F36"
PAPER_FACE = "#FBFCFD"

REPORT_METHOD_MAP = {
    "grag_report": "grag",
    "rag_report": "naive_rag",
    "hybrid_rag_report": "flat_hybrid_rag",
    "flat_hybrid_rag_report": "flat_hybrid_rag",
    "hierarchical_rag_report": "hierarchical_rag",
    "parent_child_rag_report": "parent_child_rag",
    "query_rewrite_rag_report": "query_rewrite_rag",
}

SCORE_KEYS = frozenset({"f1", "faithfulness", "context_recall"})


def load_records(report_path: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    if not report_path.exists():
        raise FileNotFoundError(f"Could not find benchmark report: {report_path}")

    with report_path.open("r", encoding="utf-8") as handle:
        report_data = json.load(handle)

    records: list[dict] = []
    panel_order: list[tuple[str, str]] = []

    for embedding_name, embedding_group in report_data.items():
        if not isinstance(embedding_group, dict):
            continue

        for llm_name, llm_group in embedding_group.items():
            if not isinstance(llm_group, dict):
                continue

            raw_records = llm_group.get("benchmark") or []
            if not raw_records:
                continue

            panel_key = (embedding_name, llm_name)
            panel_order.append(panel_key)

            method_records: dict[str, dict] = {}
            for raw_record in raw_records:
                method_name = REPORT_METHOD_MAP.get(raw_record.get("report_name"))
                if method_name is not None:
                    method_records[method_name] = raw_record

            for method_name in METHOD_ORDER:
                method_metrics = method_records.get(method_name)
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
                        "faithfulness": float(ragas["Faithfulness"]),
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


def metric_limits(
    records: list[dict], x_key: str, y_key: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    def padded_range(
        values: list[float],
        hard_min: float | None = None,
        hard_max: float | None = None,
    ) -> tuple[float, float]:
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

    x_values = [row[x_key] for row in records]
    y_values = [row[y_key] for row in records]

    x_range = (
        padded_range(x_values, hard_min=0.0, hard_max=1.0)
        if x_key in SCORE_KEYS
        else padded_range(x_values)
    )
    y_range = (
        padded_range(y_values, hard_min=0.0, hard_max=1.0)
        if y_key in SCORE_KEYS
        else padded_range(y_values)
    )
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
            axis.scatter(
                row[x_key],
                row[y_key],
                s=260 if row["is_grag"] else 80,
                marker=METHOD_MARKERS[row["method"]],
                color=GRAG_COLOR if row["is_grag"] else BASELINE_COLOR,
                edgecolors="#7C2D12" if row["is_grag"] else "white",
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
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
    )
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

    plot_comparison(
        records,
        panel_order,
        x_key="total_time_seconds",
        y_key="f1",
        x_label="Average End-to-End Time per Question (s)",
        y_label="F1",
        title="RAG Family vs GRAG: Time vs F1",
        subtitle="Higher and farther left is better. Compares end-to-end latency with retrieval F1.",
        output_stem="time_vs_f1",
    )

    plot_comparison(
        records,
        panel_order,
        x_key="total_tokens",
        y_key="faithfulness",
        x_label="Average Total Tokens per Question",
        y_label="Faithfulness",
        title="RAG Family vs GRAG: Tokens vs Faithfulness",
        subtitle="Higher and farther left is better. Compares token cost with answer faithfulness.",
        output_stem="tokens_vs_faithfulness",
    )


if __name__ == "__main__":
    main()
