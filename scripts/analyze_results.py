"""
Analysis and visualisation of federated learning experiment results.

Loads results from a JSON file produced by run_experiments.py and generates:
  - Convergence curves (accuracy vs. round)
  - Robustness comparison bar charts
  - SHAP stability plots
  - Adversarial accuracy curves (if adversarial results are present)
  - Per-method comparison tables (saved as PNG)

Outputs are written to results/figures/ and results/tables/.

Usage
-----
    python scripts/analyze_results.py --results results/experiment_results.json
    python scripts/analyze_results.py --results results/experiment_results.json \\
        --output results --format pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Figures will not be generated.")


# ---------------------------------------------------------------------------
# Colour palette and style
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "fedavg":  "#4C72B0",
    "krum":    "#DD8452",
    "trimmed": "#55A868",
    "fedprox": "#C44E52",
    "advfl":   "#8172B2",
    "ztafl":   "#937860",
}

METHOD_LABELS = {
    "fedavg":  "FedAvg",
    "krum":    "Krum",
    "trimmed": "Trimmed Mean",
    "fedprox": "FedProx",
    "advfl":   "Adv-FL",
    "ztafl":   "ZTA-FL (ours)",
}

DATASET_LABELS = {
    "Edge-IIoTset": "Edge-IIoTset",
    "CIC-IDS2017":  "CIC-IDS2017",
    "UNSW-NB15":    "UNSW-NB15",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> List[Dict[str, Any]]:
    """Load experiment results from a JSON file."""
    with open(path) as f:
        return json.load(f)


def group_results(
    results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Group results by (dataset, method).

    Returns
    -------
    dict
        ``grouped[dataset][method]`` → list of result dicts (one per seed).
    """
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["dataset"]][r["method"]].append(r)
    return grouped


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_fig(fig, path: str, fmt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = f"{path}.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Convergence curves
# ---------------------------------------------------------------------------

def plot_convergence(
    grouped: Dict[str, Dict[str, List[Dict]]],
    output_dir: str,
    fmt: str,
) -> None:
    """Plot accuracy vs. round for each dataset."""
    for dataset, methods in grouped.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False

        for method, runs in sorted(methods.items()):
            # Gather history across seeds
            histories: list[list[dict]] = [r.get("history", []) for r in runs]
            if not any(histories):
                continue

            # Align histories by round number
            round_acc: dict[int, list[float]] = defaultdict(list)
            for h in histories:
                for entry in h:
                    round_acc[entry["round"]].append(entry["accuracy"])

            rounds_sorted = sorted(round_acc.keys())
            mean_acc = [np.mean(round_acc[r]) for r in rounds_sorted]
            std_acc = [np.std(round_acc[r]) for r in rounds_sorted]

            color = METHOD_COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method)
            ax.plot(rounds_sorted, mean_acc, label=label, color=color, linewidth=1.8)
            ax.fill_between(
                rounds_sorted,
                np.array(mean_acc) - np.array(std_acc),
                np.array(mean_acc) + np.array(std_acc),
                alpha=0.12, color=color,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        _setup_axes(
            ax,
            title=f"Convergence — {DATASET_LABELS.get(dataset, dataset)}",
            xlabel="Communication Round",
            ylabel="Test Accuracy",
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ds_key = dataset.lower().replace("-", "_").replace(" ", "_")
        _save_fig(fig, os.path.join(output_dir, "figures", f"convergence_{ds_key}"), fmt)


# ---------------------------------------------------------------------------
# Final accuracy comparison bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(
    grouped: Dict[str, Dict[str, List[Dict]]],
    output_dir: str,
    fmt: str,
) -> None:
    """Bar chart comparing final accuracy across methods and datasets."""
    datasets = list(grouped.keys())
    all_methods = sorted({m for d in grouped.values() for m in d.keys()})

    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        methods_in = sorted(grouped[dataset].keys())
        means = []
        stds = []
        labels = []
        colors = []
        for method in methods_in:
            accs = [r["metrics"]["accuracy"] for r in grouped[dataset][method]]
            means.append(np.mean(accs))
            stds.append(np.std(accs))
            labels.append(METHOD_LABELS.get(method, method))
            colors.append(METHOD_COLORS.get(method, "#555555"))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        _setup_axes(ax, title=DATASET_LABELS.get(dataset, dataset),
                    xlabel="", ylabel="Test Accuracy")
        ax.set_ylim(0, 1.05)
        # Annotate bars with value
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{m:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("Final Test Accuracy by Method", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "figures", "accuracy_comparison"), fmt)


# ---------------------------------------------------------------------------
# Comparison table (saved as PNG)
# ---------------------------------------------------------------------------

def save_comparison_table(
    grouped: Dict[str, Dict[str, List[Dict]]],
    output_dir: str,
    fmt: str,
) -> None:
    """Save a per-dataset comparison table as a figure."""
    for dataset, methods in grouped.items():
        sorted_methods = sorted(methods.keys())
        rows = []
        for method in sorted_methods:
            runs = methods[method]
            accs = [r["metrics"]["accuracy"] for r in runs]
            f1s = [r["metrics"]["macro_f1"] for r in runs]
            rows.append([
                METHOD_LABELS.get(method, method),
                f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
                f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            ])

        col_labels = ["Method", "Accuracy (mean ± std)", "Macro-F1 (mean ± std)"]
        fig, ax = plt.subplots(figsize=(8, 1 + 0.4 * len(rows)))
        ax.axis("off")
        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        # Style header
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor("#2C3E50")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        # Alternate row shading
        for i in range(1, len(rows) + 1):
            bg = "#EBF5FB" if i % 2 == 0 else "#FDFEFE"
            for j in range(len(col_labels)):
                tbl[(i, j)].set_facecolor(bg)

        ds_key = dataset.lower().replace("-", "_").replace(" ", "_")
        ax.set_title(
            f"Results — {DATASET_LABELS.get(dataset, dataset)}",
            fontsize=11, fontweight="bold", pad=10
        )
        _save_fig(fig, os.path.join(output_dir, "tables", f"comparison_{ds_key}"), fmt)


# ---------------------------------------------------------------------------
# F1 comparison
# ---------------------------------------------------------------------------

def plot_f1_comparison(
    grouped: Dict[str, Dict[str, List[Dict]]],
    output_dir: str,
    fmt: str,
) -> None:
    """Horizontal bar chart of macro-F1 scores."""
    for dataset, methods in grouped.items():
        sorted_methods = sorted(methods.keys())
        f1_means, f1_stds, labels, colors = [], [], [], []
        for method in sorted_methods:
            f1s = [r["metrics"]["macro_f1"] for r in methods[method]]
            f1_means.append(np.mean(f1s))
            f1_stds.append(np.std(f1s))
            labels.append(METHOD_LABELS.get(method, method))
            colors.append(METHOD_COLORS.get(method, "#555555"))

        fig, ax = plt.subplots(figsize=(6, 0.6 * len(labels) + 2))
        y = np.arange(len(labels))
        ax.barh(y, f1_means, xerr=f1_stds, color=colors, capsize=3,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        _setup_axes(
            ax,
            title=f"Macro-F1 — {DATASET_LABELS.get(dataset, dataset)}",
            xlabel="Macro-F1 Score",
            ylabel="",
        )
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ds_key = dataset.lower().replace("-", "_").replace(" ", "_")
        _save_fig(fig, os.path.join(output_dir, "figures", f"macro_f1_{ds_key}"), fmt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse federated learning experiment results and generate figures."
    )
    parser.add_argument(
        "--results", type=str,
        default="results/experiment_results.json",
        help="Path to the JSON results file."
    )
    parser.add_argument(
        "--output", type=str,
        default="results",
        help="Root directory for saving figures and tables."
    )
    parser.add_argument(
        "--format", type=str, default="png", choices=["png", "pdf"],
        help="Output file format."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not HAS_MPL:
        print("Error: matplotlib is required for figure generation.")
        sys.exit(1)

    print(f"Loading results from {args.results} ...")
    results = load_results(args.results)
    print(f"  Loaded {len(results)} experiment records.")

    grouped = group_results(results)
    print(f"  Datasets: {list(grouped.keys())}")

    os.makedirs(os.path.join(args.output, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "tables"), exist_ok=True)

    print("Generating convergence curves ...")
    plot_convergence(grouped, args.output, args.format)

    print("Generating accuracy comparison charts ...")
    plot_accuracy_comparison(grouped, args.output, args.format)

    print("Generating F1 comparison charts ...")
    plot_f1_comparison(grouped, args.output, args.format)

    print("Saving comparison tables ...")
    save_comparison_table(grouped, args.output, args.format)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
