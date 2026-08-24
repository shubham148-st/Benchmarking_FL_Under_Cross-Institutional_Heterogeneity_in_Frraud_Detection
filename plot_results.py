"""
IEEE-Quality Visualization Suite for FL Fraud Detection Benchmark.

Generates two types of publication-ready figures:

1. **Results Summary Plots** (from results_*.csv)
   - Grouped bar charts comparing algorithms across metrics per stage
   - Cross-stage comparison heatmaps

2. **Training Convergence Plots** (from history_*.csv)
   - Per-round metric evolution over 50 FL rounds
   - Multi-metric subplot grids

Usage:
    python plot_results.py                          # Plot all available data
    python plot_results.py --stage stage1_natural   # Plot specific stage
    python plot_results.py --history-only           # Only convergence plots
    python plot_results.py --results-only           # Only summary plots

Output:
    figures/  directory with high-res PDF/PNG files
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════════════════
# IEEE-Style Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# IEEE column widths (inches): single=3.5, double=7.16
SINGLE_COL = 3.5
DOUBLE_COL = 7.16

# Professional color palette (colorblind-friendly, IEEE-safe)
ALGO_COLORS = {
    "fedavg":   "#2E86AB",   # Steel blue
    "fedprox":  "#A23B72",   # Plum
    "scaffold": "#F18F01",   # Amber
    "ditto":    "#C73E1D",   # Vermillion
}

ALGO_MARKERS = {
    "fedavg":   "o",
    "fedprox":  "s",
    "scaffold": "^",
    "ditto":    "D",
}

ALGO_LINESTYLES = {
    "fedavg":   "-",
    "fedprox":  "--",
    "scaffold": "-.",
    "ditto":    ":",
}

ALGO_LABELS = {
    "fedavg":   "FedAvg",
    "fedprox":  "FedProx",
    "scaffold": "SCAFFOLD",
    "ditto":    "Ditto",
}

STAGE_LABELS = {
    "stage1_natural":    "Stage 1: Natural",
    "stage2_balanced":   "Stage 2: Balanced",
    "stage3_dirichlet":  "Stage 3: Dirichlet",
}

METRIC_LABELS = {
    "auc_pr":           "AUC-PR",
    "f1_at_1pct":       "F1 @ 1%",
    "f1_at_5pct":       "F1 @ 5%",
    "precision_at_1pct": "Precision @ 1%",
    "recall_at_1pct":   "Recall @ 1%",
    "loss":             "Loss",
}


def setup_ieee_style():
    """Configure matplotlib for IEEE-quality figures."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        # Axes
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",

        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",

        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all results_*.csv files, keyed by stage name."""
    results = {}
    for fpath in sorted(results_dir.glob("results_stage*.csv")):
        if "copy" in fpath.stem.lower():
            continue
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        stage = df["stage"].iloc[0] if "stage" in df.columns else fpath.stem.replace("results_", "")
        results[stage] = df
    return results


def load_histories(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all history_*.csv files, keyed by 'stage_algo'."""
    histories = {}
    for fpath in sorted(results_dir.rglob("history_*.csv")):
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        key = fpath.stem.replace("history_", "")
        histories[key] = df
    return histories


def group_histories_by_stage(histories: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Group and concatenate histories by stage."""
    stage_data = {}
    for key, df in histories.items():
        # key format: stage1_natural_fedavg
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            stage_key = parts[0]
        else:
            stage_key = key
        if stage_key not in stage_data:
            stage_data[stage_key] = []
        stage_data[stage_key].append(df)

    return {stage: pd.concat(dfs, ignore_index=True) for stage, dfs in stage_data.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Results Summary -- Grouped Bar Charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results_bars(results: dict[str, pd.DataFrame], output_dir: Path):
    """
    Generate grouped bar charts comparing algorithms for each stage.
    One figure per stage with subplots for key metrics.
    """
    metrics = ["auc_pr", "f1_at_1pct", "f1_at_5pct", "precision_at_1pct", "recall_at_1pct"]

    for stage, df in results.items():
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(DOUBLE_COL, 2.2), sharey=False)

        if n_metrics == 1:
            axes = [axes]

        algorithms = df["algorithm"].tolist()
        x = np.arange(len(algorithms))
        bar_width = 0.6

        for ax, metric in zip(axes, metrics):
            values = df[metric].values
            colors = [ALGO_COLORS.get(a, "#888888") for a in algorithms]

            bars = ax.bar(x, values, bar_width, color=colors, edgecolor="white",
                         linewidth=0.5, zorder=3)

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=5.5,
                       fontweight="medium")

            ax.set_title(METRIC_LABELS.get(metric, metric), fontweight="bold", pad=4)
            ax.set_xticks(x)
            ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in algorithms],
                              rotation=35, ha="right", fontsize=6)
            ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 0.1)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        stage_label = STAGE_LABELS.get(stage, stage)
        fig.suptitle(f"Algorithm Comparison -- {stage_label}",
                    fontsize=10, fontweight="bold", y=1.02)
        fig.tight_layout()

        out_path = output_dir / f"results_bars_{stage}.png"
        fig.savefig(out_path, format="png")
        plt.close(fig)
        print(f"  [OK] Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Cross-Stage Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cross_stage_heatmap(results: dict[str, pd.DataFrame], output_dir: Path):
    """
    Generate a heatmap showing algorithm × stage performance.
    Only created if multiple stages are available.
    """
    if len(results) < 2:
        print("  [!] Skipping cross-stage heatmap (need ≥2 stages)")
        return

    metric = "auc_pr"

    # Build pivot table
    all_rows = []
    for stage, df in results.items():
        for _, row in df.iterrows():
            all_rows.append({
                "Algorithm": ALGO_LABELS.get(row["algorithm"], row["algorithm"]),
                "Stage": STAGE_LABELS.get(stage, stage),
                metric: row[metric],
            })

    pivot_df = pd.DataFrame(all_rows)
    pivot = pivot_df.pivot(index="Algorithm", columns="Stage", values=metric)

    # Reorder
    algo_order = [v for v in ALGO_LABELS.values() if v in pivot.index]
    stage_order = [v for v in STAGE_LABELS.values() if v in pivot.columns]
    pivot = pivot.reindex(index=algo_order, columns=stage_order)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))

    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("AUC-PR", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=6.5, rotation=20, ha="right")
    ax.set_yticklabels(pivot.index, fontsize=7)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                       fontsize=6, fontweight="bold", color=text_color)

    ax.set_title("AUC-PR: Algorithm × Stage", fontsize=9, fontweight="bold", pad=6)
    fig.tight_layout()

    out_path = output_dir / "cross_stage_heatmap.png"
    fig.savefig(out_path, format="png")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Convergence Curves (Training History)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence_curves(stage_histories: dict[str, pd.DataFrame], output_dir: Path):
    """
    Generate per-stage convergence plots showing metrics over 50 FL rounds.
    2×2 subplot grid: AUC-PR, F1@1%, Loss, Recall@1%.
    """
    metrics_to_plot = [
        ("auc_pr",          "AUC-PR",        False),
        ("f1_at_1pct",      "F1 @ 1%",       False),
        ("loss",            "Aggregated Loss", True),
        ("recall_at_1pct",  "Recall @ 1%",   False),
    ]

    for stage, df in stage_histories.items():
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.0))
        axes = axes.flatten()

        algorithms = df["algorithm"].unique()

        for ax, (metric, label, invert) in zip(axes, metrics_to_plot):
            if metric not in df.columns:
                ax.set_visible(False)
                continue

            for algo in sorted(algorithms):
                algo_df = df[df["algorithm"] == algo].sort_values("round")
                rounds = algo_df["round"].values
                values = algo_df[metric].values

                if invert:
                    values = -values  # Loss is stored as negative

                ax.plot(
                    rounds, values,
                    color=ALGO_COLORS.get(algo, "#888"),
                    linestyle=ALGO_LINESTYLES.get(algo, "-"),
                    marker=ALGO_MARKERS.get(algo, "o"),
                    markevery=5,
                    markersize=3.5,
                    label=ALGO_LABELS.get(algo, algo),
                    zorder=3,
                )

            ax.set_xlabel("Communication Round", fontsize=7)
            ax.set_ylabel(label, fontsize=7)
            ax.set_title(label, fontsize=8, fontweight="bold", pad=4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(0, rounds.max() + 1)

        # Single legend at bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(algorithms),
                  frameon=True, bbox_to_anchor=(0.5, -0.02), fontsize=7,
                  columnspacing=1.5, handletextpad=0.5)

        stage_label = STAGE_LABELS.get(stage, stage)
        fig.suptitle(f"Training Convergence -- {stage_label}",
                    fontsize=10, fontweight="bold", y=1.01)
        fig.tight_layout(rect=[0, 0.04, 1, 0.98])

        out_path = output_dir / f"convergence_{stage}.png"
        fig.savefig(out_path, format="png")
        plt.close(fig)
        print(f"  [OK] Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Radar / Spider Chart (Per-Stage Algorithm Comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radar_chart(results: dict[str, pd.DataFrame], output_dir: Path):
    """
    Generate radar charts comparing algorithm profiles per stage.
    Shows multi-dimensional performance at a glance.
    """
    metrics = ["auc_pr", "f1_at_1pct", "f1_at_5pct", "precision_at_1pct", "recall_at_1pct"]
    metric_labels = [METRIC_LABELS.get(m, m) for m in metrics]
    n_metrics = len(metrics)

    for stage, df in results.items():
        if len(df) < 2:
            continue

        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL),
                              subplot_kw=dict(polar=True))

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        # Normalize values to [0, 1] for radar
        max_vals = df[metrics].max().values
        max_vals[max_vals == 0] = 1  # Avoid division by zero

        for _, row in df.iterrows():
            algo = row["algorithm"]
            values = (row[metrics].values / max_vals).tolist()
            values += values[:1]

            ax.plot(angles, values,
                   color=ALGO_COLORS.get(algo, "#888"),
                   linestyle=ALGO_LINESTYLES.get(algo, "-"),
                   linewidth=1.3, label=ALGO_LABELS.get(algo, algo))
            ax.fill(angles, values,
                   color=ALGO_COLORS.get(algo, "#888"),
                   alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=6.5)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.15)

        # Concentric grid circles
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_rlabel_position(30)
        ax.tick_params(axis="y", labelsize=5, colors="gray")

        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15),
                 fontsize=6.5, frameon=True)

        stage_label = STAGE_LABELS.get(stage, stage)
        ax.set_title(f"Performance Profile -- {stage_label}",
                    fontsize=9, fontweight="bold", pad=15)

        fig.tight_layout()
        out_path = output_dir / f"radar_{stage}.png"
        fig.savefig(out_path, format="png")
        plt.close(fig)
        print(f"  [OK] Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Final Metrics Comparison Table (LaTeX-ready)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_latex_table(results: dict[str, pd.DataFrame], output_dir: Path):
    """
    Generate a LaTeX-formatted table for direct inclusion in IEEE papers.
    Bolds the best value per metric per stage.
    """
    metrics = ["auc_pr", "f1_at_1pct", "f1_at_5pct", "precision_at_1pct", "recall_at_1pct"]
    col_labels = ["AUC-PR", "F1@1\\%", "F1@5\\%", "Prec@1\\%", "Rec@1\\%"]

    lines = []
    lines.append("% Auto-generated LaTeX table -- FL Fraud Detection Benchmark")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Final-round evaluation metrics across FL algorithms and stages.}")
    lines.append("\\label{tab:results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ll" + "r" * len(metrics) + "}")
    lines.append("\\toprule")
    lines.append("Stage & Algorithm & " + " & ".join(col_labels) + " \\\\")
    lines.append("\\midrule")

    for stage, df in results.items():
        stage_label = STAGE_LABELS.get(stage, stage).replace("Stage ", "S")
        best = {m: df[m].max() for m in metrics}

        for i, (_, row) in enumerate(df.iterrows()):
            algo = ALGO_LABELS.get(row["algorithm"], row["algorithm"])
            vals = []
            for m in metrics:
                v = row[m]
                s = f"{v:.4f}"
                if v == best[m]:
                    s = f"\\textbf{{{s}}}"
                vals.append(s)

            prefix = stage_label if i == 0 else ""
            lines.append(f"{prefix} & {algo} & " + " & ".join(vals) + " \\\\")

        lines.append("\\midrule")

    # Remove last midrule, replace with bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = output_dir / "results_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [OK] Saved: {tex_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate IEEE-quality figures for FL Fraud Detection Benchmark",
    )
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing CSV results")
    parser.add_argument("--output-dir", type=str, default="figures",
                       help="Directory to save generated figures")
    parser.add_argument("--stage", type=str, default=None,
                       help="Filter to specific stage (e.g. stage1_natural)")
    parser.add_argument("--results-only", action="store_true",
                       help="Only generate results summary plots")
    parser.add_argument("--history-only", action="store_true",
                       help="Only generate convergence history plots")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_ieee_style()

    print(f"\n{'='*60}")
    print(f"  IEEE Figure Generator -- FL Fraud Detection Benchmark")
    print(f"{'='*60}")
    print(f"  Source:  {results_dir}")
    print(f"  Output:  {output_dir}\n")

    # ── Results plots ──
    if not args.history_only:
        print("[*] Generating results summary plots...")
        results = load_results(results_dir)
        if args.stage:
            results = {k: v for k, v in results.items() if args.stage in k}

        if results:
            plot_results_bars(results, output_dir)
            plot_cross_stage_heatmap(results, output_dir)
            plot_radar_chart(results, output_dir)
            generate_latex_table(results, output_dir)
        else:
            print("  [!] No results CSV files found.")

    # ── History / convergence plots ──
    if not args.results_only:
        print("\n[*] Generating convergence plots...")
        histories = load_histories(results_dir)
        if args.stage:
            histories = {k: v for k, v in histories.items() if args.stage in k}

        if histories:
            stage_histories = group_histories_by_stage(histories)
            plot_convergence_curves(stage_histories, output_dir)
        else:
            print("  [!] No history CSV files found.")

    print(f"\n[OK] All figures saved to: {output_dir}/\n")


if __name__ == "__main__":
    main()
