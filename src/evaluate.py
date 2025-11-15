# src/evaluate.py
"""Independent evaluation & visualisation script.

After all training runs have finished this script gathers their WandB
histories/summaries, dumps them to `results_dir`, produces per-run plots
(learning curves, confusion matrices), then aggregates metrics across
runs – including bar/box charts and statistical-significance tests.

CLI (matches GitHub Actions call):
    uv run python -m src.evaluate results_dir=<DIR> run_ids='["run-1", "run-2"]'
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import ConfusionMatrixDisplay

PRIMARY_METRIC_KEY = "Area Under Accuracy–Energy curve (AUAE) within 2 k steps."

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# CLI parsing (key=value overrides style)
# ---------------------------------------------------------------------------

def _parse_kv(argv: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Unexpected argument '{arg}'. Expected key=value.")
        k, v = arg.split("=", 1)
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_learning_curves(df, run_id: str, out_dir: Path, ylim=None, xlim=None) -> Path:
    import pandas as pd
    plt.figure(figsize=(8, 5), dpi=300)

    # Convert step to numeric
    step = pd.to_numeric(df["_step"], errors='coerce')

    plotted_any = False
    if "train_loss" in df.columns:
        # Convert to numeric and drop NaN values
        train_loss = pd.to_numeric(df["train_loss"], errors='coerce')
        valid_mask = ~(train_loss.isna() | step.isna())
        if valid_mask.any():
            sns.lineplot(x=step[valid_mask], y=train_loss[valid_mask], label="Training Loss", linewidth=2)
            plotted_any = True

    if "val_acc" in df.columns:
        # Convert to numeric and drop NaN values
        val_acc = pd.to_numeric(df["val_acc"], errors='coerce')
        valid_mask = ~(val_acc.isna() | step.isna())
        # Only plot if there are non-zero values
        if valid_mask.any() and val_acc[valid_mask].max() > 0:
            sns.lineplot(x=step[valid_mask], y=val_acc[valid_mask], label="Validation Accuracy", linewidth=2)
            plotted_any = True

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    # Shorten title for better readability
    short_title = run_id.replace("-iter1-Qwen3-0.6B-4-bit-QLoRA-gsm8k", "")
    plt.title(f"Learning Curves: {short_title}", fontsize=13, fontweight='bold')

    if plotted_any:
        plt.legend(fontsize=10, loc='best')

    # Apply consistent scales if provided
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def _plot_confusion_matrix(cm: np.ndarray, run_id: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=True, ax=ax, values_format='d')
    # Shorten title for better readability
    short_title = run_id.replace("-iter1-Qwen3-0.6B-4-bit-QLoRA-gsm8k", "")
    ax.set_title(f"Confusion Matrix: {short_title}", fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    fname = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def _plot_comparison_bar(metric_map: Dict[str, float], title: str, out_dir: Path) -> Path:
    run_ids = list(metric_map.keys())
    values = list(metric_map.values())

    # Shorten labels for better readability
    short_labels = [rid.replace("iter1-Qwen3-0.6B-4-bit-QLoRA-gsm8k", "").replace("-", " ").strip() for rid in run_ids]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar(range(len(short_labels)), values, color=['#2ecc71' if 'proposed' in rid else '#3498db' for rid in run_ids],
                  edgecolor='black', linewidth=1.2, alpha=0.8)

    # Add value labels on top of bars
    for i, (bar, v) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Run Configuration", fontsize=13, fontweight='bold')
    ax.set_title(f"Comparison: {title}", fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='Proposed'),
                       Patch(facecolor='#3498db', edgecolor='black', label='Comparative')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    fname = out_dir / f"comparison_{title.replace(' ', '_')}.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def _plot_comparison_box(metric_map: Dict[str, float], title: str, out_dir: Path) -> Path:
    categories = ["Proposed" if "proposed" in rid else "Comparative" for rid in metric_map]
    data = {
        "run_id": list(metric_map.keys()),
        title: list(metric_map.values()),
        "Category": categories,
    }

    import pandas as pd
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Custom colors for the boxes
    colors = {'Proposed': '#2ecc71', 'Comparative': '#3498db'}
    sns.boxplot(x="Category", y=title, data=df, palette=colors, ax=ax, linewidth=2)
    sns.stripplot(x="Category", y=title, data=df, color="black", alpha=0.7, size=8, ax=ax)

    ax.set_ylabel(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Method", fontsize=13, fontweight='bold')
    ax.set_title(f"Distribution Comparison: {title}", fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = out_dir / f"comparison_{title.replace(' ', '_')}_box.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    kv = _parse_kv(sys.argv[1:])
    if "results_dir" not in kv or "run_ids" not in kv:
        raise SystemExit('Usage: python -m src.evaluate results_dir=<DIR> run_ids=\'["run",...]\'')

    results_dir = Path(kv["results_dir"]).expanduser().resolve()
    run_ids: List[str] = json.loads(kv["run_ids"])

    # load global wandb config ------------------------------------------------
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    entity = cfg.wandb.entity
    project = cfg.wandb.project

    api = wandb.Api()

    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    primary_metric_values: Dict[str, float] = {}

    per_run_files: List[Path] = []
    comparison_files: List[Path] = []

    # First pass: collect all data and determine global scales
    import pandas as pd
    all_histories = {}
    all_summaries = {}
    all_configs = {}
    global_y_min, global_y_max = float('inf'), float('-inf')
    global_x_max = 0

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()  # pd.DataFrame
        summary = run.summary._json_dict
        cfg_run = dict(run.config)

        all_histories[run_id] = history
        all_summaries[run_id] = summary
        all_configs[run_id] = cfg_run

        # Calculate global scales
        step = pd.to_numeric(history["_step"], errors='coerce')
        if "train_loss" in history.columns:
            train_loss = pd.to_numeric(history["train_loss"], errors='coerce')
            valid_mask = ~(train_loss.isna() | step.isna())
            if valid_mask.any():
                global_y_min = min(global_y_min, train_loss[valid_mask].min())
                global_y_max = max(global_y_max, train_loss[valid_mask].max())
                global_x_max = max(global_x_max, step[valid_mask].max())

    # Add some padding to y-axis
    y_range = global_y_max - global_y_min
    ylim = (global_y_min - 0.05 * y_range, global_y_max + 0.05 * y_range)
    xlim = (0, global_x_max * 1.02)

    # Second pass: create plots with consistent scales
    for run_id in run_ids:
        history = all_histories[run_id]
        summary = all_summaries[run_id]
        cfg_run = all_configs[run_id]

        run_out = results_dir / run_id
        run_out.mkdir(parents=True, exist_ok=True)

        # dump metrics --------------------------------------------------------
        history_json = history.to_dict(orient="list")
        mfile = run_out / "metrics.json"
        with open(mfile, "w", encoding="utf-8") as fp:
            json.dump({"history": history_json, "summary": summary, "config": cfg_run}, fp, indent=2)
        per_run_files.append(mfile)

        # plots ----------------------------------------------------------------
        lc_path = _plot_learning_curves(history, run_id, run_out, ylim=ylim, xlim=xlim)
        per_run_files.append(lc_path)

        if "confusion_matrix" in summary:
            cm = np.array(summary["confusion_matrix"], dtype=int)
            cm_path = _plot_confusion_matrix(cm, run_id, run_out)
            per_run_files.append(cm_path)

        # collect metrics ------------------------------------------------------
        if PRIMARY_METRIC_KEY in summary:
            primary_metric_values[run_id] = float(summary[PRIMARY_METRIC_KEY])
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                aggregated_metrics.setdefault(k, {})[run_id] = float(v)

    # -----------------------------------------------------------------------
    # aggregated analysis
    # -----------------------------------------------------------------------
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # bar chart
    bar_path = _plot_comparison_bar(primary_metric_values, "AUAE", comp_dir)
    comparison_files.append(bar_path)
    # box plot
    box_path = _plot_comparison_box(primary_metric_values, "AUAE", comp_dir)
    comparison_files.append(box_path)

    # significance tests ------------------------------------------------------
    prop_vals = [v for k, v in primary_metric_values.items() if "proposed" in k]
    base_vals = [v for k, v in primary_metric_values.items() if ("baseline" in k or "comparative" in k)]
    t_stat, p_val = (np.nan, np.nan)
    u_stat, p_u = (np.nan, np.nan)
    if len(prop_vals) >= 2 and len(base_vals) >= 2:
        t_stat, p_val = ttest_ind(prop_vals, base_vals, equal_var=False)
        u_stat, p_u = mannwhitneyu(prop_vals, base_vals, alternative="two-sided")

    sig_file = comp_dir / "comparison_significance_tests.txt"
    with open(sig_file, "w", encoding="utf-8") as fp:
        fp.write("Two-sample t-test (Welch):\n")
        fp.write(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}\n")
        fp.write("Mann–Whitney U test:\n")
        fp.write(f"  U-statistic = {u_stat:.4f}, p-value = {p_u:.4e}\n")
    comparison_files.append(sig_file)

    # best runs & gap ---------------------------------------------------------
    best_prop = {
        "run_id": max((rid for rid in primary_metric_values if "proposed" in rid), key=lambda r: primary_metric_values[r]),
        "value": 0.0,
    }
    best_prop["value"] = primary_metric_values[best_prop["run_id"]]

    best_base = {
        "run_id": max((rid for rid in primary_metric_values if "baseline" in rid or "comparative" in rid), key=lambda r: primary_metric_values[r]),
        "value": 0.0,
    }
    best_base["value"] = primary_metric_values[best_base["run_id"]]

    gap = (best_prop["value"] - best_base["value"]) / max(best_base["value"], 1e-8) * 100.0

    aggregated_json = {
        "primary_metric": PRIMARY_METRIC_KEY,
        "metrics": aggregated_metrics,
        "best_proposed": best_prop,
        "best_baseline": best_base,
        "gap": gap,
        "statistics": {
            "t_test": {"t": t_stat, "p": p_val},
            "mann_whitney": {"u": u_stat, "p": p_u},
        },
    }

    agg_file = comp_dir / "aggregated_metrics.json"
    with open(agg_file, "w", encoding="utf-8") as fp:
        json.dump(aggregated_json, fp, indent=2)
    comparison_files.append(agg_file)

    # -----------------------------------------------------------------------
    # stdout report (paths)
    # -----------------------------------------------------------------------
    for p in per_run_files + comparison_files:
        print(p)


if __name__ == "__main__":
    main()