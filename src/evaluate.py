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

def _plot_learning_curves(df, run_id: str, out_dir: Path) -> Path:
    plt.figure(figsize=(6, 4))
    if "train_loss" in df.columns:
        sns.lineplot(x=df["_step"], y=df["train_loss"], label="train_loss")
    if "val_acc" in df.columns:
        sns.lineplot(x=df["_step"], y=df["val_acc"], label="val_acc")
    plt.xlabel("Step")
    plt.ylabel("Metric value")
    plt.title(run_id)
    plt.legend()
    plt.tight_layout()
    fname = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def _plot_confusion_matrix(cm: np.ndarray, run_id: str, out_dir: Path) -> Path:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=True)
    plt.title(f"{run_id} – Confusion Matrix")
    plt.tight_layout()
    fname = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def _plot_comparison_bar(metric_map: Dict[str, float], title: str, out_dir: Path) -> Path:
    run_ids = list(metric_map.keys())
    values = list(metric_map.values())
    plt.figure(figsize=(max(4, len(run_ids) * 0.6), 4))
    sns.barplot(x=run_ids, y=values, palette="Set2")
    plt.ylabel(title)
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    fname = out_dir / f"comparison_{title.replace(' ', '_')}.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def _plot_comparison_box(metric_map: Dict[str, float], title: str, out_dir: Path) -> Path:
    categories = ["proposed" if "proposed" in rid else "baseline" for rid in metric_map]
    data = {
        "run_id": list(metric_map.keys()),
        title: list(metric_map.values()),
        "category": categories,
    }
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="category", y=title, data=data, palette="pastel")
    sns.stripplot(x="category", y=title, data=data, color="black", alpha=0.6)
    plt.tight_layout()
    fname = out_dir / f"comparison_{title.replace(' ', '_')}_box.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    kv = _parse_kv(sys.argv[1:])
    if "results_dir" not in kv or "run_ids" not in kv:
        raise SystemExit("Usage: python -m src.evaluate results_dir=<DIR> run_ids='["run",...]'")

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

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()  # pd.DataFrame
        summary = run.summary._json_dict
        cfg_run = dict(run.config)

        run_out = results_dir / run_id
        run_out.mkdir(parents=True, exist_ok=True)

        # dump metrics --------------------------------------------------------
        history_json = history.to_dict(orient="list")
        mfile = run_out / "metrics.json"
        with open(mfile, "w", encoding="utf-8") as fp:
            json.dump({"history": history_json, "summary": summary, "config": cfg_run}, fp, indent=2)
        per_run_files.append(mfile)

        # plots ----------------------------------------------------------------
        lc_path = _plot_learning_curves(history, run_id, run_out)
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