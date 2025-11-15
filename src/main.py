# src/main.py
"""Orchestrator – spawns exactly *one* training subprocess.

GitHub Actions execute multiple runs independently; this file just
locates the run config, applies mode-specific patches, then launches
`src.train` as a child Hydra job.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    repo_root = Path(__file__).resolve().parents[1]
    run_yaml = repo_root / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_yaml.is_file():
        raise FileNotFoundError(f"Run YAML not found: {run_yaml}")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_yaml))
    OmegaConf.set_struct(cfg, True)

    # mode handling ----------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.total_updates = 1
        cfg.training.gradient_accumulation_steps = 1
    else:
        cfg.wandb.mode = "online"

    overrides: List[str] = [f"run={cfg.run}", f"results_dir={cfg.results_dir}", f"mode={cfg.mode}"]
    train_script = repo_root / "src" / "train.py"
    cmd = [sys.executable, str(train_script)] + overrides
    print("[main] launching:\n  ", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()