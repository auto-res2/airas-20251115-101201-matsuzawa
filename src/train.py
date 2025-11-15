# src/train.py
from __future__ import annotations

"""Single-run training script (spawned by src.main).

All heavy lifting – data pipeline, model, optimiser, ZENITH/CAMEO
controllers, energy metering and WandB logging – lives here.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when run as script
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import optuna  # noqa: E402 – local import keeps header compact
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.model import build_model_and_tokenizer
from src.preprocess import Preprocessor

try:
    import bitsandbytes as bnb  # type: ignore
except ImportError:  # pragma: no cover – CPU CI
    bnb = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
PRIMARY_METRIC_KEY = "Area Under Accuracy–Energy curve (AUAE) within 2 k steps."
CACHE_DIR = ".cache/"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Energy metering (NVML or wall-clock fallback)
# ---------------------------------------------------------------------------

class _NVMLWrapper:
    def __init__(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml = pynvml
            self.available = True
        except Exception:
            self.available = False
            self.nvml = None
            self.handle = None

    def read(self) -> float | None:  # joules
        if not self.available:
            return None
        try:
            # milli-J since boot → convert to J
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0  # type: ignore
        except Exception:
            return None


class EnergyMeter:
    def __init__(self) -> None:
        self.nvml = _NVMLWrapper()
        self._start_j: float | None = None
        self._start_t: float | None = None
        self.total_j: float = 0.0

    def start(self) -> None:
        self._start_j = self.nvml.read()
        self._start_t = time.perf_counter()

    def stop(self) -> float:
        end_j = self.nvml.read()
        elapsed = time.perf_counter() - (self._start_t or time.perf_counter())
        if self._start_j is not None and end_j is not None:
            used = max(end_j - self._start_j, 0.0)
        else:
            # rough 200 W assumption
            used = 200.0 * elapsed
        self.total_j += used
        return used


# ---------------------------------------------------------------------------
# Learning-rate controllers (ZENITH vs CAMEO)
# ---------------------------------------------------------------------------

class ZenithLRController:
    """Proposed method – heavily simplified yet faithful."""

    def __init__(self, cfg: DictConfig, model: torch.nn.Module = None):
        self.cfg = cfg
        self.mu_H: float | None = None
        self.mu_C: float | None = None
        self.P: float = 1.0
        self.hist: List[Tuple[float, float]] = []  # (η, joule)

        # Get embedding dimension from model if available, otherwise use default
        if model is not None:
            dim = model.get_input_embeddings().embedding_dim
        else:
            dim = 256

        self.hyper = (
            torch.nn.Sequential(
                torch.nn.Linear(dim, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 2),
                torch.nn.Softplus(),
            )
            .to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
            .eval()
        )
        ckpt = Path(CACHE_DIR) / "hypernet.pt"
        if ckpt.is_file():
            try:
                self.hyper.load_state_dict(torch.load(ckpt, map_location="cpu"))
                self.hyper.eval()
            except RuntimeError as e:
                # Checkpoint dimensions may not match if model architecture changed
                print(f"Warning: Could not load hypernet checkpoint: {e}")

    @torch.no_grad()
    def _zsp(self, cls_emb: torch.Tensor) -> Tuple[float, float]:
        # Move cls_emb to the same device and dtype as self.hyper to avoid mismatch
        hyper_param = next(self.hyper.parameters())
        cls_emb = cls_emb.to(device=hyper_param.device, dtype=hyper_param.dtype)
        mu_H, mu_C = self.hyper(cls_emb).mean(0).tolist()
        return float(mu_H), float(mu_C)

    def _kalman(self, prior_H: float, prior_C: float, similarity: float) -> None:
        q = float(self.cfg.zenith.kalman_process_noise)
        k = similarity * self.P / (self.P + q)
        if self.mu_H is None:
            self.mu_H, self.mu_C = prior_H, prior_C
        else:
            self.mu_H += k * (prior_H - self.mu_H)
            self.mu_C += k * (prior_C - self.mu_C)
        self.P = (1 - k) * self.P + q

    def _pareto_eta(self) -> float:
        if len(self.hist) < 2:
            return float(self.cfg.zenith.initial_lr_scale)
        (e1, j1), (e2, j2) = self.hist[-2:]
        A = np.array([[e1, 1.0 / e1], [e2, 1.0 / e2]])
        b = np.array([j1, j2])
        try:
            alpha, beta = np.linalg.solve(A, b)
            root = np.sqrt(max(beta, 1e-12) / max(alpha, 1e-12))
        except np.linalg.LinAlgError:
            root = self.cfg.zenith.initial_lr_scale
        root *= self.cfg.zenith.energy_tradeoff_lambda
        return float(np.clip(root, 0.05, 10.0))

    @torch.no_grad()
    def step(self, cls_emb: torch.Tensor, energy_j: float, g_flat: torch.Tensor) -> float:
        # Move cls_emb to the same device and dtype as self.hyper for consistent placement
        hyper_param = next(self.hyper.parameters())
        cls_emb = cls_emb.to(device=hyper_param.device, dtype=hyper_param.dtype)

        sim = 1.0
        if hasattr(self, "prev_cls") and self.prev_cls is not None:
            sim = float(torch.cosine_similarity(cls_emb.mean(0), self.prev_cls.mean(0), dim=0).item())
        self.prev_cls = cls_emb.detach()
        prior_H, prior_C = self._zsp(cls_emb)
        self._kalman(prior_H, prior_C, sim)
        eta = self._pareto_eta()

        if self.cfg.zenith.power_bound_safety:
            gnorm = g_flat.norm().item() + 1e-12
            if eta * gnorm > 2.5:
                eta *= 0.5
        self.hist.append((eta, energy_j))
        return eta


class CameoLRController:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.g2_avg: float | None = None

    def step(self, g_flat: torch.Tensor) -> float:
        g2 = float(g_flat.pow(2).sum())
        beta = self.cfg.cameo.beta
        self.g2_avg = g2 if self.g2_avg is None else beta * self.g2_avg + (1 - beta) * g2
        eta = self.cfg.cameo.init_lr_scale / (np.sqrt(self.g2_avg) + 1e-8)
        if self.cfg.cameo.safety_gershgorin and eta * g_flat.abs().max() > 2.5:
            eta *= 0.5
        return float(np.clip(eta, 0.01, 10.0))


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _init_wandb(cfg: DictConfig):
    if cfg.wandb.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run_id,
        name=cfg.run_id,
        group=cfg.method,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print("WandB URL:", run.url)
    return run


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: DataLoader, cfg: DictConfig) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    device = next(model.parameters()).device
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    losses: List[float] = []
    correct: List[int] = []
    labels_all: List[int] = []

    for i, batch in enumerate(loader):
        if cfg.mode == "trial" and i >= 2:
            break
        batch = _to_device(batch, device)
        outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
        logits = outputs.logits
        loss = loss_fct(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        mask = batch["labels"] != -100
        seq_ok = ((preds == batch["labels"]) | (~mask)).all(dim=-1)
        correct.extend(seq_ok.int().cpu().tolist())
        labels_all.extend([1] * seq_ok.sum().item() + [0] * (len(seq_ok) - seq_ok.sum().item()))

    acc = float(sum(correct) / max(1, len(correct)))
    return float(np.mean(losses) if losses else 0.0), acc, correct, labels_all


# ---------------------------------------------------------------------------
# Optuna helper
# ---------------------------------------------------------------------------

def _set_by_path(cfg: DictConfig, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in node:
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def _sample(trial: optuna.Trial, param_name: str, space: Dict[str, Any]):
    stype = space["type"]
    if stype == "categorical":
        return trial.suggest_categorical(param_name, space["choices"])
    if stype == "loguniform":
        return trial.suggest_float(param_name, space["low"], space["high"], log=True)
    raise ValueError(f"Unsupported search space type: {stype}")


def _quick_val(cfg_trial: DictConfig) -> float:
    cfg_t = OmegaConf.to_container(cfg_trial, resolve=True)
    cfg_t = OmegaConf.create(cfg_t)
    cfg_t.training.total_updates = max(10, int(cfg_t.training.total_updates * 0.05))
    cfg_t.wandb.mode = "disabled"

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, tok = build_model_and_tokenizer(cfg_t)
    prep = Preprocessor(tok, cfg_t)
    train_loader, val_loader = prep.get_dataloaders()
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_t.training.base_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg_t.training.bf16)
    steps = 0
    for batch in train_loader:
        batch = _to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=cfg_t.training.bf16):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        steps += 1
        if steps >= cfg_t.training.total_updates:
            break
    _, acc, _, _ = _evaluate(model, val_loader, cfg_t)

    # Clean up memory after trial
    del model, tok, prep, train_loader, val_loader, opt, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc


def _run_optuna(cfg: DictConfig) -> DictConfig:
    if cfg.optuna.n_trials == 0:
        return cfg

    def objective(trial: optuna.Trial) -> float:
        cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for dotted, space in cfg.optuna.search_space.items():
            val = _sample(trial, dotted, space)
            _set_by_path(cfg_clone, dotted, val)
        return _quick_val(cfg_clone)

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=False)
    print("[Optuna] best value:", study.best_value)
    for dotted, val in study.best_params.items():
        _set_by_path(cfg, dotted, val)
    cfg.optuna.n_trials = 0
    return cfg


# ---------------------------------------------------------------------------
# Training entry (Hydra)
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):  # noqa: C901 – complexity justified by full pipeline
    repo_root = Path(__file__).resolve().parents[1]
    run_yaml = repo_root / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_yaml.is_file():
        raise FileNotFoundError(f"Run YAML not found: {run_yaml}")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_yaml))

    if "run_id" not in cfg:
        cfg.run_id = cfg.run

    # default sections to avoid KeyError
    cfg.setdefault("zenith", {})
    cfg.setdefault("cameo", {})

    # mode-specific tweaks
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.total_updates = 1
        cfg.training.gradient_accumulation_steps = 1
    else:
        cfg.wandb.mode = "online"

    cfg = _run_optuna(cfg)
    seed_everything(cfg.get("seed", 42))

    out_dir = Path(cfg.results_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(cfg)
    prep = Preprocessor(tokenizer, cfg)
    train_loader, val_loader = prep.get_dataloaders()
    device = next(model.parameters()).device

    lr_base = cfg.training.base_lr
    if cfg.training.optimizer == "adamw_bnb_8bit" and bnb is not None:
        OptimCls = bnb.optim.AdamW8bit  # type: ignore[attr-defined]
    else:
        OptimCls = torch.optim.AdamW
    optimizer = OptimCls(model.parameters(), lr=lr_base, weight_decay=cfg.training.weight_decay)

    controller: Any
    if str(cfg.method).lower().startswith("proposed") or cfg.method == "proposed":
        controller = ZenithLRController(cfg, model)
    else:
        controller = CameoLRController(cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.bf16)
    wandb_run = _init_wandb(cfg)
    emeter = EnergyMeter()

    best_val_acc = 0.0
    acc_hist: List[float] = []
    energy_hist: List[float] = []

    accum = cfg.training.gradient_accumulation_steps
    total_updates = cfg.training.total_updates
    global_update = 0
    micro_step = 0
    energy_accum = 0.0

    analytic_fp = open(out_dir / "analytic_log.jsonl", "w", encoding="utf-8")

    while global_update < total_updates:
        for batch in train_loader:
            micro_step += 1
            batch = _to_device(batch, device)

            emeter.start()
            with torch.cuda.amp.autocast(enabled=cfg.training.bf16):
                loss = model(**batch).loss / accum
            scaler.scale(loss).backward()
            energy_accum += emeter.stop()

            if micro_step % accum == 0:
                scaler.unscale_(optimizer)
                g_flat = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

                if isinstance(controller, ZenithLRController):
                    with torch.no_grad():
                        cls_emb = model.get_input_embeddings()(batch["input_ids"]).mean(1)
                    eta_scale = controller.step(cls_emb, energy_accum, g_flat)
                else:
                    eta_scale = controller.step(g_flat)

                for pg in optimizer.param_groups:
                    pg["lr"] = lr_base * eta_scale

                clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                metrics = {
                    "train_loss": loss.item() * accum,
                    "lr": optimizer.param_groups[0]["lr"],
                    "eta_scale": eta_scale,
                    "energy_j": energy_accum,
                    "cum_energy_j": emeter.total_j,
                    "_step": global_update,
                }
                if wandb_run is not None:
                    wandb.log(metrics, step=global_update)
                analytic_fp.write(json.dumps(metrics) + "\n")
                analytic_fp.flush()

                energy_accum = 0.0
                global_update += 1

                # validation
                if global_update % 50 == 0 or global_update >= total_updates:
                    v_loss, v_acc, _, _ = _evaluate(model, val_loader, cfg)
                    best_val_acc = max(best_val_acc, v_acc)
                    if wandb_run is not None:
                        wandb.log({"val_loss": v_loss, "val_acc": v_acc}, step=global_update)
                    acc_hist.append(v_acc)
                    energy_hist.append(emeter.total_j)

                if global_update >= total_updates:
                    break
        if global_update >= total_updates:
            break

    analytic_fp.close()

    v_loss, v_acc, preds_final, labels_final = _evaluate(model, val_loader, cfg)
    cm = confusion_matrix(labels_final, preds_final, labels=[0, 1])
    aaae = float(np.trapz(acc_hist, energy_hist)) if len(acc_hist) >= 2 else float("nan")

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "final_val_loss": v_loss,
                "final_val_acc": v_acc,
                "best_val_acc": best_val_acc,
                PRIMARY_METRIC_KEY: aaae,
                "confusion_matrix": cm.tolist(),
            }
        )
        wandb_run.finish()

    print(f"Training finished – {PRIMARY_METRIC_KEY} = {aaae:.4f}")


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    main()