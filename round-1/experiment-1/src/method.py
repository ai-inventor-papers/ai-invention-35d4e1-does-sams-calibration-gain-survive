#!/usr/bin/env python3
"""SAM vs SGD on Fashion-MNIST: calibration before and after temperature scaling.

Stages (each stage reads the outputs of the previous ones, so an interrupted run resumes):
  selftest   unit checks on synthetic data (ECE, temperature recovery, SAM perturbation)
  smoke      1-epoch end-to-end pass on a 2,000-image train subset, written to scratch/smoke
  timecheck  times one SGD epoch and one SAM epoch on the device, picks N (30 or 20)
  tune       validation-only grid search at N epochs (seed 0), writes results/tuning.json
  final      trains conditions A, B, C for the given seeds, saves validation/test/corrupted logits
  analyze    temperature scaling, metrics, paired bootstrap analysis, method_out.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "scratch" / "data"
DEVICE = torch.device(os.environ.get("SAM_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))

PIXEL_MEAN = 0.2860
PIXEL_STD = 0.3530
BATCH = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_GRID_SGD = [0.05, 0.1]
LR_GRID_SAM = [0.05, 0.1]
RHO_GRID = [0.02, 0.05, 0.1]
N_BINS = 15
N_BOOT = 2000
TIME_CAP_S = 12600.0  # 3.5 h protocol cap
TOTAL_SGD_EPOCHS_PER_N = 29  # tuning 14N + finals 3 x 15N, in SGD-epoch equivalents
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CORRUPTIONS = ["noise0.1", "noise0.2", "blur3_sigma1.0", "contrast0.5"]
CONDITIONS = {
    "A": "SGD, N epochs (matched epochs control)",
    "B": "SAM, N epochs, rho*, lr_sam (main SAM arm, 2 gradient evaluations per step)",
    "C": "SGD, 2N epochs (matched gradient-budget control, main comparison)",
}


# ----------------------------------------------------------------------------- setup

def setup_logging(out: Path) -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    logger.add(out / "logs" / "run.log", rotation="30 MB", level="DEBUG")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def norm(x: torch.Tensor) -> torch.Tensor:
    return (x - PIXEL_MEAN) / PIXEL_STD


def ensure_dirs(out: Path) -> None:
    for sub in ["results", "results/logits", "results/train_logs", "checkpoints",
                "reliability", "reserved/corruptions", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def load_data(out: Path, smoke: bool) -> dict:
    """Official Fashion-MNIST splits; 55,000 train / 5,000 validation split from the training set.

    Pixels stay in [0, 1] on the device; normalisation happens inside the forward pass so
    the augmentation's zero padding is zero background pixels.
    """
    tr = datasets.FashionMNIST(root=str(DATA_ROOT), train=True, download=True)
    te = datasets.FashionMNIST(root=str(DATA_ROOT), train=False, download=True)
    if len(tr) != 60000 or len(te) != 10000:
        raise RuntimeError(f"unexpected Fashion-MNIST sizes {len(tr)} / {len(te)}")
    x_all = tr.data.float().div(255.0).unsqueeze(1)
    y_all = tr.targets.long()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=0)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y_all)), y_all.numpy()))
    tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)
    if smoke:
        tr_idx, va_idx = tr_idx[:2000], va_idx[:1000]
    else:
        split_path = out / "results" / "split_indices.json"
        split_path.write_text(json.dumps({
            "train_indices": tr_idx.tolist(), "validation_indices": va_idx.tolist(),
            "note": "StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=0) on the official train set",
        }))
    x_te = te.data.float().div(255.0).unsqueeze(1)
    y_te = te.targets.long()
    data = {
        "xtr": x_all[tr_idx].to(DEVICE), "ytr": y_all[tr_idx].to(DEVICE),
        "xva": x_all[va_idx].to(DEVICE), "yva": y_all[va_idx].to(DEVICE),
        "xte": x_te.to(DEVICE), "yte": y_te.to(DEVICE),
        "yva_np": y_all[va_idx].numpy(), "yte_np": y_te.numpy(),
    }
    logger.info(f"Device {DEVICE}; train {len(tr_idx)}, validation {len(va_idx)}, test {len(y_te)}")
    return data


# ----------------------------------------------------------------------------- model

class SmallCNN(nn.Module):
    """Conv(32)-BN-ReLU-Pool, Conv(64)-BN-ReLU-Pool, Linear(3136,64)-ReLU, Linear(64,10). About 0.22M parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def random_crop(x: torch.Tensor, gen: torch.Generator, pad: int = 2) -> torch.Tensor:
    """Vectorised per-sample random crop with zero padding of `pad` pixels (no flips)."""
    b, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (b,), generator=gen).to(x.device)
    dx = torch.randint(0, 2 * pad + 1, (b,), generator=gen).to(x.device)
    rows = dy[:, None] + torch.arange(h, device=x.device)[None, :]
    cols = dx[:, None] + torch.arange(w, device=x.device)[None, :]
    bi = torch.arange(b, device=x.device)[:, None, None]
    out = xp[:, 0][bi, rows[:, :, None], cols[:, None, :]]
    return out.unsqueeze(1)


def sam_perturb(params: list, rho: float) -> list:
    """Move every parameter by e = rho * g / ||g|| (g: gradient over all parameters). Returns e."""
    grads = [p.grad for p in params]
    gnorm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    eps = []
    with torch.no_grad():
        for p in params:
            e = rho * p.grad / (gnorm + 1e-12)
            p.add_(e)
            eps.append(e)
    return eps


def sam_restore(params: list, eps: list) -> None:
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.sub_(e)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, x.shape[0], 1000):
        out.append(model(norm(x[i:i + 1000])).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def val_nll_of(logits: np.ndarray, y: np.ndarray) -> float:
    if not np.all(np.isfinite(logits)):
        return float("inf")
    p = softmax_np(logits.astype(np.float64))
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def train_run(*, optimizer: str, epochs: int, lr: float, rho: float | None, seed: int,
              data: dict, tag: str) -> tuple[nn.Module, dict]:
    """Trains one model. `data` must contain xtr/ytr/xva/yva only (the test set is never passed here)."""
    set_seed(seed)
    model = SmallCNN().to(DEVICE)
    params = list(model.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=False)
    xtr, ytr = data["xtr"], data["ytr"]
    n = xtr.shape[0]
    steps_per_epoch = math.ceil(n / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    gen = torch.Generator().manual_seed(seed)
    history = []
    diverged = False
    t_run = time.perf_counter()
    logger.info(f"[{tag}] start: {optimizer} epochs={epochs} lr={lr} rho={rho} seed={seed} steps/epoch={steps_per_epoch}")
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=gen)
        loss_acc = torch.zeros((), device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            idx_d = idx.to(DEVICE)
            xb = norm(random_crop(xtr[idx_d], gen))
            yb = ytr[idx_d]
            # first forward / backward (both optimisers)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if optimizer == "sgd":
                opt.step()
            else:
                eps = sam_perturb(params, rho)
                opt.zero_grad(set_to_none=True)
                # second forward / backward at the perturbed weights; BN stats update here too
                loss2 = F.cross_entropy(model(xb), yb)
                loss2.backward()
                sam_restore(params, eps)
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            loss_acc += loss.detach() * len(idx)
        train_loss = loss_acc.item() / n
        sec = time.perf_counter() - t0
        if not math.isfinite(train_loss):
            diverged = True
            logger.warning(f"[{tag}] non-finite training loss at epoch {ep + 1}; stopping this run")
            history.append({"epoch": ep + 1, "train_loss": train_loss, "val_nll": float("inf"), "seconds": sec})
            break
        val_logits = predict(model, data["xva"])
        v_nll = val_nll_of(val_logits, data["yva_np"])
        history.append({"epoch": ep + 1, "train_loss": train_loss, "val_nll": v_nll, "seconds": sec})
        logger.info(f"[{tag}] epoch {ep + 1}/{epochs} train_loss={train_loss:.4f} val_nll={v_nll:.4f} sec={sec:.1f}")
    fb_passes = (0 if diverged else epochs * steps_per_epoch) * (1 if optimizer == "sgd" else 2)
    summary = {
        "optimizer": optimizer, "epochs": epochs, "lr": lr, "rho": rho, "seed": seed,
        "forward_backward_passes": fb_passes, "wall_seconds": time.perf_counter() - t_run,
        "diverged": diverged, "history": history, "device": str(DEVICE),
    }
    logger.info(f"[{tag}] done in {summary['wall_seconds']:.1f} s, fwd/bwd passes={fb_passes}")
    return model, summary


# ----------------------------------------------------------------------------- metrics

def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    """Bins are (lo, hi]; the first bin also holds 0 and the last holds 1."""
    return np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)


def ece_from(bins: np.ndarray, conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """ECE = sum_b (n_b / N) |acc_b - conf_b| computed from per-bin sums (empty bins contribute 0)."""
    sum_conf = np.bincount(bins, weights=conf, minlength=n_bins)
    sum_acc = np.bincount(bins, weights=correct.astype(np.float64), minlength=n_bins)
    return float(np.abs(sum_acc - sum_conf).sum() / len(conf))


def ece_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    return ece_from(bin_index(conf, n_bins), conf, correct, n_bins)


def fit_temperature(val_logits: np.ndarray, y_val: np.ndarray) -> float:
    z = val_logits.astype(np.float64)

    def nll_at(log_t: float) -> float:
        p = softmax_np(z / math.exp(log_t))
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, None))))

    res = minimize_scalar(nll_at, bounds=(math.log(0.05), math.log(20.0)), method="bounded",
                          options={"xatol": 1e-7})
    return float(math.exp(res.x))


def summarize(logits: np.ndarray, y: np.ndarray, temperature: float) -> dict:
    p = softmax_np(logits.astype(np.float64) / temperature)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = pred == y
    py = np.clip(p[np.arange(len(y)), y], 1e-12, None)
    onehot = np.eye(p.shape[1])[y]
    mis = (~correct).astype(int)
    auroc = float(roc_auc_score(mis, -conf)) if 0 < mis.sum() < len(mis) else float("nan")
    return {
        "temperature": temperature,
        "accuracy": float(correct.mean()),
        "ece_15": ece_bins(conf, correct, 15),
        "ece_10": ece_bins(conf, correct, 10),
        "ece_20": ece_bins(conf, correct, 20),
        "nll": float(-np.log(py).mean()),
        "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "auroc_misclassification": auroc,
        "mean_conf_misclassified": float(conf[~correct].mean()) if mis.sum() else None,
        "mean_conf_all": float(conf.mean()),
    }


def reliability(logits: np.ndarray, y: np.ndarray, temperature: float, n_bins: int = N_BINS) -> dict:
    p = softmax_np(logits.astype(np.float64) / temperature)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y)
    b = bin_index(conf, n_bins)
    cnt = np.bincount(b, minlength=n_bins)
    sc = np.bincount(b, weights=conf, minlength=n_bins)
    sa = np.bincount(b, weights=correct.astype(np.float64), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(cnt > 0, sc / np.maximum(cnt, 1), np.nan)
        acc = np.where(cnt > 0, sa / np.maximum(cnt, 1), np.nan)
    return {
        "bin_edges": np.linspace(0, 1, n_bins + 1).tolist(),
        "count": cnt.tolist(),
        "mean_confidence": [None if not math.isfinite(v) else float(v) for v in mean_conf],
        "accuracy": [None if not math.isfinite(v) else float(v) for v in acc],
    }


def per_example_arrays(logits: np.ndarray, y: np.ndarray, temperature: float) -> dict:
    """Per-example quantities so each bootstrap resample is a cheap index into arrays."""
    p_raw = softmax_np(logits.astype(np.float64))
    p_sc = softmax_np(logits.astype(np.float64) / temperature)
    corr = (p_raw.argmax(axis=1) == y).astype(np.float64)
    conf_raw = p_raw.max(axis=1)
    conf_sc = p_sc.max(axis=1)
    py_sc = np.clip(p_sc[np.arange(len(y)), y], 1e-12, None)
    return {
        "corr": corr,
        "conf_raw": conf_raw, "bins_raw": bin_index(conf_raw, N_BINS),
        "conf_sc": conf_sc, "bins_sc": bin_index(conf_sc, N_BINS),
        "nll_i": -np.log(py_sc),
        "brier_i": ((p_sc - np.eye(p_sc.shape[1])[y]) ** 2).sum(axis=1),
        "mis": 1.0 - corr,
        "score": -conf_sc,
    }


PAIRED_METRICS = ["accuracy", "ece_raw", "ece_scaled", "nll_scaled", "brier_scaled", "auroc_scaled"]


def metrics_on_idx(a: dict, idx: np.ndarray | None) -> dict:
    if idx is None:
        idx = slice(None)
    corr = a["corr"][idx]
    try:
        auroc = float(roc_auc_score(a["mis"][idx], a["score"][idx]))
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(corr.mean()),
        "ece_raw": ece_from(a["bins_raw"][idx], a["conf_raw"][idx], corr),
        "ece_scaled": ece_from(a["bins_sc"][idx], a["conf_sc"][idx], corr),
        "nll_scaled": float(a["nll_i"][idx].mean()),
        "brier_scaled": float(a["brier_i"][idx].mean()),
        "auroc_scaled": auroc,
    }


def holm(pvals: list) -> list:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def two_sided_boot_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
    return min(1.0, p)


# ----------------------------------------------------------------------------- self tests

def run_selftest() -> None:
    logger.info("Self-test 1: ECE of perfectly calibrated binary probabilities")
    rng = np.random.default_rng(0)
    p_pos = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, len(p_pos)) < p_pos).astype(int)
    pred = (p_pos > 0.5).astype(int)
    conf = np.maximum(p_pos, 1 - p_pos)
    ece_cal = ece_bins(conf, pred == y, 15)
    logger.info(f"  calibrated ECE = {ece_cal:.5f}")
    assert ece_cal < 0.01, "calibrated ECE should be close to 0"

    logger.info("Self-test 2: ECE detects logits scaled by a known factor (overconfidence x3)")
    z = rng.normal(size=(50_000, 5)) * 1.0
    p = softmax_np(z)
    yy = np.array([rng.choice(5, p=row) for row in p[:20_000]])
    lg = z[:20_000]
    ece_ok = summarize(lg, yy, 1.0)["ece_15"]
    ece_over = summarize(lg * 3.0, yy, 1.0)["ece_15"]
    logger.info(f"  ECE calibrated logits {ece_ok:.4f} vs x3 logits {ece_over:.4f}")
    assert ece_over > ece_ok + 0.05, "over-confident logits should have clearly larger ECE"

    logger.info("Self-test 3: temperature fit recovers a known T within 5%")
    t_true = 2.5
    z = rng.normal(size=(20_000, 5)) * 2.0
    pz = softmax_np(z)
    labels = np.array([rng.choice(5, p=row) for row in pz])
    t_fit = fit_temperature(z * t_true, labels)
    logger.info(f"  true T={t_true}, fitted T={t_fit:.4f}")
    assert abs(t_fit - t_true) / t_true < 0.05, "temperature not recovered within 5%"

    logger.info("Self-test 4: SAM perturbation norm equals rho and weights are restored")
    set_seed(0)
    model = SmallCNN().to(DEVICE)
    params = list(model.parameters())
    x = torch.rand(64, 1, 28, 28, device=DEVICE)
    yb = torch.randint(0, 10, (64,), device=DEVICE)
    loss = F.cross_entropy(model(norm(x)), yb)
    loss.backward()
    before = [p.detach().clone() for p in params]
    rho = 0.05
    eps = sam_perturb(params, rho)
    en = torch.sqrt(sum(e.pow(2).sum() for e in eps)).item()
    logger.info(f"  ||e|| = {en:.9f} (rho = {rho})")
    assert abs(en - rho) < 1e-6, "perturbation norm is not rho"
    sam_restore(params, eps)
    maxdiff = max((p.detach() - b).abs().max().item() for p, b in zip(params, before))
    logger.info(f"  max |w_restored - w_before| = {maxdiff:.3e}")
    assert maxdiff < 1e-6, "weights not restored after the second step"
    logger.info("All self-tests passed")


# ----------------------------------------------------------------------------- corruptions

def make_corruptions(x: torch.Tensor) -> dict:
    """Test-image corruptions in pixel space. Logits only are saved; no metric is computed on them."""
    out = {}
    g = torch.Generator().manual_seed(1234)
    base = torch.randn(x.shape, generator=g).to(x.device)
    out["noise0.1"] = (x + 0.1 * base).clamp(0, 1)
    out["noise0.2"] = (x + 0.2 * base).clamp(0, 1)
    k1 = torch.tensor([math.exp(-(t * t) / (2 * 1.0 ** 2)) for t in (-1, 0, 1)], dtype=torch.float32)
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1)[None, None].to(x.device)
    out["blur3_sigma1.0"] = F.conv2d(x, k2, padding=1)
    out["contrast0.5"] = x * 0.5
    return out


# ----------------------------------------------------------------------------- stages

def load_config(out: Path) -> dict:
    return json.loads((out / "results" / "config.json").read_text())


def stage_timecheck(out: Path, data: dict) -> dict:
    path = out / "results" / "timecheck.json"
    if path.exists():
        logger.info("timecheck.json exists; reusing it")
        return json.loads(path.read_text())
    _, sgd = train_run(optimizer="sgd", epochs=1, lr=0.05, rho=None, seed=999, data=data, tag="timecheck-sgd")
    t_sgd = sgd["wall_seconds"]
    _, sam = train_run(optimizer="sam", epochs=1, lr=0.05, rho=0.05, seed=999, data=data, tag="timecheck-sam")
    t_sam = sam["wall_seconds"]
    ratio = t_sam / t_sgd
    n_epochs = 30 if TOTAL_SGD_EPOCHS_PER_N * 30 * t_sgd <= TIME_CAP_S else 20
    proj = TOTAL_SGD_EPOCHS_PER_N * n_epochs * t_sgd
    ext = 2 * 150 * n_epochs / 30 * t_sgd  # seeds 3 and 4 (A, B, C each), scaled to the chosen N
    rec = {
        "t_sgd_epoch_seconds": t_sgd, "t_sam_epoch_seconds": t_sam, "sam_to_sgd_ratio": ratio,
        "threshold_t_for_N30_seconds": TIME_CAP_S / (TOTAL_SGD_EPOCHS_PER_N * 30),
        "N": n_epochs, "projected_main_seconds": proj,
        "projected_extension_seeds_3_4_seconds": ext,
        "changed_N_from_30": n_epochs != 30,
    }
    logger.info(f"Time check: SGD epoch {t_sgd:.1f} s, SAM epoch {t_sam:.1f} s (ratio {ratio:.2f}); "
                f"N={n_epochs}; projected main run {proj / 60:.1f} min; extension {ext / 60:.1f} min")
    path.write_text(json.dumps(rec, indent=2))
    return rec


def stage_tune(out: Path, data: dict, n_epochs: int) -> dict:
    path = out / "results" / "tuning.json"
    if path.exists():
        logger.info("tuning.json exists; reusing it")
        return json.loads(path.read_text())
    # tuning receives validation data only; the test set is never passed to this stage
    tune_data = {k: data[k] for k in ("xtr", "ytr", "xva", "yva", "yva_np")}
    grid = [("sgd", lr, None) for lr in LR_GRID_SGD] + \
           [("sam", lr, rho) for lr in LR_GRID_SAM for rho in RHO_GRID]
    runs = []
    for optimizer, lr, rho in grid:
        name = f"{optimizer}_lr{lr}" + (f"_rho{rho}" if rho is not None else "")
        model, summ = train_run(optimizer=optimizer, epochs=n_epochs, lr=lr, rho=rho, seed=0,
                                data=tune_data, tag=f"tune:{name}")
        vlog = predict(model, tune_data["xva"])
        summ["val_nll"] = val_nll_of(vlog, data["yva_np"])
        summ["val_accuracy"] = float(np.mean(vlog.argmax(1) == data["yva_np"]))
        summ["name"] = name
        runs.append(summ)
        torch.save(model.state_dict(), out / "checkpoints" / f"tune_{name}.pt")
        del model
        gc.collect()
    sel = {}
    for optimizer in ("sgd", "sam"):
        cands = [r for r in runs if r["optimizer"] == optimizer and math.isfinite(r["val_nll"])]
        best = min(cands, key=lambda r: (r["val_nll"], r["lr"]))
        sel[optimizer] = {"lr": best["lr"], "rho": best["rho"], "val_nll": best["val_nll"], "name": best["name"]}
    rec = {"N": n_epochs, "seed": 0, "selection_rule": "lowest validation NLL of the uncalibrated model",
           "selected": sel, "runs": runs}
    path.write_text(json.dumps(rec, indent=2))
    logger.info(f"Selected SGD lr={sel['sgd']['lr']}; SAM lr={sel['sam']['lr']}, rho={sel['sam']['rho']}")
    return rec


def stage_final(out: Path, data: dict, seeds: list, cfg: dict) -> None:
    corrupt = make_corruptions(data["xte"])
    n_ep = cfg["N"]
    specs = [
        ("A", "sgd", n_ep, cfg["lr_sgd"], None),
        ("B", "sam", n_ep, cfg["lr_sam"], cfg["rho_sam"]),
        ("C", "sgd", 2 * n_ep, cfg["lr_sgd"], None),
    ]
    for seed in seeds:
        for cond, optimizer, epochs, lr, rho in specs:
            tag = f"{cond}_seed{seed}"
            ckpt = out / "checkpoints" / f"{tag}.pt"
            log_path = out / "results" / "train_logs" / f"{tag}.json"
            model = SmallCNN().to(DEVICE)
            if ckpt.exists() and log_path.exists():
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                logger.info(f"[{tag}] reusing checkpoint")
            else:
                # final stage gets validation and training tensors only in train_run; test is added after
                ft = {k: data[k] for k in ("xtr", "ytr", "xva", "yva", "yva_np")}
                model, summ = train_run(optimizer=optimizer, epochs=epochs, lr=lr, rho=rho, seed=seed,
                                        data=ft, tag=tag)
                torch.save(model.state_dict(), ckpt)
                log_path.write_text(json.dumps(summ, indent=2))
            np.save(out / "results" / "logits" / f"{tag}_val.npy", predict(model, data["xva"]))
            np.save(out / "results" / "logits" / f"{tag}_test.npy", predict(model, data["xte"]))
            for cname, cx in corrupt.items():
                np.save(out / "reserved" / "corruptions" / f"{tag}_{cname}.npy", predict(model, cx))
            logger.info(f"[{tag}] logits saved (val, test, {len(corrupt)} reserved corruptions)")
            del model
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()


def bootstrap_metrics(arrays: dict, seed: int) -> dict:
    """Bootstrap of test indices (same indices for all conditions in a seed)."""
    rng = np.random.default_rng(1000 + seed)
    n = len(arrays["A"]["corr"])
    boot = {c: {m: np.empty(N_BOOT) for m in PAIRED_METRICS} for c in arrays}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for c in arrays:
            mv = metrics_on_idx(arrays[c], idx)
            for m in PAIRED_METRICS:
                boot[c][m][b] = mv[m]
    return boot


def analyze(out: Path, data_labels: dict, seeds_wanted: list) -> None:
    cfg = load_config(out)
    logits_dir = out / "results" / "logits"
    per_model, arrays_by_seed, rel_written = {}, {}, []
    avail_seeds = []
    for seed in seeds_wanted:
        have = {}
        for cond in "ABC":
            tag = f"{cond}_seed{seed}"
            vp, tp = logits_dir / f"{tag}_val.npy", logits_dir / f"{tag}_test.npy"
            if vp.exists() and tp.exists():
                have[cond] = (np.load(vp), np.load(tp))
        if len(have) < 3:
            logger.warning(f"seed {seed}: conditions present {sorted(have)}; pair analysis skipped for it")
            continue
        avail_seeds.append(seed)
        arrays_by_seed[seed] = {}
        for cond, (lv, lt) in have.items():
            tag = f"{cond}_seed{seed}"
            T = fit_temperature(lv, data_labels["yva_np"])
            raw = summarize(lt, data_labels["yte_np"], 1.0)
            sc = summarize(lt, data_labels["yte_np"], T)
            per_model[tag] = {"condition": cond, "seed": seed, "raw": raw, "scaled": sc}
            rel_path = out / "reliability" / f"{tag}.json"
            rel_path.write_text(json.dumps({
                "split": "test", "n_bins": N_BINS, "temperature": T,
                "raw": reliability(lt, data_labels["yte_np"], 1.0),
                "scaled": reliability(lt, data_labels["yte_np"], T),
            }, indent=1))
            rel_written.append(str(rel_path.relative_to(out)))
            arrays_by_seed[seed][cond] = per_example_arrays(lt, data_labels["yte_np"], T)
            logger.info(f"[{tag}] T={T:.4f} acc={raw['accuracy']:.4f} ECE raw={raw['ece_15']:.4f} "
                        f"ECE scaled={sc['ece_15']:.4f} NLL scaled={sc['nll']:.4f} AUROC raw={raw['auroc_misclassification']:.4f}")

    paired = {"P1_B_minus_C": {}, "P2_B_minus_A": {}}
    seed_pvals = {"P1_B_minus_C": {}, "P2_B_minus_A": {}}
    for seed in avail_seeds:
        boot = bootstrap_metrics(arrays_by_seed[seed], seed)
        point = {c: metrics_on_idx(arrays_by_seed[seed][c], None) for c in "ABC"}
        for pname, (x, y) in {"P1_B_minus_C": ("B", "C"), "P2_B_minus_A": ("B", "A")}.items():
            rows = {}
            pvals = []
            for m in PAIRED_METRICS:
                d_boot = boot[x][m] - boot[y][m]
                d_point = point[x][m] - point[y][m]
                p_val = two_sided_boot_p(d_boot)
                rows[m] = {"point": d_point, "ci95": [float(v) for v in np.nanpercentile(d_boot, [2.5, 97.5])],
                           "p_bootstrap_two_sided": p_val}
                pvals.append(p_val if math.isfinite(p_val) else 1.0)
            for m, adj in zip(PAIRED_METRICS, holm(pvals)):
                rows[m]["p_holm_within_family_per_seed"] = adj
            paired[pname][f"seed{seed}"] = rows
    if avail_seeds:
        for pname in paired:
            cross = {}
            for m in PAIRED_METRICS:
                vals = np.array([paired[pname][f"seed{s}"][m]["point"] for s in avail_seeds])
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
                cross[m] = {
                    "per_seed_point": vals.tolist(), "mean": float(vals.mean()), "sd_over_seeds": sd,
                    "sign_pattern": "".join("+" if v > 0 else "-" if v < 0 else "0" for v in vals),
                }
            paired[pname]["across_seeds"] = cross
    seed_sd_C = {}
    if avail_seeds:
        for m in PAIRED_METRICS:
            vals = np.array([point_c for point_c in [metrics_on_idx(arrays_by_seed[s]["C"], None)[m] for s in avail_seeds]])
            seed_sd_C[m] = {"mean_over_seeds": float(vals.mean()),
                            "sd_over_seeds": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")}

    # method_out.json in exp_gen_sol_out form: top-level metadata + datasets[].examples
    classes = CLASSES
    yte = data_labels["yte_np"]
    examples = []
    pred_by_tag = {}
    for tag in per_model:
        pred_by_tag[tag] = np.load(logits_dir / f"{tag}_test.npy").argmax(1)
    for i in range(len(yte)):
        ex = {
            "input": f"Fashion-MNIST official test image {i}; classify into 10 clothing classes.",
            "output": classes[int(yte[i])],
            "metadata_index": i,
            "metadata_true_label": int(yte[i]),
        }
        for tag in sorted(pred_by_tag):
            ex[f"predict_{tag}"] = classes[int(pred_by_tag[tag][i])]
        examples.append(ex)

    timing = {}
    for f in sorted((out / "results" / "train_logs").glob("*.json")):
        s = json.loads(f.read_text())
        timing[f.stem] = {"wall_seconds": s["wall_seconds"], "forward_backward_passes": s["forward_backward_passes"],
                          "epochs": s["epochs"], "diverged": s["diverged"]}
    tuning = json.loads((out / "results" / "tuning.json").read_text()) if (out / "results" / "tuning.json").exists() else None
    timecheck = json.loads((out / "results" / "timecheck.json").read_text())

    output = {
        "metadata": {
            "method_name": "SAM vs SGD: calibration after temperature scaling (Fashion-MNIST, small CNN)",
            "description": ("Matched gradient budget comparison. B (SAM, N epochs, two gradient evaluations per step) is "
                            "compared with C (SGD, 2N epochs) in the primary contrast and with A (SGD, N epochs) in the "
                            "matched-epochs contrast. Metrics on the 10,000-image official test split; temperature fitted on "
                            "the 5,000-image validation split."),
            "conditions": CONDITIONS,
            "config": {
                "N_epochs": cfg["N"], "batch_size": BATCH, "optimizer_base": "SGD(momentum=0.9, weight_decay=5e-4, nesterov=False)",
                "scheduler": "CosineAnnealingLR, T_max = epochs * steps_per_epoch, stepped every iteration",
                "lr_sgd": cfg["lr_sgd"], "lr_sam": cfg["lr_sam"], "rho_sam": cfg["rho_sam"],
                "augmentation": "random crop, 2-pixel zero padding in pixel space, per-sample offsets, no flips",
                "normalisation": {"mean": PIXEL_MEAN, "std": PIXEL_STD},
                "ece_bins": N_BINS, "bootstrap_resamples": N_BOOT, "temperature_search": "log T in [log 0.05, log 20], bounded",
                "seeds_in_pair_analysis": avail_seeds,
            },
            "deviations": [
                "Hidden layer of the classifier head has 64 units (about 0.22M parameters), not the 128 in the direction (about 0.42M), to stay in the stated 0.1-0.3M range.",
                ("Learning rate and rho tuned at N epochs and reused for the 2N-epoch control C."
                 " C uses the SGD learning rate selected at N."),
                ("N changed to 20 (2N = 40) by the time check." if timecheck["changed_N_from_30"] else
                 "N = 30 (2N = 60); the time check passed."),
                "BatchNorm running statistics are updated on both forward passes of each SAM step (PyTorch default).",
                "Bootstrap intervals capture test-sampling noise only; decisions also use the seed-level paired differences.",
                "Corrupted-test logits are saved in reserved/corruptions and no metric is computed on them in this iteration.",
                "The optional Gauss-Newton sharpness check is not part of this artifact.",
            ],
            "device": str(DEVICE),
            "time_check": timecheck,
            "tuning": tuning,
            "timings_per_condition": timing,
            "per_model_metrics": per_model,
            "paired_differences": paired,
            "seed_sd_of_C": seed_sd_C,
            "seeds_available": avail_seeds,
            "paths": {
                "logits": "results/logits/{cond}_seed{s}_{val|test}.npy (stays on the run's volume; kept)",
                "reliability": "reliability/{cond}_seed{s}.json",
                "reserved_corruptions": "reserved/corruptions/{cond}_seed{s}_{corruption}.npy (logits only)",
                "checkpoints": "checkpoints/{cond}_seed{s}.pt (kept)",
                "split_indices": "results/split_indices.json",
            },
            "reliability_files_written": rel_written,
        },
        "datasets": [{"dataset": "fashion_mnist_official_test", "examples": examples}],
    }
    (out / "method_out.json").write_text(json.dumps(output))
    logger.info(f"Wrote {out / 'method_out.json'}")


# ----------------------------------------------------------------------------- main

@logger.catch(reraise=True)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["selftest", "smoke", "timecheck", "tune", "final", "analyze", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    out = args.out.resolve()
    ensure_dirs(out)
    setup_logging(out)
    logger.info(f"Stage {args.stage}; out={out}; device={DEVICE}; torch {torch.__version__}")

    if args.stage == "selftest":
        run_selftest()
        return

    if args.stage == "smoke":
        run_selftest()
        smoke_out = HERE / "scratch" / "smoke"
        ensure_dirs(smoke_out)
        setup_logging(smoke_out)
        data = load_data(smoke_out, smoke=True)
        smoke_cfg = {"N": 1, "lr_sgd": 0.05, "lr_sam": 0.05, "rho_sam": 0.05, "note": "smoke test"}
        (smoke_out / "results" / "config.json").write_text(json.dumps(smoke_cfg, indent=2))
        (smoke_out / "results" / "timecheck.json").write_text(json.dumps(
            {"N": 1, "changed_N_from_30": True, "note": "smoke"}, indent=2))
        stage_final(smoke_out, data, [0], smoke_cfg)
        labels = {"yva_np": data["yva_np"], "yte_np": data["yte_np"]}
        analyze(smoke_out, labels, [0])
        return

    data = load_data(out, smoke=False)
    labels = {"yva_np": data["yva_np"], "yte_np": data["yte_np"]}

    if args.stage in ("timecheck", "all"):
        tc = stage_timecheck(out, data)
    if args.stage in ("tune", "all"):
        tc = json.loads((out / "results" / "timecheck.json").read_text())
        tuning = stage_tune(out, data, tc["N"])
        sel = tuning["selected"]
        cfg = {"N": tc["N"], "lr_sgd": sel["sgd"]["lr"], "lr_sam": sel["sam"]["lr"], "rho_sam": sel["sam"]["rho"]}
        (out / "results" / "config.json").write_text(json.dumps(cfg, indent=2))
        logger.info(f"Config: {cfg}")
    if args.stage in ("final", "all"):
        stage_final(out, data, args.seeds, load_config(out))
    if args.stage in ("analyze", "all"):
        analyze(out, labels, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    main()
