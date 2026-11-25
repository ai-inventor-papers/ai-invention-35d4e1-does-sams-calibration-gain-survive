#!/usr/bin/env python3
"""SAM vs SGD on Fashion-MNIST with a small CNN: checks, tuning, final runs.

Stages (run one at a time with --stage):
  checks  unit checks (parameter count, crop shape, SAM perturbation norm,
          BatchNorm statistics untouched by the second SAM pass, counter)
  smoke   one-epoch run of each arm to measure seconds per epoch
  tune    seed-0 grid on the 5,000-image validation split (no test set touched)
  final   seeds 0,1,2 for SAM_30, SGD_30, SGD_60; test set evaluated once per model
  repro   rerun seed-0 SGD_30 once to check reproducibility of the test accuracy
  all     checks, tune, final, repro in that order

Every training run writes results/runs/<tag>.json and is skipped on rerun if
that file exists, so an interrupted stage resumes where it stopped.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

WS = Path(__file__).resolve().parent
DATA_DIR = WS / "data"
RES = WS / "results"
RUNS = RES / "runs"
TEST_DIR = RES / "test"
CK = WS / "checkpoints"
LOG_DIR = WS / "logs"

BATCH = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SPLIT_SEED = 1234
N_VAL = 5000
EPOCHS_N = 30
EPOCHS_2N = 60
FINAL_SEEDS = [0, 1, 2]
TUNE_SEED = 0
RAM_CAP_GB = 12.0

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")


def start_ram_watchdog(cap_gb: float) -> None:
    """Exit if this process tree's resident memory exceeds cap_gb (macOS has no RLIMIT_AS)."""
    me = psutil.Process()

    def _watch() -> None:
        while True:
            rss = sum(p.memory_info().rss for p in [me, *me.children(recursive=True)])
            if rss > cap_gb * 1024**3:
                logger.error(f"RAM cap hit: {rss / 1024**3:.1f} GB > {cap_gb} GB")
                import os
                os._exit(137)
            time.sleep(1)

    threading.Thread(target=_watch, daemon=True).start()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class Data:
    """Raw [0,1] images on DEVICE plus the fixed split and normalisation stats."""

    def __init__(self) -> None:
        from torchvision import datasets

        tr = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True)
        te = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True)
        X = tr.data.float().div(255.0).unsqueeze(1)
        Y = tr.targets.long()
        perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SPLIT_SEED))
        val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
        self.train_x = X[tr_idx].to(DEVICE)
        self.train_y = Y[tr_idx].to(DEVICE)
        self.val_x = X[val_idx].to(DEVICE)
        self.val_y = Y[val_idx].to(DEVICE)
        self.test_x = te.data.float().div(255.0).unsqueeze(1).to(DEVICE)
        self.test_y = te.targets.long().to(DEVICE)
        self.mean = float(self.train_x.mean().item())
        self.std = float(self.train_x.std().item())
        RES.mkdir(parents=True, exist_ok=True)
        split_path = RES / "split_indices.json"
        if not split_path.exists():
            split_path.write_text(json.dumps({
                "split_seed": SPLIT_SEED,
                "val_indices": sorted(val_idx.tolist()),
                "train_indices": sorted(tr_idx.tolist()),
            }))
        (RES / "norm_stats.json").write_text(json.dumps({"mean": self.mean, "std": self.std,
                                                         "computed_on": "55000-image training split, raw [0,1] pixels"}))
        logger.info(f"Data: train {tuple(self.train_x.shape)}, val {tuple(self.val_x.shape)}, "
                    f"test {tuple(self.test_x.shape)}; mean={self.mean:.4f} std={self.std:.4f}")

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def random_crop(x_raw: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """RandomCrop(28, padding=2) on raw [0,1] images: zero-pad by 2, crop back to 28x28."""
    B = x_raw.shape[0]
    padded = F.pad(x_raw, (2, 2, 2, 2))[:, 0]  # B x 32 x 32
    ox = torch.randint(0, 5, (B,), generator=gen).to(DEVICE)
    oy = torch.randint(0, 5, (B,), generator=gen).to(DEVICE)
    ar = torch.arange(28, device=DEVICE)
    rows = oy[:, None] + ar[None, :]
    cols = ox[:, None] + ar[None, :]
    bi = torch.arange(B, device=DEVICE)
    out = padded[bi[:, None, None], rows[:, :, None], cols[:, None, :]]
    return out.unsqueeze(1)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# SAM (non-adaptive, Foret et al. 2021) on top of a base SGD optimiser
# ----------------------------------------------------------------------------

def sam_perturb(params: list[torch.nn.Parameter], rho: float) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Add e_p = rho * g_p / (||g|| + 1e-12) in place; return the e_p list and ||g||."""
    grads = [p.grad for p in params]
    norm = torch.sqrt(torch.stack([g.pow(2).sum() for g in grads]).sum())
    scale = rho / (norm + 1e-12)
    es = [g * scale for g in grads]
    with torch.no_grad():
        for p, e in zip(params, es):
            p.add_(e)
    return es, norm


def sam_step(model: nn.Module, base: torch.optim.Optimizer, xb: torch.Tensor,
             yb: torch.Tensor, rho: float, diag: dict | None = None) -> tuple[torch.Tensor, int]:
    """One SAM update. Returns (first-pass loss, forward-backward passes used = 2).

    If diag is a dict, it receives the BatchNorm buffers before and after the
    second pass so that the check stage can confirm they are unchanged.
    """
    params = list(model.parameters())
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

    base.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    es, _ = sam_perturb(params, rho)

    saved_mom = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0  # running_mean = (1-0)*running + 0*batch: statistics frozen
    if diag is not None:
        diag["before"] = [b.detach().clone() for m in bns for b in (m.running_mean, m.running_var)]

    base.zero_grad(set_to_none=True)
    loss2 = F.cross_entropy(model(xb), yb)
    loss2.backward()

    if diag is not None:
        diag["after"] = [b.detach().clone() for m in bns for b in (m.running_mean, m.running_var)]
    for m, mo in zip(bns, saved_mom):
        m.momentum = mo

    with torch.no_grad():
        for p, e in zip(params, es):
            p.sub_(e)
    base.step()
    return loss.detach(), 2


def sgd_step(model: nn.Module, base: torch.optim.Optimizer, xb: torch.Tensor,
             yb: torch.Tensor) -> tuple[torch.Tensor, int]:
    base.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    base.step()
    return loss.detach(), 1


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(model: nn.Module, data: Data, x_raw: torch.Tensor) -> torch.Tensor:
    model.eval()
    out = []
    for s in range(0, x_raw.shape[0], 1000):
        probs = F.softmax(model(data.norm(x_raw[s:s + 1000])), dim=1)  # float32 on DEVICE (MPS has no float64)
        out.append(probs.cpu().double())
    return torch.cat(out)


def metrics(probs: torch.Tensor, y: torch.Tensor) -> dict:
    y_cpu = y.cpu()
    p_true = probs[torch.arange(len(y_cpu)), y_cpu].clamp_min(1e-12)
    onehot = F.one_hot(y_cpu, 10).double()
    correct = (probs.argmax(1) == y_cpu).numpy()
    return {
        "acc": float(correct.mean()),
        "nll": float((-torch.log(p_true)).mean().item()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean().item()),
        "correct": correct,
    }


def sync() -> None:
    if DEVICE.type == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_run(data: Data, *, arm: str, lr: float, rho: float | None, epochs: int,
              seed: int, tag: str, ckpt_dir: Path) -> dict:
    """Train one model; write results/runs/<tag>.json and checkpoint; skip if already done."""
    RUNS.mkdir(parents=True, exist_ok=True)
    rec_path = RUNS / f"{tag}.json"
    if rec_path.exists():
        logger.info(f"[skip] {tag}: record exists")
        return json.loads(rec_path.read_text())

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}.pt"
    start_iso = datetime.now().isoformat(timespec="seconds")
    logger.info(f"[start] {tag}: arm={arm} lr={lr} rho={rho} epochs={epochs} seed={seed} device={DEVICE}")

    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    count = n_params(model)
    assert 0.1e6 <= count <= 0.3e6, f"parameter count {count} outside 0.1M-0.3M"
    base = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)

    n = data.train_x.shape[0]
    fb_count = 0
    curve: list[dict] = []
    epoch_secs: list[float] = []
    for ep in range(epochs):
        model.train()
        sync()
        t0 = time.time()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        losses = []
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb = data.norm(random_crop(data.train_x[idx], gen))
            yb = data.train_y[idx]
            if arm == "sgd":
                loss, k = sgd_step(model, base, xb, yb)
            else:
                loss, k = sam_step(model, base, xb, yb, rho)
            fb_count += k
            losses.append(loss)
        sync()
        ep_sec = time.time() - t0
        epoch_secs.append(ep_sec)
        sched.step()
        train_loss = float(torch.stack(losses).mean().item())
        vm = metrics(predict_probs(model, data, data.val_x), data.val_y)
        curve.append({"epoch": ep + 1, "train_loss": train_loss, "epoch_seconds": ep_sec,
                      "val_acc": vm["acc"], "val_nll": vm["nll"]})
        logger.info(f"  {tag} ep {ep + 1}/{epochs}: loss={train_loss:.4f} val_acc={vm['acc']:.4f} "
                    f"val_nll={vm['nll']:.4f} {ep_sec:.1f}s")
        if not np.isfinite(train_loss):
            logger.error(f"{tag}: non-finite training loss at epoch {ep + 1}")
            break

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)
    final = curve[-1]
    rec = {
        "tag": tag, "arm": arm, "lr": lr, "rho": rho, "epochs": epochs, "seed": seed,
        "device": str(DEVICE), "params": count, "fb_count": fb_count,
        "train_seconds": float(sum(epoch_secs)),
        "mean_epoch_seconds": float(np.mean(epoch_secs)),
        "final_val_acc": final["val_acc"], "final_val_nll": final["val_nll"],
        "epochs_completed": len(curve), "val_curve": curve,
        "checkpoint": str(ckpt_path.relative_to(WS)),
        "start": start_iso, "end": datetime.now().isoformat(timespec="seconds"),
    }
    rec_path.write_text(json.dumps(rec, indent=1))
    with (RES / "run_log.jsonl").open("a") as f:
        f.write(json.dumps({"tag": tag, "start": start_iso, "end": rec["end"],
                            "train_seconds": rec["train_seconds"]}) + "\n")
    logger.info(f"[done] {tag}: val_acc={rec['final_val_acc']:.4f} val_nll={rec['final_val_nll']:.4f} "
                f"fb={fb_count} train_s={rec['train_seconds']:.0f}")
    del model, base
    gc.collect()
    return rec


def test_eval(data: Data, *, name: str, ckpt: Path, run_rec: dict) -> dict:
    """Evaluate the final-epoch weights on the test set once; store metrics and per-example correctness."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEST_DIR / f"{name}.json"
    if out_path.exists():
        logger.info(f"[skip] test {name}: already evaluated")
        return json.loads(out_path.read_text())
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m = metrics(predict_probs(model, data, data.test_x), data.test_y)
    np.save(TEST_DIR / f"{name}_correct.npy", m["correct"])
    rec = {
        "name": name, "tag": run_rec["tag"], "arm": run_rec["arm"], "lr": run_rec["lr"],
        "rho": run_rec["rho"], "epochs": run_rec["epochs"], "seed": run_rec["seed"],
        "test_acc": m["acc"], "test_nll": m["nll"], "test_brier": m["brier"],
        "final_val_acc": run_rec["final_val_acc"], "final_val_nll": run_rec["final_val_nll"],
        "fb_count": run_rec["fb_count"], "train_seconds": run_rec["train_seconds"],
        "mean_epoch_seconds": run_rec["mean_epoch_seconds"], "device": run_rec["device"],
        "correct_file": f"results/test/{name}_correct.npy",
    }
    out_path.write_text(json.dumps(rec, indent=1))
    logger.info(f"[test] {name}: acc={m['acc']:.4f} nll={m['nll']:.4f} brier={m['brier']:.4f}")
    del model
    return rec


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------

def stage_checks(data: Data) -> None:
    logger.info("=== checks ===")
    results: dict = {}
    torch.manual_seed(0)
    model = SmallCNN().to(DEVICE)
    count = n_params(model)
    assert 0.1e6 <= count <= 0.3e6, count
    results["param_count"] = count

    x = torch.rand(4, 1, 28, 28, device=DEVICE)
    cropped = random_crop(x, torch.Generator().manual_seed(0))
    assert tuple(cropped.shape) == (4, 1, 28, 28), cropped.shape
    results["crop_output_shape"] = list(cropped.shape)

    xb = data.norm(data.train_x[:BATCH])
    yb = data.train_y[:BATCH]
    params = list(model.parameters())
    base = torch.optim.SGD(model.parameters(), lr=0.05, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    model.train()
    base.zero_grad(set_to_none=True)
    F.cross_entropy(model(xb), yb).backward()
    snapshot = [p.detach().clone() for p in params]
    es, _ = sam_perturb(params, 0.05)
    norm_e = torch.sqrt(sum((e ** 2).sum() for e in es)).item()
    assert abs(norm_e - 0.05) < 1e-6, norm_e
    results["sam_perturbation_norm_at_rho_0.05"] = norm_e
    with torch.no_grad():
        for p, e in zip(params, es):
            p.sub_(e)
    max_restore_err = max((p.detach() - s).abs().max().item() for p, s in zip(params, snapshot))
    assert max_restore_err < 1e-6, max_restore_err
    results["sam_perturbation_restore_max_abs_err"] = max_restore_err

    diag: dict = {}
    _, k_sam = sam_step(model, base, xb, yb, 0.05, diag=diag)
    bn_unchanged = all(torch.equal(a, b) for a, b in zip(diag["before"], diag["after"]))
    assert bn_unchanged, "BatchNorm running statistics changed during the second SAM pass"
    results["bn_stats_unchanged_in_second_pass"] = bool(bn_unchanged)
    _, k_sgd = sgd_step(model, base, xb, yb)
    assert (k_sgd, k_sam) == (1, 2), (k_sgd, k_sam)
    results["fb_count_per_step_sgd_sam"] = [k_sgd, k_sam]

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "checks.json").write_text(json.dumps(results, indent=1))
    logger.info(f"checks passed: {results}")


def stage_smoke(data: Data) -> None:
    logger.info("=== smoke (1 epoch per arm) ===")
    sgd = train_run(data, arm="sgd", lr=0.1, rho=None, epochs=1, seed=99,
                    tag="smoke_sgd", ckpt_dir=CK / "smoke")
    sam = train_run(data, arm="sam", lr=0.1, rho=0.05, epochs=1, seed=99,
                    tag="smoke_sam", ckpt_dir=CK / "smoke")
    ratio = sam["mean_epoch_seconds"] / sgd["mean_epoch_seconds"]
    logger.info(f"smoke: SGD {sgd['mean_epoch_seconds']:.1f}s/epoch, SAM {sam['mean_epoch_seconds']:.1f}s/epoch, "
                f"ratio {ratio:.2f}")
    logger.info(f"smoke: SGD train-acc proxy (val acc after 1 epoch) {sgd['final_val_acc']:.4f}, "
                f"SAM {sam['final_val_acc']:.4f}")


def parse_list(s: str, cast=float) -> list:
    return [cast(v) for v in s.split(",") if v.strip()]


def stage_tune(data: Data, sgd_lrs: list[float], sam_lrs: list[float], rhos: list[float]) -> None:
    logger.info("=== tune (seed 0, validation split only) ===")
    tdir = CK / "tuning"
    table = []
    for lr in sgd_lrs:
        r = train_run(data, arm="sgd", lr=lr, rho=None, epochs=EPOCHS_N, seed=TUNE_SEED,
                      tag=f"tune_sgd30_lr{lr}", ckpt_dir=tdir)
        table.append(r)
    for lr in sam_lrs:
        for rho in rhos:
            r = train_run(data, arm="sam", lr=lr, rho=rho, epochs=EPOCHS_N, seed=TUNE_SEED,
                          tag=f"tune_sam30_lr{lr}_rho{rho}", ckpt_dir=tdir)
            table.append(r)
    for lr in sgd_lrs:
        r = train_run(data, arm="sgd", lr=lr, rho=None, epochs=EPOCHS_2N, seed=TUNE_SEED,
                      tag=f"tune_sgd60_lr{lr}", ckpt_dir=tdir)
        table.append(r)

    def best(arm: str, epochs: int) -> dict:
        cands = [r for r in table if r["arm"] == arm and r["epochs"] == epochs]
        return min(cands, key=lambda r: r["final_val_nll"])

    sel = {"sgd30": best("sgd", EPOCHS_N), "sam30": best("sam", EPOCHS_N), "sgd60": best("sgd", EPOCHS_2N)}
    selection = {k: {"tag": v["tag"], "lr": v["lr"], "rho": v["rho"], "epochs": v["epochs"],
                     "val_nll": v["final_val_nll"], "val_acc": v["final_val_acc"]} for k, v in sel.items()}
    (RES / "tuning.json").write_text(json.dumps({
        "selection_rule": "lowest final-epoch validation NLL on the 5,000-image validation split, seed 0",
        "runs": [{"tag": r["tag"], "arm": r["arm"], "lr": r["lr"], "rho": r["rho"], "epochs": r["epochs"],
                  "val_nll": r["final_val_nll"], "val_acc": r["final_val_acc"], "fb_count": r["fb_count"],
                  "train_seconds": r["train_seconds"]} for r in table],
        "selected": selection,
    }, indent=1))
    (RES / "selection.json").write_text(json.dumps(selection, indent=1))
    logger.info(f"selected: {selection}")


def load_selection() -> dict:
    return json.loads((RES / "selection.json").read_text())


def stage_final(data: Data) -> None:
    logger.info("=== final runs (seeds 0,1,2) ===")
    sel = load_selection()
    fdir = CK / "final"
    for seed in FINAL_SEEDS:
        if seed == TUNE_SEED:
            # seed-0 final runs are the selected tuning runs (same code and seed): reuse, do not rerun
            specs = {
                "sam30": (RUNS / f"{sel['sam30']['tag']}.json"),
                "sgd30": (RUNS / f"{sel['sgd30']['tag']}.json"),
                "sgd60": (RUNS / f"{sel['sgd60']['tag']}.json"),
            }
            recs = {k: json.loads(v.read_text()) for k, v in specs.items()}
        else:
            recs = {
                "sam30": train_run(data, arm="sam", lr=sel["sam30"]["lr"], rho=sel["sam30"]["rho"],
                                   epochs=EPOCHS_N, seed=seed, tag=f"final_sam30_seed{seed}", ckpt_dir=fdir),
                "sgd30": train_run(data, arm="sgd", lr=sel["sgd30"]["lr"], rho=None,
                                   epochs=EPOCHS_N, seed=seed, tag=f"final_sgd30_seed{seed}", ckpt_dir=fdir),
                "sgd60": train_run(data, arm="sgd", lr=sel["sgd60"]["lr"], rho=None,
                                   epochs=EPOCHS_2N, seed=seed, tag=f"final_sgd60_seed{seed}", ckpt_dir=fdir),
            }
        for arm, rec in recs.items():
            test_eval(data, name=f"{arm}_seed{seed}", ckpt=ROOT_DIR / rec["checkpoint"], run_rec=rec)


ROOT_DIR = WS


def stage_repro(data: Data) -> None:
    logger.info("=== reproducibility check (seed 0 SGD_30 rerun) ===")
    sel = load_selection()
    orig = json.loads((TEST_DIR / "sgd30_seed0.json").read_text())
    rec = train_run(data, arm="sgd", lr=sel["sgd30"]["lr"], rho=None, epochs=EPOCHS_N, seed=TUNE_SEED,
                    tag="repro_sgd30_seed0", ckpt_dir=CK / "repro")
    test = test_eval(data, name="repro_sgd30_seed0", ckpt=ROOT_DIR / rec["checkpoint"], run_rec=rec)
    diff_pp = abs(test["test_acc"] - orig["test_acc"]) * 100
    out = {
        "original_test_acc": orig["test_acc"], "rerun_test_acc": test["test_acc"],
        "abs_difference_pp": diff_pp,
        "exact_match": bool(diff_pp == 0.0),
        "within_0.1pp": bool(diff_pp <= 0.1),
        "device": str(DEVICE),
    }
    (RES / "repro_check.json").write_text(json.dumps(out, indent=1))
    logger.info(f"repro: {out}")


@logger.catch(reraise=True)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["checks", "smoke", "tune", "final", "repro", "all"])
    ap.add_argument("--sgd-lrs", default="0.05,0.1")
    ap.add_argument("--sam-lrs", default="0.05,0.1")
    ap.add_argument("--rhos", default="0.02,0.05,0.1")
    args = ap.parse_args()
    start_ram_watchdog(RAM_CAP_GB)
    logger.info(f"device={DEVICE} workspace={WS}")
    data = Data()
    sgd_lrs = parse_list(args.sgd_lrs)
    sam_lrs = parse_list(args.sam_lrs)
    rhos = parse_list(args.rhos)
    if args.stage in ("checks", "all"):
        stage_checks(data)
    if args.stage == "smoke":
        stage_smoke(data)
    if args.stage in ("tune", "all"):
        stage_tune(data, sgd_lrs, sam_lrs, rhos)
    if args.stage in ("final", "all"):
        stage_final(data)
    if args.stage in ("repro", "all"):
        stage_repro(data)
    logger.info(f"stage {args.stage} finished")


if __name__ == "__main__":
    main()
