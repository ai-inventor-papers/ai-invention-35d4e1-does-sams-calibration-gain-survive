#!/usr/bin/env python3
"""Analysis: paired differences, bootstrap intervals, Holm, decision rule, curves, timings.

Reads results/ and checkpoints/ written by method.py; writes method_out.json.
Predictions are recomputed from the saved checkpoints on CPU and must reproduce
the stored test accuracies, so the per-example predictions in method_out.json
are verified against the per-model JSON files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy import stats

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))
from method import SmallCNN  # noqa: E402  (module-level code only defines things)

RES = WS / "results"
TEST_DIR = RES / "test"
N_BOOT = 10000
BOOT_SEED = 20240601
T_CRIT_2DF = 4.303  # two-sided 95% t, 2 degrees of freedom (3 seeds)
SEEDS = [0, 1, 2]
ARMS = {"sam30": "SAM, 30 epochs", "sgd30": "SGD, 30 epochs", "sgd60": "SGD, 60 epochs"}
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WS / "logs" / "analysis.log", level="DEBUG")


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def bootstrap_diffs(a_list: list[np.ndarray], b_list: list[np.ndarray]) -> np.ndarray:
    """Paired bootstrap over test examples (same resampled indices for every seed).

    Returns array (n_seeds, N_BOOT) of accuracy differences in percentage points.
    """
    rng = np.random.default_rng(BOOT_SEED)
    n = a_list[0].shape[0]
    out = np.empty((len(a_list), N_BOOT))
    chunk = 500
    for s0 in range(0, N_BOOT, chunk):
        idx = rng.integers(0, n, size=(min(chunk, N_BOOT - s0), n))
        for k, (a, b) in enumerate(zip(a_list, b_list)):
            out[k, s0:s0 + idx.shape[0]] = (a[idx].mean(1) - b[idx].mean(1)) * 100.0
    return out


def predict_classes(ckpt: Path, test_x: torch.Tensor, mean: float, std: float) -> np.ndarray:
    model = SmallCNN()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, test_x.shape[0], 1000):
            preds.append(model((test_x[s:s + 1000] - mean) / std).argmax(1))
    return torch.cat(preds).numpy()


def main() -> None:
    from torchvision import datasets

    logger.info("=== analysis ===")
    norm = json.loads((RES / "norm_stats.json").read_text())
    te = datasets.FashionMNIST(str(WS / "data"), train=False, download=False)
    test_x = te.data.float().div(255.0).unsqueeze(1)
    test_y = te.targets.numpy()

    tests: dict[str, dict] = {}
    corr: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    for arm in ARMS:
        for s in SEEDS:
            name = f"{arm}_seed{s}"
            rec = json.loads((TEST_DIR / f"{name}.json").read_text())
            tests[name] = rec
            corr[name] = np.load(WS / rec["correct_file"])
            run_tag = rec["tag"]
            ckpt = WS / json.loads((RES / "runs" / f"{run_tag}.json").read_text())["checkpoint"]
            p = predict_classes(ckpt, test_x, norm["mean"], norm["std"])
            preds[name] = p
            recomputed = (p == test_y)
            assert np.array_equal(recomputed, corr[name]), f"prediction mismatch for {name}"
            logger.info(f"{name}: test acc {rec['test_acc']:.4f} (verified from checkpoint)")

    # Paired differences per seed (SAM 30 minus SGD 30 and SGD 60), in percentage points
    comparisons = {"d30": ("sam30", "sgd30"), "d60": ("sam30", "sgd60")}
    comp_out: dict[str, dict] = {}
    seed_p: dict[str, float] = {}
    for cname, (a_arm, b_arm) in comparisons.items():
        a_list = [corr[f"{a_arm}_seed{s}"] for s in SEEDS]
        b_list = [corr[f"{b_arm}_seed{s}"] for s in SEEDS]
        per_seed = np.array([(tests[f"{a_arm}_seed{s}"]["test_acc"] - tests[f"{b_arm}_seed{s}"]["test_acc"]) * 100
                             for s in SEEDS])
        boot = bootstrap_diffs(a_list, b_list)
        pooled = boot.mean(0)
        per_seed_ci = [np.percentile(boot[k], [2.5, 97.5]).tolist() for k in range(len(SEEDS))]
        pooled_ci = np.percentile(pooled, [2.5, 97.5]).tolist()
        mean_d = float(per_seed.mean())
        sd_d = float(per_seed.std(ddof=1))
        half = T_CRIT_2DF * sd_d / np.sqrt(len(SEEDS))
        t_p = float(stats.ttest_1samp(per_seed, 0.0).pvalue)
        seed_p[cname] = t_p
        comp_out[cname] = {
            "comparison": f"{ARMS[a_arm]} minus {ARMS[b_arm]}",
            "per_seed_pp": per_seed.tolist(),
            "mean_pp": mean_d,
            "sd_pp": sd_d,
            "seed_level_t_interval_95_pp": [mean_d - half, mean_d + half],
            "seed_level_paired_t_p_two_sided": t_p,
            "per_seed_bootstrap_ci_95_pp": per_seed_ci,
            "mean_of_per_seed_bootstrap_ci_95_pp": np.mean(per_seed_ci, axis=0).tolist(),
            "pooled_bootstrap_ci_95_pp": pooled_ci,
            "pooled_bootstrap_mean_pp": float(pooled.mean()),
        }
        logger.info(f"{cname}: mean {mean_d:+.3f} pp, sd {sd_d:.3f}, seed-t p={t_p:.3f}, "
                    f"pooled CI [{pooled_ci[0]:+.3f}, {pooled_ci[1]:+.3f}]")

    holm_p = holm([seed_p["d30"], seed_p["d60"]])
    comp_out["d30"]["holm_adjusted_p"] = holm_p[0]
    comp_out["d60"]["holm_adjusted_p"] = holm_p[1]

    # Seed spreads
    spreads = {}
    for arm in ARMS:
        accs = np.array([tests[f"{arm}_seed{s}"]["test_acc"] * 100 for s in SEEDS])
        spreads[arm] = {"test_acc_pp_per_seed": accs.tolist(), "mean_pp": float(accs.mean()),
                        "sd_pp": float(accs.std(ddof=1))}
    sgd30_sd = spreads["sgd30"]["sd_pp"]

    # Decision rule
    d30, d60 = comp_out["d30"], comp_out["d60"]
    s1 = (d30["mean_pp"] > sgd30_sd and d30["pooled_bootstrap_ci_95_pp"][0] > 0
          and d30["holm_adjusted_p"] < 0.05 and d30["mean_pp"] >= 0.2)
    s2 = s1 and (d60["pooled_bootstrap_ci_95_pp"][0] <= 0 or d60["mean_pp"] < 0)
    if s2:
        label = "S2: gain at equal epochs, disappears under matched gradient budget"
    elif s1:
        label = "S1: SAM gains accuracy at equal epochs (supported)"
    else:
        label = "S3: no accuracy gain"
    notes = []
    for cname, c in comp_out.items():
        if abs(c["mean_pp"]) < 0.2:
            notes.append(f"{cname} mean difference {c['mean_pp']:+.2f} pp is within 0.2 pp")
    decision = {"label": label, "rule_inputs": {
        "d30_mean_pp": d30["mean_pp"], "sgd30_seed_sd_pp": sgd30_sd,
        "d30_pooled_ci_95_pp": d30["pooled_bootstrap_ci_95_pp"],
        "d30_holm_p": d30["holm_adjusted_p"],
        "d60_pooled_ci_95_pp": d60["pooled_bootstrap_ci_95_pp"], "d60_mean_pp": d60["mean_pp"]},
        "within_0.2pp_notes": notes}
    logger.info(f"DECISION: {label}; notes={notes}")

    # Validation curves
    curves: dict[str, dict] = {}
    runs = {}
    for arm in ARMS:
        per = []
        for s in SEEDS:
            tag = tests[f"{arm}_seed{s}"]["tag"]
            r = json.loads((RES / "runs" / f"{tag}.json").read_text())
            runs[f"{arm}_seed{s}"] = r
            per.append([100 * c["val_acc"] for c in r["val_curve"]])
        L = min(len(x) for x in per)
        per_arr = np.array([x[:L] for x in per])
        curves[arm] = {"epochs": L, "val_acc_pct_per_seed": per_arr.tolist(),
                       "val_acc_pct_mean": per_arr.mean(0).tolist(),
                       "val_acc_pct_sd": per_arr.std(0, ddof=1).tolist()}
    sam_mean = np.array(curves["sam30"]["val_acc_pct_mean"])
    sgd60_mean = np.array(curves["sgd60"]["val_acc_pct_mean"][:30])
    exceed_epochs = [int(e + 1) for e in range(30) if sgd60_mean[e] > sam_mean[e]]
    val_check = {
        "epoch30_val_acc_pct_sgd30_mean": curves["sgd30"]["val_acc_pct_mean"][29],
        "epoch30_val_acc_pct_sam30_mean": curves["sam30"]["val_acc_pct_mean"][29],
        "sgd60_mean_exceeds_sam30_mean_at_epochs_1_to_30": exceed_epochs,
    }

    # Timings
    def mean_sec(arm: str) -> float:
        return float(np.mean([runs[f"{arm}_seed{s}"]["mean_epoch_seconds"] for s in SEEDS]))

    t_sam, t_sgd30, t_sgd60 = mean_sec("sam30"), mean_sec("sgd30"), mean_sec("sgd60")
    timings = {
        "mean_seconds_per_epoch": {"sam30": t_sam, "sgd30": t_sgd30, "sgd60": t_sgd60},
        "ratio_sam30_over_sgd30": t_sam / t_sgd30,
        "note": "MPS timings with a shared GPU; the first epoch of each run includes warm-up, "
                "so per-epoch means are averages over all epochs of each run.",
    }
    fb = {arm: tests[f"{arm}_seed0"]["fb_count"] for arm in ARMS}

    # Per-example predictions for the output file
    examples = []
    keys = [f"{arm}_seed{s}" for arm in ARMS for s in SEEDS]
    for i in range(test_y.shape[0]):
        ex = {"input": f"Fashion-MNIST test image {i}", "output": CLASSES[test_y[i]]}
        for k in keys:
            ex[f"predict_{k}"] = CLASSES[preds[k][i]]
        examples.append(ex)

    tuning = json.loads((RES / "tuning.json").read_text())
    repro = json.loads((RES / "repro_check.json").read_text())
    checks = json.loads((RES / "checks.json").read_text())

    metadata = {
        "method_name": "SAM vs SGD test-accuracy comparison (Fashion-MNIST, small CNN)",
        "title": "Does SAM beat SGD on test accuracy?",
        "config": {
            "dataset": "Fashion-MNIST, official 60,000/10,000 split",
            "validation_split": "5,000 of the 60,000 training images (permutation, generator seed 1234)",
            "model": "Conv(16,3)-BN-ReLU-Pool, Conv(32,3)-BN-ReLU-Pool, FC128, FC10; 207,018 parameters",
            "augmentation": "RandomCrop(28, padding=2) on training batches only; no flips",
            "batch_size": 128, "momentum": 0.9, "weight_decay": 5e-4, "lr_schedule": "cosine, per epoch",
            "sam": "non-adaptive SAM (Foret et al. 2021), BN momentum 0 in the second pass",
            "seeds": SEEDS, "epochs": {"sam30": 30, "sgd30": 30, "sgd60": 60},
            "selected_hyperparameters": json.loads((RES / "selection.json").read_text()),
            "device": str(torch.device("mps" if torch.backends.mps.is_available() else "cpu")),
            "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED,
                          "scope": "test-sampling noise only; training-seed variance is in the seed-level summary"},
        },
        "tuning": tuning,
        "per_model_metrics": {k: {kk: vv for kk, vv in tests[k].items()} for k in keys},
        "paired_differences": comp_out,
        "seed_spreads": spreads,
        "validation_curves": curves,
        "validation_checks": val_check,
        "timings": timings,
        "forward_backward_passes_seed0": fb,
        "decision": decision,
        "reproducibility_check": repro,
        "unit_checks": checks,
        "deviations_from_plan": [
            "Three seeds only (plan's default). Seeds 3 and 4 were not added.",
            "SAM_30 and SGD_30 are compared per epoch, not per gradient budget; the matched-budget reading is SGD_60.",
            "The shifted-test set was not generated (no calibration metrics), as the plan specifies.",
            "Test accuracy, NLL and Brier score are computed once per model on final-epoch weights.",
        ],
    }
    out = {"metadata": metadata, "datasets": [{"dataset": "Fashion-MNIST test set (10,000 images)",
                                               "examples": examples}]}
    (WS / "method_out.json").write_text(json.dumps(out))
    logger.info(f"wrote method_out.json with {len(examples)} examples")


if __name__ == "__main__":
    main()
