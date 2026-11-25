#!/usr/bin/env python3
"""Independent re-derivation of the headline numbers from the raw saved logits.

Deliberately a different code path from method.py:
  - probabilities from log-softmax (logsumexp), not from exp / sum
  - ECE from an explicit loop over bin masks, not from np.bincount
  - temperature from a dense grid search over log T, not scipy minimize_scalar
  - labels read straight from the Fashion-MNIST files via torchvision, split indices from results/split_indices.json
  - bootstrap with its own generator seeds, so intervals are re-simulated, not copied
Also runs a placebo: the same paired test on test labels shuffled at random, which should NOT reject.
Writes results/audit_recompute.json and prints a comparison against method_out.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from torchvision import datasets

HERE = Path(__file__).resolve().parent
LOG = HERE / "results" / "logits"
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 2000


def log_softmax(z: np.ndarray) -> np.ndarray:
    return z - logsumexp(z, axis=1, keepdims=True)


def ece_loop(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    total = 0.0
    n = len(conf)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        nb = mask.sum()
        if nb:
            total += nb / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def nll_val(logp: np.ndarray, y: np.ndarray, log_t: float) -> float:
    lp = log_softmax(logp / np.exp(log_t))
    return float(-np.mean(np.maximum(lp[np.arange(len(y)), y], np.log(1e-12))))


def fit_T_grid(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.linspace(np.log(0.05), np.log(20.0), 4001)
    vals = np.array([nll_val(logits, y, g) for g in grid])
    return float(np.exp(grid[int(np.argmin(vals))]))


def metrics(logits: np.ndarray, y: np.ndarray, T: float) -> dict:
    lp = log_softmax(logits / T)
    p = np.exp(lp)
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    nll = float(-np.mean(np.maximum(lp[np.arange(len(y)), y], np.log(1e-12))))
    brier = float(np.mean(((p - np.eye(10)[y]) ** 2).sum(1)))
    return {"accuracy": float(correct.mean()), "ece_raw_or_scaled": ece_loop(conf, correct),
            "nll": nll, "brier": brier, "conf": conf, "correct": correct}


def paired_metrics(lb, lc, y, Tb, Tc, idx=None):
    if idx is None:
        idx = np.arange(len(y))
    mb = metrics(lb[idx], y[idx], Tb)
    mc = metrics(lc[idx], y[idx], Tc)
    mbr = metrics(lb[idx], y[idx], 1.0)
    mcr = metrics(lc[idx], y[idx], 1.0)
    return {
        "accuracy": mb["accuracy"] - mc["accuracy"],
        "ece_raw": mbr["ece_raw_or_scaled"] - mcr["ece_raw_or_scaled"],
        "ece_scaled": mb["ece_raw_or_scaled"] - mc["ece_raw_or_scaled"],
        "nll_scaled": mb["nll"] - mc["nll"],
        "brier_scaled": mb["brier"] - mc["brier"],
    }


def boot_p(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def main() -> None:
    tr = datasets.FashionMNIST(root=str(HERE / "scratch" / "data"), train=True, download=False)
    te = datasets.FashionMNIST(root=str(HERE / "scratch" / "data"), train=False, download=False)
    y_train = tr.targets.numpy()
    y_test = te.targets.numpy()
    split = json.loads((HERE / "results" / "split_indices.json").read_text())
    y_val = y_train[np.array(split["validation_indices"])]
    assert len(y_val) == 5000 and len(y_test) == 10000

    out = {"seeds": {}, "P1_B_minus_C": {}, "placebo_permuted_labels": {}}
    for seed in SEEDS:
        lv = {c: np.load(LOG / f"{c}_seed{seed}_val.npy").astype(np.float64) for c in "ABC"}
        lt = {c: np.load(LOG / f"{c}_seed{seed}_test.npy").astype(np.float64) for c in "ABC"}
        T = {c: fit_T_grid(lv[c], y_val) for c in "ABC"}
        out["seeds"][seed] = {"T": T}
        d = paired_metrics(lt["B"], lt["C"], y_test, T["B"], T["C"])
        out["seeds"][seed]["P1_point"] = d
        # bootstrap, independent RNG
        rng = np.random.default_rng(777 + seed)
        boots = {k: np.empty(N_BOOT) for k in d}
        for b in range(N_BOOT):
            idx = rng.integers(0, len(y_test), len(y_test))
            db = paired_metrics(lt["B"], lt["C"], y_test, T["B"], T["C"], idx)
            for k in d:
                boots[k][b] = db[k]
        out["seeds"][seed]["P1_ci95"] = {k: [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))] for k in d}
        out["seeds"][seed]["P1_p_boot"] = {k: boot_p(boots[k]) for k in d}
        # placebo: shuffle test labels, same paired test
        perm = np.random.default_rng(4242 + seed).permutation(y_test)
        dp = paired_metrics(lt["B"], lt["C"], perm, T["B"], T["C"])
        rng2 = np.random.default_rng(999 + seed)
        pb = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng2.integers(0, len(perm), len(perm))
            pb[b] = paired_metrics(lt["B"], lt["C"], perm, T["B"], T["C"], idx)["accuracy"]
        out["placebo_permuted_labels"][seed] = {"accuracy_diff": dp["accuracy"], "p_boot": boot_p(pb)}
        print(f"seed {seed}: T={ {k: round(v, 4) for k, v in T.items()} } "
              f"P1 acc={d['accuracy']:+.4f} rawECE={d['ece_raw']:+.4f} scECE={d['ece_scaled']:+.4f} "
              f"nll={d['nll_scaled']:+.4f} brier={d['brier_scaled']:+.4f} | placebo acc p={out['placebo_permuted_labels'][seed]['p_boot']:.3f}",
              flush=True)

    # cross-seed summary (independent)
    for k in ["accuracy", "ece_raw", "ece_scaled", "nll_scaled", "brier_scaled"]:
        vals = np.array([out["seeds"][s]["P1_point"][k] for s in SEEDS])
        out["P1_B_minus_C"][k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                                  "signs": "".join("+" if v > 0 else "-" for v in vals)}
        print(f"P1 {k}: mean={vals.mean():+.4f} sd={vals.std(ddof=1):.4f} signs={out['P1_B_minus_C'][k]['signs']}")
    (HERE / "results" / "audit_recompute.json").write_text(json.dumps(out, indent=2, default=float))

    # compare with method_out.json
    m = json.loads((HERE / "method_out.json").read_text())["metadata"]
    print("\nComparison with method_out.json (point estimates per seed, P1):")
    worst = 0.0
    for s in SEEDS:
        mp = m["paired_differences"]["P1_B_minus_C"][f"seed{s}"]
        for k_audit, k_main in [("accuracy", "accuracy"), ("ece_raw", "ece_raw"), ("ece_scaled", "ece_scaled"),
                                ("nll_scaled", "nll_scaled"), ("brier_scaled", "brier_scaled")]:
            diff = abs(out["seeds"][s]["P1_point"][k_audit] - mp[k_main]["point"])
            worst = max(worst, diff)
    print(f"max |audit - method| over seeds x 5 metrics = {worst:.2e}")
    cmp_mean = {k: m["paired_differences"]["P1_B_minus_C"]["across_seeds"][k]["mean"] for k in
                ["accuracy", "ece_raw", "ece_scaled", "nll_scaled", "brier_scaled"]}
    print("method mean:", {k: round(v, 4) for k, v in cmp_mean.items()})
    print("audit mean :", {k: round(out["P1_B_minus_C"][k]["mean"], 4) for k in cmp_mean})
    print("placebo rejections (p<0.05) of 5:", sum(v["p_boot"] < 0.05 for v in out["placebo_permuted_labels"].values()))


if __name__ == "__main__":
    main()
