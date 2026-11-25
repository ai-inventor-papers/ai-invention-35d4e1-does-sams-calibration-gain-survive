"""Independent re-derivation of headline numbers from raw per-example correctness files.

Different code path from analysis.py: reads only results/test/*_correct.npy and the
raw test JSON (for NLL/Brier untouched), computes differences with plain Python loops,
and runs a sign-flip placebo test on the seed-level differences.
"""
import json, random, math
from pathlib import Path
W = Path(__file__).resolve().parent
T = W / "results" / "test"

def correct(name):
    import numpy as np  # only for loading .npy
    return [bool(x) for x in np.load(T / f"{name}_correct.npy")]

def acc(v):
    return sum(v) / len(v)

seeds = [0, 1, 2]
sam = {s: correct(f"sam30_seed{s}") for s in seeds}
sgd30 = {s: correct(f"sgd30_seed{s}") for s in seeds}
sgd60 = {s: correct(f"sgd60_seed{s}") for s in seeds}

d30 = [(acc(sam[s]) - acc(sgd30[s])) * 100 for s in seeds]
d60 = [(acc(sam[s]) - acc(sgd60[s])) * 100 for s in seeds]
print("arm accs (%):")
for lab, d in [("sam30", sam), ("sgd30", sgd30), ("sgd60", sgd60)]:
    print(" ", lab, [round(acc(d[s]) * 100, 2) for s in seeds])
m30, m60 = sum(d30) / 3, sum(d60) / 3
sd30 = math.sqrt(sum((x - m30) ** 2 for x in d30) / 2)
sd60 = math.sqrt(sum((x - m60) ** 2 for x in d60) / 2)
print(f"d30 per seed {[round(x,3) for x in d30]} mean {m30:.3f} sd {sd30:.3f}")
print(f"d60 per seed {[round(x,3) for x in d60]} mean {m60:.3f} sd {sd60:.3f}")

# Seed-level two-sided paired t-test (df=2) computed from scratch via the t CDF-free route:
# with df=2 the two-sided p has the closed form p = 1 - t/sqrt(2 + t^2).
def p_two_sided_df2(d, m, sd):
    t = m / (sd / math.sqrt(3))
    return 1 - abs(t) / math.sqrt(2 + t * t), t

p30, t30 = p_two_sided_df2(d30, m30, sd30)
p60, t60 = p_two_sided_df2(d60, m60, sd60)
print(f"t30={t30:.3f} p30={p30:.4f}  t60={t60:.3f} p60={p60:.4f}")

# Holm across 2 comparisons
p_sorted = sorted([("d30", p30), ("d60", p60)], key=lambda x: x[1])
h1 = min(1, 2 * p_sorted[0][1])
h2 = max(h1, min(1, 1 * p_sorted[1][1]))
print("Holm:", {p_sorted[0][0]: round(h1, 4), p_sorted[1][0]: round(h2, 4)})

# Placebo: sign-flip test over the 3 seed differences (8 sign patterns, exact). A valid test
# of "no gain" should NOT show significance when the sign of each difference is random.
# Here we check the seed-level t-statistic on PERMUTED/shuffled input: flip signs at random
# on d30 (placebo) and count how often |t| >= 4.303 (df=2, two-sided 0.05).
random.seed(0)
hits = 0
N = 100000
for _ in range(N):
    pd = [x * random.choice([-1, 1]) for x in d30]
    mm = sum(pd) / 3
    ss = math.sqrt(sum((x - mm) ** 2 for x in pd) / 2)
    if ss > 0 and abs(mm / (ss / math.sqrt(3))) >= 4.303:
        hits += 1
print(f"placebo (random sign flips of d30): fraction with |t|>=4.303 = {hits / N:.4f} (expected ~0.05 or lower)")

# Placebo 2: shuffle SAM labels within each seed's per-example correctness (breaks pairing)
rng = random.Random(1)
pl = []
for s in seeds:
    a = sam[s][:]
    rng.shuffle(a)
    pl.append((acc(a) - acc(sgd30[s])) * 100)
print("placebo (shuffled SAM correctness vs SGD-30):", [round(x, 3) for x in pl],
      "mean", round(sum(pl) / 3, 3))

# Pooled bootstrap CI recomputed independently with a different resampling implementation
rng2 = random.Random(20240601)
n = 10000
boot = []
for _ in range(2000):  # 2000 for speed in this audit; the reported figure uses 10000
    idx = [rng2.randrange(n) for _ in range(n)]
    vals = []
    for s in seeds:
        a = sum(sam[s][i] for i in idx) / n
        b = sum(sgd30[s][i] for i in idx) / n
        vals.append((a - b) * 100)
    boot.append(sum(vals) / 3)
boot.sort()
print(f"pooled d30 bootstrap 2.5/97.5 (2000 resamples, pure python): {boot[50]:.3f}, {boot[1949]:.3f}")
