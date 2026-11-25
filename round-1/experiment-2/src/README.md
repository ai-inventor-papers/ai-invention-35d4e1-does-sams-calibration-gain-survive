# Does SAM beat SGD on test accuracy? (Fashion-MNIST, small CNN)

## What was done
A small CNN (207,018 parameters) was trained on Fashion-MNIST with:
- **SAM, 30 epochs** (non-adaptive SAM, Foret et al. 2021, rho and lr tuned),
- **SGD, 30 epochs** (lr tuned), the equal-epoch baseline,
- **SGD, 60 epochs** (lr tuned), the compute-matched control: it uses the same number of forward-backward passes as SAM at 30 epochs (25,800 each).

Tuning used seed 0 and the 5,000-image validation split; selection was by lowest final validation NLL. Final runs used seeds 0, 1 and 2. The test set was evaluated once per model on final-epoch weights. Metrics: test accuracy, NLL and Brier score.

## Results (test accuracy, %, seeds 0 / 1 / 2)
| Arm | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| SAM, 30 epochs | 92.74 | 92.66 | 92.53 | 92.64 |
| SGD, 30 epochs | 93.12 | 93.00 | 92.94 | 93.02 |
| SGD, 60 epochs | 93.44 | 93.11 | 93.33 | 93.29 |

- SAM minus SGD-30: **-0.38 pp** (seed SD of the difference 0.04; seed-level paired t p = 0.003, Holm-adjusted p = 0.006; pooled bootstrap 95% CI [-0.58, -0.18]).
- SAM minus SGD-60: **-0.65 pp** (pooled bootstrap 95% CI [-0.89, -0.42]; Holm-adjusted p = 0.025).
- Test NLL: SAM 0.203, SGD-30 0.200, SGD-60 0.195. Brier: SAM 0.108, SGD-30 0.105, SGD-60 0.101.
- **Decision rule: S3, no accuracy gain.** In this setting SAM was slightly worse than SGD at both equal epochs and matched gradient budget. It did not improve test accuracy or the NLL/Brier calibration measures.
- Timing: measured SAM/SGD seconds-per-epoch ratio 1.19 (plan expected about 2). The gradient-pass counts match (see above). MPS timings are on a shared GPU and include warm-up.
- Reproducibility: rerunning seed-0 SGD-30 gave the identical test accuracy (93.12% both times, exact match).

## Layout
- `method.py`: data split, model, SAM and SGD steps, unit checks, tuning, final and reproducibility stages.
- `analysis.py`: paired differences, bootstrap intervals (10,000 resamples, seed 20240601), Holm correction, decision rule, validation curves, timings; recomputes predictions from checkpoints and checks them against stored test accuracies.
- `make_figures.py`: builds chart specs and renders them with the aii-data-fig-gen generator.
- `method_out.json`: per-example test predictions for the nine final models (schema `exp_gen_sol_out`, validated), plus config, tuning table, per-model metrics, paired differences, decision and deviations in `metadata`.
- `results/`: `tuning.json`, `selection.json`, `checks.json`, `norm_stats.json`, `split_indices.json`, `repro_check.json`, `run_log.jsonl`, `runs/*.json` (per-run records with per-epoch validation curves), `test/*.json` and `test/*_correct.npy`. Kept.
- `checkpoints/`: state_dicts of all tuning, final and reproducibility models (about 14 MB; stays on the run's volume, not in the published repository). Kept.
- `figures/`: `forest_paired_differences.{pdf,png}` and `validation_curves.{pdf,png}` with JSON specs.
- `logs/`: run and analysis logs.
- `data/`: Fashion-MNIST download (deleted after the round; see below).

## How to run
```
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python torch torchvision numpy scipy loguru psutil matplotlib jsonschema
.venv/bin/python method.py --stage checks
.venv/bin/python method.py --stage all      # tune, final seeds 0-2, reproducibility rerun (about 80 min on a shared MPS GPU)
.venv/bin/python analysis.py
.venv/bin/python make_figures.py
```

## Deviations and limits
- Three seeds only. The plan's seeds 3 and 4 were not added. The seed-level interval is wide and cannot resolve differences below about 0.5 pp; the gap found here (0.38 pp at 30 epochs) is within the range where that limit applies, but its seed-level p-value is small.
- The bootstrap over test examples captures test-sampling noise only. Training-seed variance is in the seed-level summary.
- SAM-30 vs SGD-30 is a per-epoch comparison; the matched-gradient reading is SGD-60.
- Hyperparameter grid was small (2 learning rates; 3 rho values). Selected: SGD-30 lr 0.05; SAM-30 lr 0.05 with rho 0.02 (the smallest rho in the grid); SGD-60 lr 0.05.
- The shifted-test set was not generated (no calibration analysis in this candidate).
- MPS is not guaranteed deterministic; the reproducibility rerun matched exactly on this machine.

## Restoring removed files
- `data/` (Fashion-MNIST, about 83 MB): `uv run python -c "from torchvision import datasets; datasets.FashionMNIST('data', train=True, download=True); datasets.FashionMNIST('data', train=False, download=True)"` (`method.py` also downloads it automatically when run).
- `.venv/` (about 800 MB): `uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python torch torchvision numpy scipy loguru psutil matplotlib jsonschema`.

## Audit of headline numbers (`audit_rederive.py`, output in `results/audit_rederive.txt`)
- Re-derived from the per-example correctness files (a different code path from `analysis.py`): arm accuracies, d30 = -0.377 pp, d60 = -0.650 pp, seed SDs 0.035 and 0.180 pp, Holm-adjusted seed-level p 0.006 and 0.025. These match the reported numbers.
- Pure-Python bootstrap (2,000 resamples, not 10,000): pooled d30 interval [-0.577, -0.180], consistent with the reported [-0.577, -0.177].
- **Caveat on the seed-level p-value.** With three seeds, the exact sign-flip permutation test cannot give a two-sided p below 0.25. The parametric t-test (p = 0.003) assumes the three seed differences are roughly normal, which three values cannot check. The sign of the effect (SAM lower at all three seeds) is consistent, but its seed-level significance should be read as weak. The decision label (S3) is unaffected.
- The shuffle placebo in the audit is vacuous: shuffling correctness within a seed leaves accuracy unchanged. It should not be cited as evidence.
