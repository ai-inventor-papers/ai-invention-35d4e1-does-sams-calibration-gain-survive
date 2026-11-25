# Does SAM's calibration edge survive temperature scaling?

Sharpness-aware minimization (SAM) versus plain SGD for a small CNN on Fashion-MNIST, under a matched gradient budget.
Metrics are reported before and after temperature scaling (Guo et al., 2017). Runs one at a time on the Apple GPU
(`mps`), CPU fallback.

## Result (5 seeds, official 10,000-image test split)

Condition A = SGD, N = 30 epochs. B = SAM, N = 30 epochs, rho = 0.02, lr = 0.05 (two gradient evaluations per step).
C = SGD, 2N = 60 epochs (matched gradient budget, the primary control). Mean ± SD over seeds 0–4.

| Metric | A: SGD, N | B: SAM, N | C: SGD, 2N |
|---|---|---|---|
| Accuracy | 0.9280 ± 0.0022 | 0.9267 ± 0.0013 | 0.9322 ± 0.0015 |
| ECE, raw (15 bins) | 0.0099 ± 0.0012 | 0.0052 ± 0.0017 | 0.0119 ± 0.0014 |
| ECE, temperature-scaled | 0.0078 ± 0.0016 | 0.0063 ± 0.0013 | 0.0073 ± 0.0014 |
| NLL, scaled | 0.1995 ± 0.0027 | 0.2042 ± 0.0036 | 0.1897 ± 0.0035 |
| Brier, scaled | 0.1042 ± 0.0016 | 0.1068 ± 0.0021 | 0.0990 ± 0.0018 |
| AUROC (misclassification), raw | 0.9261 ± 0.0035 | 0.9242 ± 0.0010 | 0.9277 ± 0.0009 |
| Temperature T | 1.030 ± 0.014 | 0.970 ± 0.008 | 1.084 ± 0.032 |

Paired differences (seed mean; sign per seed in order 0–4):

| Comparison | Metric | Mean difference | SD over seeds | Signs |
|---|---|---|---|---|
| P1: B − C (primary) | Raw ECE | −0.0067 | 0.0023 | −−−−− |
| P1: B − C | Scaled ECE | −0.0010 | 0.0026 | −++−− |
| P1: B − C | Accuracy | −0.0055 | 0.0020 | −−−−− |
| P1: B − C | Scaled NLL | +0.0145 | 0.0043 | +++++ |
| P2: B − A (matched epochs) | Raw ECE | −0.0047 | 0.0010 | −−−−− |
| P2: B − A | Scaled ECE | −0.0014 | 0.0013 | −+−−− |

Reading:
- SAM has lower **raw** ECE than matched-compute SGD on every seed (about 0.007 absolute, about 0.7 points).
- That raw-ECE edge **does not survive temperature scaling**. The scaled ECE gap is −0.0010 with seed SD 0.0026 and
  mixed signs. The scaled ECE gap is smaller than the seed-to-seed spread, so the design cannot distinguish it from zero.
  The matched-compute control C is already well calibrated after scaling (T = 1.08), and the temperature fit removes most
  of its raw miscalibration.
- SAM is worse than C on accuracy (about 0.55 points), scaled NLL and Brier on every seed. Its AUROC for error detection
  is lower and raw and scaled AUROC are nearly identical, so temperature scaling does not change error ranking much.
- Bootstrap intervals (2,000 test-index resamples) reflect test sampling only and are narrower than the seed-level
  variation; the decision above rests on the seed-level signs and SDs, as the protocol specifies.

## Layout

- `method.py`: the full pipeline. Stages: `selftest`, `smoke`, `timecheck`, `tune`, `final`, `analyze`, `all`.
- `make_figures.py`: builds the figure specs from the saved outputs and renders them with `aii-data-fig-gen`
  (set `CHART_GEN` to the renderer's `chart_gen.py`).
- `audit_recompute.py`: independent re-derivation of the paired results from the raw logits, with a placebo test on
  shuffled labels. Output: `results/audit_recompute.json`.
- `make_json_variants.py`: writes `full_`, `mini_` and `preview_method_out.json`.
- `pyproject.toml`: pinned dependencies (35 pins, from the run's venv). `reproducibility.md`: step-by-step reproduction.
- `method_out.json`: output in the exp_gen_sol_out format. `metadata` holds the config, deviations, tuning, timings,
  per-model metrics, paired differences with bootstrap intervals and Holm-adjusted p-values, and seed SDs of C.
  `datasets[0].examples` lists the 10,000 test images with the true label and each model's prediction.
- `results/`: `config.json` (selected N, lr and rho), `timecheck.json`, `tuning.json` (validation-only grid),
  `split_indices.json` (55,000/5,000 stratified split), `train_logs/` (per-epoch loss, validation NLL, seconds),
  `logits/` (float32 validation and test logits for every final model).
- `checkpoints/`: final and tuning model weights (kept).
- `reliability/`: 15-bin reliability data per model, raw and scaled, on the test set.
- `reserved/corruptions/`: logits on the reserved corrupted test sets (Gaussian noise 0.1 and 0.2, 3×3 Gaussian blur
  with sigma 1.0, contrast 0.5). No metric is computed on these in this artifact.
- `figures/`: `fig1_ece_paired_differences` (paired ECE differences, mean ± 1 seed SD) and `fig2_reliability_scaled`
  (test-set reliability after scaling, bin means averaged over seeds; the low-confidence bins hold few examples, so they
  are noisy). Each figure has its spec in `figures/specs/`.
- `logs/`: run logs for the main run and the extension seeds.

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python torch torchvision numpy scipy scikit-learn loguru psutil matplotlib jsonschema
.venv/bin/python method.py --stage selftest          # unit checks: ECE, temperature recovery, SAM perturbation
.venv/bin/python method.py --stage smoke             # 1-epoch end-to-end pass on a subset, writes scratch/smoke
.venv/bin/python method.py --stage all --seeds 0 1 2 # time check, tuning, seeds 0-2 finals, analysis
.venv/bin/python method.py --stage all --seeds 3 4   # optional extension seeds, then re-analysis over seeds 0-4
.venv/bin/python audit_recompute.py                  # independent re-derivation and placebo check
.venv/bin/python make_json_variants.py               # full, mini and preview JSON
CHART_GEN=/path/to/chart_gen.py .venv/bin/python make_figures.py
```

For the Ubuntu or non-Apple case, see `reproducibility.md`: the Apple `mps` build used in the run is not on PyPI, and
the Ubuntu CPU path has not been tested.

Stages skip work whose outputs exist (timing, tuning, per-model checkpoints), so an interrupted run resumes.

## Deviations from the direction

1. **Hidden layer 64 units, not 128.** The classifier head is Linear(3136, 64), Linear(64, 10), about 0.22M parameters,
   inside the stated 0.1–0.3M range. The 128-unit version would be about 0.42M.
2. **Learning rate chosen at N = 30 and reused for the 2N control C.** Learning rate for SGD (0.05) and SAM (0.05) and
   rho (0.02) were chosen by validation NLL of the uncalibrated model at 30 epochs, seed 0. C uses the SGD learning rate
   unchanged, so it is not favoured.
3. **Five seeds.** The plan's time check passed at N = 30 (measured SGD epoch 8.7 s, projected main run about 2.1 h),
   so seeds 3 and 4 were added for A, B and C, within the 3.5 h protocol cap. Seed 0 is both a tuning seed and a final seed;
   tuning reads only the validation split.
4. **Timing ratio.** The time check measured SAM and SGD epochs at about the same cost (0.97×), against the expected
   about 2×. The first SGD epoch includes MPS warm-up. Median epoch times over all final runs (`results/train_logs/`)
   give a SAM-to-SGD ratio of about 1.4×, though the machine was shared with other agents and epoch times varied
   between about 5 s and 19 s. The protocol's N = 30 decision is unchanged.
5. **AUROC is computed on raw and scaled outputs** rather than assumed invariant under temperature. The values differ
   by at most 0.00014.
6. **Bootstrap p-values** are two-sided sign-based values from the test-index bootstrap. Holm correction is applied
   across the six metrics within each seed (`p_holm_within_family_per_seed`), because only three to five seeds exist to
   correct across.
7. **Gauss-Newton sharpness check** is not part of this artifact.
8. **Augmentation padding** is zero in pixel space (black background), applied before normalisation.
9. **BatchNorm statistics** update on both forward passes of each SAM step (PyTorch default).

## Restoring removed files

- `.venv/` (the Python environment): `uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python torch torchvision numpy scipy scikit-learn loguru psutil matplotlib jsonschema`
- `scratch/` (the Fashion-MNIST download and the smoke-test output): `uv run --python .venv/bin/python -c "from torchvision import datasets; datasets.FashionMNIST(root='scratch/data', train=True, download=True); datasets.FashionMNIST(root='scratch/data', train=False, download=True)"`. The smoke-test output is recreated by `.venv/bin/python method.py --stage smoke`.

Kept on the run's volume and in the published repository: `checkpoints/` (about 15 MB) and `reserved/corruptions/`
(about 14 MB). Logits, reliability files and `method_out.json` are also kept.
