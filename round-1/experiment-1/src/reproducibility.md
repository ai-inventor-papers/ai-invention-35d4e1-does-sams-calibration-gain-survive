# Reproducing the SAM vs SGD calibration results

This describes what was actually run, not an idealised setup.

## 1. Getting the artifact

The workspace is published as one folder of a public GitHub repository. Clone the repository and change into this
folder (the folder that contains `method.py`):

```bash
git clone <repository-url>
cd <repository>/<this-artifact-folder>
```

Every path in the code and in this file is relative to this folder. The code finds its own location with `Path(__file__)`.

No input from another artifact is used. The Fashion-MNIST data is downloaded by the code itself (see section 3).
The run's user-uploads folder did not exist at the time of this artifact, so no private input is needed.

## 2. Platform, Python and packages

**What was actually run:** macOS 26.4 on an Apple M3 Max virtual machine (Apple silicon), Python 3.12.14, PyTorch
`mps` backend (Apple GPU), with a 20 GB GPU memory cap. Other agents shared the machine during the run, so epoch times
varied between about 5 s and 19 s.

**Not run here:** Ubuntu. The code selects `mps` when available and falls back to `cpu` otherwise. A CPU run on Ubuntu
has not been tested and would be much slower.

Python: 3.12 (the venv was made with `--python=3.12`). The exact package versions are pinned in `pyproject.toml`
(35 pins, from `uv pip freeze` on the run's venv). Two of them need attention:

- `torch==2.16.0.dev20261116+aii1` and `torchvision==0.30.0.dev20261116` are pre-release builds with the Apple `mps`
  backend supplied by the AI Inventor environment. They are not on PyPI, so the exact pins cannot be installed from a
  public index. Installing public torch gives an equivalent API and the same method, but not byte-identical numbers.

Create the environment and install the pinned set:

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python torch torchvision numpy scipy scikit-learn loguru psutil matplotlib jsonschema
```

To install the exact pinned list from `pyproject.toml` (written to `requirements-pinned.txt`), run:

```bash
.venv/bin/python -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > requirements-pinned.txt
uv pip install --python .venv/bin/python -r requirements-pinned.txt
```

The pinned list includes the two pre-release torch builds noted above, so this step needs the same package index as the run.
On a standard index, install the unpinned packages from the first command instead.

No system packages are needed beyond Python and `uv`.

## 3. Data and credentials

- **Dataset:** Fashion-MNIST, downloaded by torchvision into `scratch/data/` on first run (about 30 MB). The code
  downloads it automatically; the official 60,000 train and 10,000 test images are used. The Hugging Face fallback
  (`zalando-datasets/fashion_mnist`) was not needed.
- **Split:** the 55,000/5,000 train/validation split is written to `results/split_indices.json` by the first run.
- **Environment variables:** none are required. `SAM_DEVICE` (optional) forces `cpu` or `mps`. `CHART_GEN` (optional)
  points at the figure renderer used by `make_figures.py`.
- **API keys:** none. No paid services are used.

## 4. Commands, in the order they were run

Run from this folder with `.venv` active or using `.venv/bin/python`.

```bash
# 4.1 Unit checks and smoke test (smoke writes scratch/smoke/, not results/)
.venv/bin/python method.py --stage smoke

# 4.2 Time check: one SGD and one SAM epoch on the full training set; chooses N = 30 (passed, 8.7 s per SGD epoch)
.venv/bin/python method.py --stage timecheck

# 4.3 Main run: validation-only tuning (8 runs, 30 epochs, seed 0), seeds 0-2 finals (A, B, C), analysis
.venv/bin/python method.py --stage all --seeds 0 1 2

# 4.4 Optional extension seeds 3-4 (A, B, C), then re-analysis over seeds 0-4
.venv/bin/python method.py --stage all --seeds 3 4

# 4.5 Re-derivation audit (independent code path, placebo test)
.venv/bin/python audit_recompute.py

# 4.6 Figures and JSON variants
CHART_GEN=/path/to/aii-data-fig-gen/scripts/chart_gen.py .venv/bin/python make_figures.py
.venv/bin/python make_json_variants.py
```

Stages skip work whose output already exists (`timecheck.json`, `tuning.json`, per-model checkpoints and logits), so an
interrupted run resumes. Delete those files to force a rerun.

**Selected configuration (from `results/tuning.json` and `results/config.json`):** N = 30 epochs, batch 128, SGD learning
rate 0.05 (selected by validation NLL over {0.05, 0.1}), SAM learning rate 0.05 and rho 0.02 (selected over
{0.05, 0.1} x {0.02, 0.05, 0.1}), momentum 0.9, weight decay 5e-4, cosine schedule per iteration, 64-unit hidden layer.

**Hardware and runtime (measured on the run):**

| Stage | Wall time (shared machine) |
|---|---|
| Smoke test and self-tests | about 1 min |
| Time check | about 1 min |
| Tuning (8 runs at 30 epochs) | about 33 min |
| Finals, seeds 0-2 (A, B, C) | about 67 min |
| Extension, seeds 3-4 (A, B, C) and re-analysis | about 40 min |
| Audit (`audit_recompute.py`) | about 5 min |

Total for the full protocol: about 2.2 h. The protocol cap was 3.5 h.

## 5. What a reader should get, and where it appears

- `method_out.json` (exp_gen_sol_out format; `metadata` holds the results, `datasets[0].examples` the 10,000 test images
  with each model's prediction). Key fields:
  - `metadata.config`, `metadata.deviations`, `metadata.tuning`, `metadata.time_check`
  - `metadata.per_model_metrics`: accuracy, ECE (15 bins; 10 and 20 bins in the sensitivity fields), NLL, Brier, AUROC, and the
    temperature T, raw and scaled, per condition and seed
  - `metadata.paired_differences.P1_B_minus_C` (SAM at N vs SGD at 2N, the primary comparison) and
    `P2_B_minus_A` (matched epochs): per-seed point estimates, 95% bootstrap intervals, two-sided bootstrap p-values,
    Holm-adjusted p-values within each seed, and the cross-seed mean, SD and sign pattern
  - `metadata.seed_sd_of_C`
- `full_method_out.json`, `mini_method_out.json` (first 3 examples), `preview_method_out.json` (strings cut to 200
  characters): generated by `make_json_variants.py` from `method_out.json`.
- `results/audit_recompute.json`: the independent re-derivation of the primary comparison and the placebo test.
- `results/logits/{cond}_seed{s}_{val|test}.npy`: float32 logits, the raw data behind every metric.
- `reliability/{cond}_seed{s}.json`: 15-bin reliability data, raw and scaled.
- `reserved/corruptions/{cond}_seed{s}_{corruption}.npy`: logits on the reserved corrupted test sets (no metric computed).
- `checkpoints/{cond}_seed{s}.pt`: final weights. `results/train_logs/`: per-epoch training loss, validation NLL and seconds.
- `figures/fig1_ece_paired_differences` and `figures/fig2_reliability_scaled` (PDF and PNG).

Headline numbers in the write-up (README and final text) come from `method_out.json`. The audit re-derives them from the
raw logits. See `README.md` for the table of means and SDs.

## 6. Known differences between the audit and the main run

- Temperature: the main run uses `scipy.optimize.minimize_scalar`, the audit a dense grid search over log T. The fitted T
  values agree to about 1e-3 and the per-seed scaled differences agree to within 4e-4.
- The scaled-ECE cross-seed SD is 0.0026 in the main run and 0.0024 in the audit, for the same means (the difference comes
  from those per-seed values). The main run's SD is the one reported.
- Bootstrap p-values use different random generators and agree within about 0.05. On seed 2, raw ECE gives p = 0.023 in
  the main run and 0.012 in the audit, the same decision (p < 0.05) on both.
- The placebo test (test labels shuffled at random) gives no rejection at 0.05 on any of the five seeds (smallest p = 0.073).

## Restoring removed files

Removed by the publish step (do not ship): `.venv/` and `scratch/`.

- `.venv/`: `uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python torch torchvision numpy scipy scikit-learn loguru psutil matplotlib jsonschema`
- `scratch/data/` (Fashion-MNIST): `uv run --python .venv/bin/python -c "from torchvision import datasets; datasets.FashionMNIST(root='scratch/data', train=True, download=True); datasets.FashionMNIST(root='scratch/data', train=False, download=True)"`
- `scratch/smoke/`: `.venv/bin/python method.py --stage smoke`
