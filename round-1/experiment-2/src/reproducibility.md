# Reproducing the SAM vs SGD Fashion-MNIST results

These steps describe what was actually run. Everything was run on macOS (Apple silicon) using PyTorch's MPS backend. The request asked for Ubuntu instructions; they are given at the end, with what would change.

## 1. Getting the artifact
This workspace is published as one folder of a public GitHub repository. Clone the repository, then `cd` into the folder that contains `method.py`, `analysis.py`, `make_figures.py` and `pyproject.toml`. All paths in the code are relative to that folder (anchored on `Path(__file__)`).

Not in the repository: `.venv/` and `data/` (both deleted after the round, see the manifest). `checkpoints/` (about 14 MB, 25 small state dicts) is published with the repository, unless your copy omits it.

## 2. System, Python and packages
- Hardware used: Apple silicon (arm64), MPS GPU through PyTorch, unified memory shared with other processes. The GPU was shared with other jobs during the run, so timings vary.
- Python 3.12 (`requires-python = "==3.12.*"`).
- Create the environment and install the exact pinned versions from `pyproject.toml`:
  ```
  uv venv .venv --python=3.12
  uv pip install --python .venv/bin/python -r pyproject.toml
  ```
  Note on `torch`: the pinned `torch==2.16.0.dev20261116+aii1` and `torchvision==0.30.0.dev20261116` are development builds from the AI Inventor package index, with an MPS build for macOS. They are not on the public PyPI index. If that index is unavailable, install the nightly or closest available PyTorch build for your platform; the results may then differ slightly.
- No environment variables, API keys or external services are needed.

## 3. Data
Fashion-MNIST is downloaded automatically by `torchvision` the first time `method.py` runs, into `data/`. No manual step is needed. The official 60,000/10,000 split is used. The 5,000-image validation split is drawn from the training set with `torch.randperm(60000, generator=manual_seed(1234))` and saved to `results/split_indices.json`.

## 4. Exact commands run, in order
Run from the folder containing `method.py`:
```
.venv/bin/python method.py --stage checks    # unit checks (~2 s); writes results/checks.json
.venv/bin/python method.py --stage smoke     # 1-epoch smoke run of each arm (~1 min); its records were deleted before the full run, so they are not in the published results
.venv/bin/python method.py --stage all       # tuning (10 runs), final seeds 0-2 (9 models), test evaluation, reproducibility rerun (~85 min on a shared GPU)
.venv/bin/python analysis.py                 # paired differences, bootstrap, Holm, decision rule, method_out.json (~1 min)
.venv/bin/python make_figures.py             # renders figures/ from method_out.json via the aii-data-fig-gen generator
.venv/bin/python audit_rederive.py > results/audit_rederive.txt   # independent re-derivation of the headline numbers
```
Defaults used by `method.py` (no flags were changed from the defaults): sgd learning-rate grid `0.05,0.1`, sam learning-rate grid `0.05,0.1`, rho grid `0.02,0.05,0.1`. The fallback grid reductions were not used.

Recorded reference values from the run (test accuracy, %): SAM-30 92.74 / 92.66 / 92.53; SGD-30 93.12 / 93.00 / 92.94; SGD-60 93.44 / 93.11 / 93.33 (seeds 0 / 1 / 2). Decision label: S3 (no accuracy gain; SAM is lower). Reproducibility rerun of seed-0 SGD-30 matched exactly (93.12%).

Expected run-to-run variation: MPS is not guaranteed deterministic. On this machine the rerun matched exactly, but a different machine may differ by up to about 0.1 pp per model.

## 5. Output files and where they appear
- `method_out.json`: per-example test predictions for nine models (schema `exp_gen_sol_out`) and a `metadata` block with the config, tuning table, per-model metrics, paired differences with 95% intervals and Holm-adjusted p, seed spreads, validation curves, timings, the decision label, the reproducibility check and the deviations from plan. The headline numbers are in `metadata.paired_differences`, `metadata.decision`, `metadata.seed_spreads` and `metadata.per_model_metrics`.
- `full_method_out.json`, `mini_method_out.json` (first 3 examples), `preview_method_out.json` (first 3 examples, strings truncated to 200 characters): the same file at three sizes.
- `results/`: `tuning.json`, `selection.json`, `repro_check.json`, `checks.json`, `norm_stats.json`, `split_indices.json`, `run_log.jsonl`, `runs/*.json` (per-epoch validation curves), `test/*.json` and `test/*_correct.npy`, `audit_rederive.txt`.
- `figures/forest_paired_differences.{pdf,png}` and `figures/validation_curves.{pdf,png}`, each with its JSON spec.
- `README.md` describes the layout and the caveats, including the weak seed-level p-value.

## Ubuntu
These steps should work on Ubuntu 22.04 or later with Python 3.12 and `uv` installed, with these changes:
- The code uses `mps` if available and otherwise `cpu`. On Ubuntu the run uses CPU; its speed was not measured here, and it is expected to be much slower than the MPS run.
- The `torch` and `torchvision` pins are macOS development builds and will not install on Ubuntu from the same index. Install a PyTorch release for your platform and remove those two lines from `pyproject.toml`, then keep the other pins.
- The numeric results are expected to match only approximately across platforms, because the GPU kernels differ.

## Private inputs
No user-uploaded files were used. Nothing in this artifact is private.
