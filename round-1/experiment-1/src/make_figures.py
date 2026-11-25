#!/usr/bin/env python3
"""Builds figure specs from method_out.json and the saved reliability files, then renders them with aii-data-fig-gen."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "bin" / "python"
# Path to aii-data-fig-gen's chart_gen.py; set CHART_GEN to point at your copy of the skill.
GEN = Path(os.environ.get("CHART_GEN", "chart_gen.py"))
FIG = HERE / "figures"
SPEC = HERE / "figures" / "specs"


def render(spec: dict, name: str) -> None:
    SPEC.mkdir(parents=True, exist_ok=True)
    path = SPEC / f"{name}.json"
    path.write_text(json.dumps(spec, indent=2))
    cmd = [str(PY), str(GEN), "--spec", str(path), "--out", str(FIG / name)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"render {name}: exit {res.returncode} {res.stdout.strip()[-200:]} {res.stderr.strip()[-300:]}")
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    out = json.loads((HERE / "method_out.json").read_text())
    m = out["metadata"]
    paired = m["paired_differences"]
    p1, p2 = paired["P1_B_minus_C"]["across_seeds"], paired["P2_B_minus_A"]["across_seeds"]
    n_seeds = len(m["seeds_available"])

    # Figure 1: paired ECE differences (percentage points), seed mean with +/- 1 seed SD
    cats = ["Raw ECE, SAM minus SGD 2N", "Scaled ECE, SAM minus SGD 2N",
            "Raw ECE, SAM minus SGD N", "Scaled ECE, SAM minus SGD N"]
    vals = [p1["ece_raw"]["mean"], p1["ece_scaled"]["mean"], p2["ece_raw"]["mean"], p2["ece_scaled"]["mean"]]
    errs = [p1["ece_raw"]["sd_over_seeds"], p1["ece_scaled"]["sd_over_seeds"],
            p2["ece_raw"]["sd_over_seeds"], p2["ece_scaled"]["sd_over_seeds"]]
    forest = {
        "type": "forest",
        "null_line": 0.0,
        "title": "Paired ECE differences (15 bins)",
        "xlabel": "Difference in ECE (percentage points)",
        "aspect": "4:3",
        "categories": cats,
        "series": [{"label": f"Mean over {n_seeds} seeds, error = 1 SD",
                    "values": [round(v * 100, 3) for v in vals],
                    "errors": [round(e * 100, 3) for e in errs]}],
    }
    render(forest, "fig1_ece_paired_differences")

    # Figure 2: reliability diagram on the test set, pooled over seeds (mean per bin across seeds)
    rel_dir = HERE / "reliability"
    series = []
    for cond, label in [("B", "SAM, N epochs (scaled)"), ("C", "SGD, 2N epochs (scaled)")]:
        xs, ys = [], []
        files = sorted(rel_dir.glob(f"{cond}_seed*.json"))
        conf_stack = []
        acc_stack = []
        for f in files:
            r = json.loads(f.read_text())["scaled"]
            conf_stack.append([np.nan if v is None else v for v in r["mean_confidence"]])
            acc_stack.append([np.nan if v is None else v for v in r["accuracy"]])
        conf = np.nanmean(np.array(conf_stack), axis=0)
        acc = np.nanmean(np.array(acc_stack), axis=0)
        for c, a in zip(conf, acc):
            if np.isfinite(c) and np.isfinite(a):
                xs.append(round(float(c), 4))
                ys.append(round(float(a), 4))
        series.append({"label": label, "x": xs, "values": ys})
    series.insert(0, {"label": "Perfect calibration", "x": [0.0, 1.0], "values": [0.0, 1.0]})
    reliab = {
        "type": "line",
        "title": "Reliability on the test set after temperature scaling",
        "xlabel": "Mean confidence in bin",
        "ylabel": "Accuracy in bin",
        "aspect": "4:3",
        "series": series,
    }
    render(reliab, "fig2_reliability_scaled")
    logger.info("Figures written to figures/")


if __name__ == "__main__":
    main()
