#!/usr/bin/env python3
"""Build chart specs from method_out.json and render them with aii-data-fig-gen."""
import json
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent
GEN = Path("/Volumes/aii-s20001/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py")
FIG = WS / "figures"
FIG.mkdir(exist_ok=True)
meta = json.loads((WS / "method_out.json").read_text())["metadata"]
pd = meta["paired_differences"]
curves = meta["validation_curves"]

forest = {
    "type": "forest", "null_line": 0.0, "width_in": 6.5, "aspect": "4:3",
    "title": "Paired test accuracy difference (SAM 30 minus SGD)",
    "show_title": False,
    "xlabel": "Accuracy difference, points (3-seed mean, 95% bootstrap CI)",
    "categories": ["SAM 30 vs SGD 30", "SAM 30 vs SGD 60"],
    "series": [{
        "label": "Mean difference",
        "values": [pd["d30"]["mean_pp"], pd["d60"]["mean_pp"]],
        "errors": [(pd["d30"]["pooled_bootstrap_ci_95_pp"][1] - pd["d30"]["pooled_bootstrap_ci_95_pp"][0]) / 2,
                   (pd["d60"]["pooled_bootstrap_ci_95_pp"][1] - pd["d60"]["pooled_bootstrap_ci_95_pp"][0]) / 2],
    }],
}

def line_series(label: str, arm: str) -> dict:
    mean = curves[arm]["val_acc_pct_mean"]
    return {"label": label, "x": list(range(1, len(mean) + 1)), "values": mean}

val = {
    "type": "line", "width_in": 6.5, "aspect": "4:3", "show_title": False,
    "xlabel": "Epoch", "ylabel": "Validation accuracy (%)",
    "series": [line_series("SAM, 30 epochs", "sam30"),
               line_series("SGD, 30 epochs", "sgd30"),
               line_series("SGD, 60 epochs", "sgd60")],
}

for name, spec in [("forest_paired_differences", forest), ("validation_curves", val)]:
    spec_path = FIG / f"{name}.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    r = subprocess.run([sys.executable, str(GEN), "--spec", str(spec_path), "--out", str(FIG / name)],
                       capture_output=True, text=True)
    print(name, "exit", r.returncode, r.stdout[-300:], r.stderr[-600:])
