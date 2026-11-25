#!/usr/bin/env python3
"""Writes full_, mini_ and preview_ versions of method_out.json.

Same rules as aii-json's aii_json_format_mini_preview.py (which refuses files outside its own project
root, so its logic is reproduced here):
  full     identical content
  mini     every dataset kept, examples cut to the first 3 per dataset; metadata kept
  preview  mini, with every string longer than 200 characters cut to 200 characters plus '...'
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "method_out.json"
MAX_ITEMS = 3
MAX_STRING = 200
MARK = "..."


def truncate(value):
    if isinstance(value, list):
        return [truncate(v) for v in value]
    if isinstance(value, str):
        return value[:MAX_STRING] + MARK if len(value) > MAX_STRING else value
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def slice_examples(data: dict) -> dict:
    datasets = [{**ds, "examples": ds.get("examples", [])[:MAX_ITEMS]} for ds in data["datasets"]]
    return {**data, "datasets": datasets}


def main() -> None:
    data = json.loads(SRC.read_text())
    full = data
    mini = slice_examples(data)
    preview = truncate(slice_examples(data))
    for name, obj in [("full", full), ("mini", mini), ("preview", preview)]:
        path = HERE / f"{name}_method_out.json"
        path.write_text(json.dumps(obj))
        n = sum(len(ds["examples"]) for ds in obj["datasets"])
        print(f"{path.name}: {n} examples")


if __name__ == "__main__":
    main()
