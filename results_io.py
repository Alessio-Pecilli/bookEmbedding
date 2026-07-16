from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _ensure_results_dir(base_dir: Optional[str] = None) -> str:
    root = base_dir or os.path.dirname(__file__) or "."
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def utc_timestamp_compact() -> str:
    # Example: 20260716T110312Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_run_json(payload: Dict[str, Any], filename: str | None = None) -> str:
    """
    Save payload to results/run_<timestamp>.json (pretty printed).
    Returns absolute path to the created file.
    """
    out_dir = _ensure_results_dir()
    if "timestamp" not in payload:
        payload = dict(payload)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    if filename is None:
        filename = f"run_{utc_timestamp_compact()}.json"
    path = os.path.join(out_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    return os.path.abspath(path)


def append_summary_csv(
    row: Dict[str, Any],
    filename: str = "summary.csv",
) -> str:
    """
    Append a flat dict row to results/summary.csv.
    Creates the file (with header) if missing.
    Returns absolute path.
    """
    import csv

    out_dir = _ensure_results_dir()
    path = os.path.join(out_dir, filename)
    abs_path = os.path.abspath(path)

    # Flatten nested values to JSON strings if needed.
    flat: Dict[str, str] = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple)):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = "" if v is None else str(v)

    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)

    return abs_path

