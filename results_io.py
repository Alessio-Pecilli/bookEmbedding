from __future__ import annotations

import json
import os
import csv
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
    base_dir: Optional[str] = None,
) -> str:
    """
    Append a flat dict row to results/summary.csv.
    Creates the file (with header) if missing.
    Returns absolute path.
    """
    out_dir = _ensure_results_dir(base_dir)
    path = os.path.join(out_dir, filename)
    abs_path = os.path.abspath(path)

    # Flatten nested values to JSON strings if needed.
    flat: Dict[str, str] = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple)):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = "" if v is None else str(v)

    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)
        return abs_path

    # Keep a valid rectangular CSV when later runs add metrics.  The previous
    # implementation wrote a new header schema only implicitly, corrupting the
    # column alignment for consumers of the summary file.
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    fieldnames = old_fieldnames + [key for key in flat if key not in old_fieldnames]
    rows.append(flat)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return abs_path
