#!/usr/bin/env python3
"""
Validate benchmark output files against the required submission schema.

Each output file must be a JSON array of records matching this structure:

    [
      {
        "uuid":       "<string>",
        "domain":     "<string>",
        "status":     "success" | "error",
        "error":      "<string>",
        "duration_s": <float>,
        "output": [
          {
            "turn_id":  <int>,
            "query":    "<string>",
            "answer":   "<string>",
            "sequence": {
              "tool_call": [
                {"name": "<string>", "arguments": {<object>}}
              ]
            }
          }
        ]
      }
    ]

Usage:
    python validate_output.py results/address.json
    python validate_output.py results/address.json results/hockey.json
    python validate_output.py results/          # validate all .json files in a dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Schema models — keep in sync with examples/quick_start_benchmark/run_benchmark.py
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class Sequence(BaseModel):
    tool_call: list[ToolCall]


class Turn(BaseModel):
    turn_id: int
    query: str
    answer: str
    sequence: Sequence


class OutputRecord(BaseModel):
    uuid: str
    domain: str
    status: Literal["success", "error"]
    error: str
    duration_s: float
    output: list[Turn]


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_file(path: Path) -> list[str]:
    """
    Validate a single output file.  Returns a list of error strings (empty
    list means the file is valid).
    """
    errors: list[str] = []

    # Parse JSON
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {exc}"]

    if not isinstance(raw, list):
        return ["Top-level value must be a JSON array."]

    if len(raw) == 0:
        errors.append("Array is empty — no records found.")
        return errors

    # Validate each record
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"Record {i}: expected an object, got {type(item).__name__}")
            continue
        uuid_label = item.get("uuid", f"<index {i}>")
        try:
            OutputRecord.model_validate(item)
        except ValidationError as exc:
            for e in exc.errors():
                loc = " -> ".join(str(x) for x in e["loc"])
                errors.append(f"Record {i} (uuid={uuid_label}) [{loc}]: {e['msg']}")

    return errors


def collect_files(targets: list[str]) -> list[Path]:
    """Expand file paths and directories into a flat list of .json files.

    When scanning a directory, *_tools.json sidecar files are skipped — they
    use a different schema (tool shortlisting logs) and are not submission
    output.
    """
    paths: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            found = sorted(
                f for f in p.glob("*.json") if not f.name.endswith("_tools.json")
            )
            if not found:
                print(f"  Warning: no output .json files found in {p}")
            paths.extend(found)
        elif p.exists():
            if p.name.endswith("_tools.json"):
                print(f"  Skipping tool-log file: {p}")
            else:
                paths.append(p)
        else:
            print(f"  Warning: path not found: {p}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate benchmark output files against the submission schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="FILE_OR_DIR",
        help="Output JSON file(s) or directory of JSON files to validate.",
    )
    args = parser.parse_args()

    files = collect_files(args.targets)
    if not files:
        print("No files to validate.")
        return 1

    total_errors = 0
    for path in files:
        errors = validate_file(path)
        if errors:
            print(f"FAIL  {path}  ({len(errors)} error(s))")
            for err in errors:
                print(f"      {err}")
            total_errors += len(errors)
        else:
            # Count records for the success message
            try:
                n = len(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                n = 0
            print(f"OK    {path}  ({n} record(s))")

    print()
    if total_errors:
        print(f"FAILED — {total_errors} schema error(s) across {len(files)} file(s).")
        return 1

    print(f"PASSED — {len(files)} file(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
