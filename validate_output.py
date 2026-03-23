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
    # Validate all four capabilities under output/
    python validate_output.py --all

    # Validate a single capability (finds all output/capability_2_*/ dirs)
    python validate_output.py --capability 2

    # Validate a specific run directory directly
    python validate_output.py output/capability_2_mar_22_11_30am/

    # Validate a single domain file
    python validate_output.py output/capability_2_mar_22_11_30am/hockey.json

    # Use a non-default output root
    python validate_output.py --all --output-dir my_results/
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

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {exc}"]

    if not isinstance(raw, list):
        return ["Top-level value must be a JSON array."]

    if len(raw) == 0:
        errors.append("Array is empty — no records found.")
        return errors

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


def find_capability_dirs(capability_id: int, output_dir: Path) -> list[Path]:
    """Return all output directories matching capability_{id}_* under output_dir."""
    return sorted(
        d for d in output_dir.glob(f"capability_{capability_id}_*") if d.is_dir()
    )


def validate_targets(targets: list[str]) -> tuple[int, int]:
    """Validate a list of file/dir targets. Returns (total_errors, file_count)."""
    files = collect_files(targets)
    if not files:
        print("  No files to validate.")
        return 0, 0

    total_errors = 0
    for path in files:
        errors = validate_file(path)
        if errors:
            print(f"  FAIL  {path}  ({len(errors)} error(s))")
            for err in errors:
                print(f"        {err}")
            total_errors += len(errors)
        else:
            try:
                n = len(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                n = 0
            print(f"  OK    {path}  ({n} record(s))")

    return total_errors, len(files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate benchmark output files against the submission schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "targets",
        nargs="*",
        metavar="FILE_OR_DIR",
        help="Output JSON file(s) or directory/directories to validate.",
    )
    parser.add_argument(
        "--capability", "-c",
        type=int,
        choices=[1, 2, 3, 4],
        metavar="N",
        help="Validate all output directories for capability N (looks in --output-dir).",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Validate output directories for all four capabilities.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Root output directory to search when using --capability or --all (default: output/).",
    )
    args = parser.parse_args()

    if not args.targets and not args.capability and not args.all:
        parser.print_help()
        return 1

    output_dir = Path(args.output_dir)

    # Determine which capability IDs to scan
    if args.all:
        cap_ids = [1, 2, 3, 4]
    elif args.capability:
        cap_ids = [args.capability]
    else:
        cap_ids = []

    # When specific file/dir targets are given, validate them directly
    if args.targets and not cap_ids:
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

    # Capability-scoped validation
    grand_errors = 0
    grand_files = 0

    for cap_id in cap_ids:
        cap_dirs = find_capability_dirs(cap_id, output_dir)
        # Also include any explicit targets alongside --capability
        extra = list(args.targets) if args.targets else []
        targets_for_cap = [str(d) for d in cap_dirs] + extra

        print(f"\n── Capability {cap_id} ──────────────────────────────────────────")
        if not cap_dirs and not extra:
            print(f"  No output directories found under {output_dir}/capability_{cap_id}_*/")
            continue

        errs, nfiles = validate_targets(targets_for_cap)
        grand_errors += errs
        grand_files += nfiles

        if nfiles:
            status = "PASSED" if errs == 0 else "FAILED"
            print(f"  → {status}: {nfiles} file(s), {errs} error(s)")

    print()
    if grand_files == 0:
        print("No files found to validate.")
        return 1

    if grand_errors:
        print(f"OVERALL FAILED — {grand_errors} schema error(s) across {grand_files} file(s).")
        return 1

    print(f"OVERALL PASSED — {grand_files} file(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
