#!/usr/bin/env python3
"""
Generate apis/configs/mcp_tool_universe_id_mapping.yaml from task_1 output files.

Usage:
    python scripts/generate_mcp_mapping.py [--dry-run]
"""
import json
import sys
import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIRS_GLOB = "data/tasks/task_1/*/output"
MAPPING_FILE = "apis/configs/mcp_tool_universe_id_mapping.yaml"


def generate_mapping(dry_run: bool = False):
    output_dirs = sorted(REPO_ROOT.glob(OUTPUT_DIRS_GLOB))
    if not output_dirs:
        print(
            f"ERROR: No output directories found matching {OUTPUT_DIRS_GLOB}",
            file=sys.stderr,
        )
        sys.exit(1)

    mapping = {}
    duplicate_count = 0

    for output_dir in output_dirs:
        for json_file in sorted(output_dir.glob("*.json")):
            with open(json_file) as f:
                records = json.load(f)

            for record in records:
                uuid = record["uuid"]
                domain = record["domain"]

                if uuid in mapping:
                    duplicate_count += 1
                    continue

                ground_truth = record.get("ground_truth", [])
                if not ground_truth:
                    print(
                        f"WARNING: No ground_truth for uuid {uuid}",
                        file=sys.stderr,
                    )
                    continue

                first_turn = ground_truth[0]
                gold_sequence = first_turn.get("gold_sequence", [])
                if not gold_sequence:
                    print(
                        f"WARNING: No gold_sequence for uuid {uuid}",
                        file=sys.stderr,
                    )
                    continue

                tool_calls = gold_sequence[0].get("tool_call", [])
                init_call = next(
                    (c for c in tool_calls if c["name"] == "initialize_active_data"),
                    None,
                )
                if init_call is None:
                    print(
                        f"WARNING: No initialize_active_data for uuid {uuid}",
                        file=sys.stderr,
                    )
                    continue

                mapping[uuid] = {
                    "domain": domain,
                    "init_args": init_call["arguments"],
                    "query": first_turn["query"],
                }

    if duplicate_count:
        print(
            f"WARNING: Skipped {duplicate_count} duplicate UUID occurrences "
            f"({len(mapping)} unique IDs kept)",
            file=sys.stderr,
        )

    out_path = REPO_ROOT / MAPPING_FILE
    print(f"Generated {len(mapping)} entries")

    if dry_run:
        print(f"DRY RUN: would write to {out_path}")
        return

    with open(out_path, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, allow_unicode=True)

    print(f"Written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    generate_mapping(dry_run=args.dry_run)
