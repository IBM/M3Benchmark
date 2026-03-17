#!/usr/bin/env python3
"""
Generate Tool Checksums
=======================

Connects to running MCP servers for each capability/domain combination, calls
list_tools(), computes checksums, and writes them to tool_checksums.json.

This script is for REPO OWNERS / SERVER MAINTAINERS only. Clients should
not run this — they read the committed tool_checksums.json.

Re-run this script whenever tool definitions change intentionally (e.g. new
endpoints, renamed parameters) and commit the updated tool_checksums.json.

Requirements:
    - Docker containers must be running (docker compose up -d, or equivalent).
    - The benchmark data directory must be populated (make download).

Usage:
    # Generate checksums for all capabilities and all domains
    python generate_checksums.py

    # Generate for specific capabilities only
    python generate_checksums.py --capability 2 --capability 3

    # Generate for specific domains only (across all capabilities)
    python generate_checksums.py --domain address --domain airline

    # Dry-run: print checksums without writing to file
    python generate_checksums.py --dry-run
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.mcp_client import create_client_and_connect, load_mcp_config
from benchmark.runner_helpers import load_benchmark_data
from environment.tool_checksums import compute_tool_checksum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

CHECKSUMS_PATH = PROJECT_ROOT / "tool_checksums.json"
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "benchmark" / "mcp_connection_config.yaml")

# Capabilities supported for checksum generation (Capability 1 excluded by design).
SUPPORTED_CAPABILITIES = [2, 3, 4]


async def compute_checksums_for_capability(
    capability_id: int,
    mcp_configs: dict,
    domains: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Return {domain: checksum} for a single capability.

    Args:
        capability_id: Capability ID to process.
        mcp_configs:   Loaded MCP connection configs (capability_id → MCPConnectionConfig).
        domains:       If given, only process these domains; otherwise all domains
                       found in the benchmark data directory.

    Returns:
        Dict mapping domain name to its SHA-256 tool checksum.
    """
    cfg = mcp_configs.get(capability_id)
    if cfg is None:
        logger.warning("No MCP config found for capability %s — skipping.", capability_id)
        return {}

    try:
        _, available_domains = load_benchmark_data(
            capability_id=capability_id, domain_names_only=True
        )
    except Exception as e:
        logger.error("Failed to load domain list for capability %s: %s", capability_id, e)
        return {}

    if not available_domains:
        logger.warning("No domains found for capability %s — skipping.", capability_id)
        return {}

    domains_to_process = (
        [d for d in domains if d in available_domains]
        if domains
        else available_domains
    )

    if not domains_to_process:
        logger.warning(
            "None of the requested domains %s are available for capability %s "
            "(available: %s).",
            domains,
            capability_id,
            available_domains,
        )
        return {}

    results: Dict[str, str] = {}

    for domain in domains_to_process:
        logger.info("  [capability %s] domain='%s' — connecting...", capability_id, domain)
        try:
            async with create_client_and_connect(cfg, domain) as session:
                response = await session.list_tools()
                tools = response.tools
                checksum = compute_tool_checksum(tools)
                results[domain] = checksum
                logger.info(
                    "  [capability %s] domain='%s' — %d tools → %s",
                    capability_id,
                    domain,
                    len(tools),
                    checksum[:16] + "...",
                )
        except ExceptionGroup as eg:
            # Cleanup errors from asyncio TaskGroup (non-fatal)
            logger.warning(
                "  [capability %s] domain='%s' — cleanup warning: %s",
                capability_id,
                domain,
                eg,
            )
        except Exception as e:
            logger.error(
                "  [capability %s] domain='%s' — ERROR: %s", capability_id, domain, e
            )

    return results


def load_existing_checksums() -> Dict[str, Dict[str, str]]:
    """Load existing tool_checksums.json, stripping the _comment key."""
    if not CHECKSUMS_PATH.exists():
        return {}
    with open(CHECKSUMS_PATH) as f:
        data = json.load(f)
    data.pop("_comment", None)
    return data


def save_checksums(checksums: Dict[str, Dict[str, str]], dry_run: bool) -> None:
    """Write checksums dict to tool_checksums.json."""
    output = {
        "_comment": (
            "Auto-generated by generate_checksums.py. "
            "Do not edit manually. "
            "Re-run generate_checksums.py after any intentional tool changes "
            "and commit the result."
        ),
    }
    # Emit capabilities in sorted order for deterministic diffs.
    for cap_key in sorted(checksums.keys(), key=lambda k: int(k)):
        output[cap_key] = dict(sorted(checksums[cap_key].items()))

    if dry_run:
        print("\n--- DRY RUN: would write to tool_checksums.json ---")
        print(json.dumps(output, indent=2))
        print("--- END DRY RUN ---\n")
        return

    with open(CHECKSUMS_PATH, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    logger.info("Checksums written to %s", CHECKSUMS_PATH)


async def main(
    capability_ids: Optional[List[int]] = None,
    domains: Optional[List[str]] = None,
    dry_run: bool = False,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> None:
    caps = capability_ids or SUPPORTED_CAPABILITIES
    logger.info("Generating checksums for capabilities: %s", caps)
    if domains:
        logger.info("Filtering to domains: %s", domains)

    mcp_configs = load_mcp_config(config_path)

    # Start from existing checksums so we don't clobber unrelated capabilities/domains.
    all_checksums = load_existing_checksums()
    # Ensure keys for all supported capabilities exist.
    for cid in SUPPORTED_CAPABILITIES:
        all_checksums.setdefault(str(cid), {})

    for capability_id in caps:
        if capability_id not in SUPPORTED_CAPABILITIES:
            logger.warning(
                "Capability %s is not supported for checksum generation "
                "(capability 1 is excluded by design). Skipping.",
                capability_id,
            )
            continue

        logger.info("=== Capability %s ===", capability_id)
        cap_checksums = await compute_checksums_for_capability(
            capability_id, mcp_configs, domains
        )

        # Merge: new values overwrite existing ones for processed domains.
        all_checksums[str(capability_id)].update(cap_checksums)

    total = sum(len(v) for v in all_checksums.values())
    logger.info(
        "Done. Total (capability, domain) pairs with checksums: %d", total
    )

    save_checksums(all_checksums, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tool_checksums.json for MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--capability",
        dest="capabilities",
        type=int,
        action="append",
        metavar="CAPABILITY_ID",
        help=(
            "Capability ID to generate checksums for (repeatable). "
            f"Defaults to all supported capabilities: {SUPPORTED_CAPABILITIES}."
        ),
    )
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        metavar="DOMAIN",
        help=(
            "Domain name to generate checksums for (repeatable). "
            "Defaults to all domains found in the benchmark data directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print checksums to stdout without writing to tool_checksums.json.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to MCP connection config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            capability_ids=args.capabilities,
            domains=args.domains,
            dry_run=args.dry_run,
            config_path=args.config,
        )
    )
