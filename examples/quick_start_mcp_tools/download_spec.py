#!/usr/bin/env python3
"""
Download the OpenAPI spec from the FastAPI backend running inside a capability container.

The M3 REST and Retriever servers expose a standard FastAPI /openapi.json endpoint
on localhost inside the container. This script fetches it via `docker exec`.

Default ports (inside the container):
  - M3 REST (Capabilities 1, 2, 3, 4):   http://localhost:8000/openapi.json
  - Retriever    (Capability 4 only):     http://localhost:8001/openapi.json

Usage:
    python examples/download_spec.py --capability-id 2
    python examples/download_spec.py --capability-id 2 --out spec_capability2.json
    python examples/download_spec.py --capability-id 4 --port 8001
    python examples/download_spec.py --capability-id 2 --domain hockey  # filtered view
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from benchmark.mcp_client import load_mcp_config, MCPConnectionConfig
from benchmark.utils import detect_container_runtime

DEFAULT_CONFIG = "benchmark/mcp_connection_config.yaml"

# Python one-liner run inside the container to fetch the spec
_FETCH_SCRIPT = (
    "import httpx, sys; "
    "r = httpx.get(sys.argv[1], timeout=10); "
    "r.raise_for_status(); "
    "print(r.text)"
)


def _fetch_spec_from_container(cfg: MCPConnectionConfig, port: int) -> dict:
    """Exec into the container and fetch the OpenAPI spec using Python + httpx."""
    runtime = cfg.container_runtime or detect_container_runtime()
    url = f"http://localhost:{port}/openapi.json"
    cmd = [
        runtime, "exec", cfg.container_name,
        "python", "-c", _FETCH_SCRIPT, url,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to fetch spec from container:\n"
            f"  stdout: {result.stdout[:200]}\n"
            f"  stderr: {result.stderr[:200]}"
        )
    return json.loads(result.stdout)


def _summarize_spec(spec: dict):
    """Print a human-readable summary of the spec."""
    info = spec.get("info", {})
    print(f"\nTitle:   {info.get('title', '(untitled)')}")
    print(f"Version: {info.get('version', '?')}")
    desc = info.get("description", "")
    if desc:
        print(f"Desc:    {desc.split(chr(10))[0]}")

    paths = spec.get("paths", {})
    print(f"\nEndpoints ({len(paths)}):")
    for path, path_item in sorted(paths.items()):
        methods = [m.upper() for m in path_item if m in ("get", "post", "put", "delete", "patch")]
        op = path_item.get(methods[0].lower(), {}) if methods else {}
        summary = op.get("summary", op.get("operationId", ""))
        print(f"  {','.join(methods):8s} {path}  {summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the OpenAPI spec from a capability's FastAPI backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch Capability 2 spec and print a summary
  python examples/download_spec.py --capability-id 2

  # Save to file
  python examples/download_spec.py --capability-id 2 --out spec_capability2.json

  # Capability 4 has two backends: M3 REST (8000) and Retriever (8001)
  python examples/download_spec.py --capability-id 4 --port 8000
  python examples/download_spec.py --capability-id 4 --port 8001

  # Print raw JSON (no summary)
  python examples/download_spec.py --capability-id 2 --json
        """,
    )
    parser.add_argument("--capability-id", type=int, required=True,
                        help="Capability ID (1, 2, 3, or 4)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port of the FastAPI server inside the container (default: 8000). "
                             "Capability 4 retriever uses port 8001.")
    parser.add_argument("--out", type=str, default=None,
                        help="Save spec JSON to this file (default: print to stdout)")
    parser.add_argument("--json", action="store_true",
                        help="Print the raw JSON spec instead of a summary")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"Path to MCP connection config YAML (default: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    configs = load_mcp_config(args.config)
    cfg = configs.get(args.capability_id)
    if cfg is None:
        print(f"Error: Capability {args.capability_id} not found in {args.config}. "
              f"Available: {sorted(configs.keys())}", file=sys.stderr)
        sys.exit(1)

    if cfg.command:
        # Local subprocess mode — no container to exec into.
        print(
            f"Capability {args.capability_id} is configured as a local subprocess (no container).\n"
            "Start the FastAPI server locally, then fetch the spec directly:\n"
            f"  curl http://localhost:{args.port}/openapi.json"
        )
        sys.exit(0)

    print(f"Fetching OpenAPI spec from Capability {args.capability_id} "
          f"container '{cfg.container_name}' on port {args.port} ...")

    try:
        spec = _fetch_spec_from_container(cfg, args.port)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"\nSpec saved to: {args.out}  ({len(spec.get('paths', {}))} endpoints)")
    elif args.json:
        print(json.dumps(spec, indent=2))
    else:
        _summarize_spec(spec)


if __name__ == "__main__":
    main()
