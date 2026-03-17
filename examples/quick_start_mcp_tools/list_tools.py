#!/usr/bin/env python3
"""
List all MCP tools available for a given capability and domain.

Usage:
    python examples/list_tools.py --capability-id 2 --domain hockey
    python examples/list_tools.py --capability-id 1 --domain superhero
    python examples/list_tools.py --capability-id 3 --domain bpo
    python examples/list_tools.py --capability-id 2 --domain hockey --json
    python examples/list_tools.py --capability-id 2 --domain hockey --verbose
"""
import asyncio
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from benchmark.mcp_client import load_mcp_config, create_client_and_connect, MCPConnectionConfig

DEFAULT_CONFIG = "benchmark/mcp_connection_config.yaml"


async def _fetch_tools(capability_id: int, domain: str, config_path: str):
    configs = load_mcp_config(config_path)
    cfg = configs.get(capability_id)
    if cfg is None:
        raise ValueError(
            f"Capability {capability_id} not found in {config_path}. "
            f"Available: {sorted(configs.keys())}"
        )
    async with create_client_and_connect(cfg, domain) as session:
        result = await session.list_tools()
    return result.tools


def _print_tools(tools, verbose: bool):
    for i, tool in enumerate(tools, 1):
        print(f"  [{i:3d}] {tool.name}")
        if tool.description:
            lines = tool.description.strip().split("\n")
            if verbose:
                for line in lines:
                    print(f"         {line}")
            else:
                print(f"         {lines[0]}")
        if verbose and tool.inputSchema:
            schema = tool.inputSchema
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            if props:
                print("         Parameters:")
                for name, info in props.items():
                    req_marker = "*" if name in required else " "
                    ptype = info.get("type", info.get("$ref", "any"))
                    desc = info.get("description", "")
                    desc_short = desc.split("\n")[0][:60] if desc else ""
                    print(f"           {req_marker} {name}: {ptype}  {desc_short}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="List MCP tools available for a capability and domain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List tools for Capability 2 / hockey domain
  python examples/list_tools.py --capability-id 2 --domain hockey

  # Include parameter details
  python examples/list_tools.py --capability-id 2 --domain hockey --verbose

  # Machine-readable JSON output
  python examples/list_tools.py --capability-id 2 --domain hockey --json

  # BPO domain (Capability 3)
  python examples/list_tools.py --capability-id 3 --domain bpo
        """,
    )
    parser.add_argument("--capability-id", type=int, required=True,
                        help="Capability ID (1, 2, 3, or 4)")
    parser.add_argument("--domain", type=str, default="",
                        help="Domain name (e.g. hockey, bpo, superhero). "
                             "Leave empty to see all tools without domain filtering.")
    parser.add_argument("--json", action="store_true",
                        help="Print full tool definitions as JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full descriptions and parameter details")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"Path to MCP connection config YAML (default: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    domain_label = args.domain or "(no domain filter)"
    print(f"Connecting to Capability {args.capability_id} MCP server  [domain: {domain_label}] ...")

    try:
        tools = asyncio.run(_fetch_tools(args.capability_id, args.domain, args.config))
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        tools_data = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }
            for t in tools
        ]
        print(json.dumps(tools_data, indent=2))
        return

    print(f"\nFound {len(tools)} tool(s):\n")
    _print_tools(tools, verbose=args.verbose)


if __name__ == "__main__":
    main()
