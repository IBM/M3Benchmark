#!/usr/bin/env python3
"""
Simple example: connect to a capability container via docker exec and list / call tools.

Requires:
    pip install mcp
    docker compose up -d          # start the containers

Usage:
    python examples/m3_quickstart/simple_docker.py
    python examples/m3_quickstart/simple_docker.py --capability-id 2 --domain hockey
    python examples/m3_quickstart/simple_docker.py --capability-id 3 --domain bpo
    python examples/m3_quickstart/simple_docker.py --runtime podman
"""

import argparse
import asyncio
import shutil
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Container config for each capability ─────────────────────────────────────
CAPABILITIES = {
    1: {"container": "capability_1_bi_apis",
        "env": {"MCP_DB_ROOT": "/app/db", "CAPABILITY_ID": "1"}},
    2: {"container": "capability_2_dashboard_apis",
        "env": {"MCP_DB_ROOT": "/app/db", "CAPABILITY_ID": "2"}},
    3: {"container": "capability_3_multihop_reasoning",
        "env": {"MCP_DB_ROOT": "/app/db", "CAPABILITY_ID": "3"}},
    4: {"container": "capability_4_multiturn",
        "env": {"MCP_DB_ROOT": "/app/db", "CAPABILITY_ID": "4",
                "PRELOAD_COLLECTIONS": "false"}},
}
CONTAINER_COMMAND = ["python", "/app/mcp_dispatch.py"]


def get_runtime(preferred: str = "docker") -> str:
    if shutil.which(preferred):
        return preferred
    fallback = "podman" if preferred == "docker" else "docker"
    if shutil.which(fallback):
        print(f"[{preferred} not found — using {fallback}]")
        return fallback
    sys.exit("Error: neither docker nor podman found on PATH.")


async def main(capability_id: int, domain: str, runtime: str) -> None:
    cap = CAPABILITIES[capability_id]
    container = cap["container"]

    # Build: docker exec -i -e KEY=VAL ... <container> python /app/mcp_dispatch.py
    env_args = []
    for k, v in cap["env"].items():
        env_args += ["-e", f"{k}={v}"]
    if domain:
        env_args += ["-e", f"MCP_DOMAIN={domain}"]

    args = ["exec", "-i"] + env_args + [container] + CONTAINER_COMMAND
    params = StdioServerParameters(command=runtime, args=args, env=None)

    print(f"Connecting: {runtime} {' '.join(args)}\n")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── List tools ────────────────────────────────────────────────────
            resp  = await session.list_tools()
            tools = resp.tools
            print(f"Tools ({len(tools)}):")
            for t in tools:
                desc = (t.description or "").split("\n")[0]
                print(f"  {t.name:40s}  {desc[:60]}")

            print()

            # ── Call a tool ───────────────────────────────────────────────────
            # Replace this block with your own tool call.
            # Use `session.call_tool(tool_name, {arg: value, ...})` to invoke any tool.
            #
            # Example (uncomment and adapt):
            #
            # if tools:
            #     first_tool = tools[0].name
            #     print(f"Calling: {first_tool}")
            #     result = await session.call_tool(first_tool, {})
            #     for item in result.content:
            #         print(getattr(item, "text", repr(item)))

            print("Done. Wire in your agent or tool calls above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple docker-exec MCP example.")
    parser.add_argument("--capability-id", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--domain", type=str, default="hockey",
                        help="Domain name sent as MCP_DOMAIN (default: hockey)")
    parser.add_argument("--runtime", type=str, default="docker", choices=["docker", "podman"])
    args = parser.parse_args()

    rt = get_runtime(args.runtime)
    asyncio.run(main(args.capability_id, args.domain, rt))
