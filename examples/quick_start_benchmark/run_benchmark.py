#!/usr/bin/env python3
"""
Simple test runner for all benchmark tasks (capabilities 1–4).

Iterates over every domain for the selected capability, starts the MCP server
for that domain via docker exec (stdio), lists tools with parameters, runs the
agent (placeholder), then moves to the next domain.

Usage:
    python run_benchmark.py --capability 2                    # all domains, capability 2
    python run_benchmark.py --capability 2 --domain airline   # single domain
    python run_benchmark.py --capability 1 --runtime podman   # use podman
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / "server.yaml"


def get_runtime(preferred: str = "docker") -> str:
    if shutil.which(preferred):
        return preferred
    fallback = "podman" if preferred == "docker" else "docker"
    if shutil.which(fallback):
        print(f"  [{preferred} not found, using {fallback}]")
        return fallback
    print("Error: neither docker nor podman found on PATH.")
    sys.exit(1)


def print_tools(tools: list) -> None:
    for t in tools:
        print(f"  {t.name}")
        if t.description:
            print(f"    {t.description.strip().split(chr(10))[0]}")
        props = t.inputSchema.get("properties", {}) if t.inputSchema else {}
        required = set((t.inputSchema or {}).get("required", []))
        if props:
            print(f"    Parameters:")
            for param, schema in props.items():
                req_marker = " (required)" if param in required else ""
                ptype = schema.get("type", "any")
                desc = schema.get("description", "")
                print(f"      - {param}: {ptype}{req_marker}  {desc}")


async def run_domain(domain: str, cfg: dict, rt: str) -> None:
    container = cfg["container"]
    env = cfg.get("env", {})
    command = cfg["command"]

    docker_args = ["exec", "-i"]
    for k, v in env.items():
        docker_args += ["-e", f"{k}={v}"]
    docker_args += ["-e", f"MCP_DOMAIN={domain}"]
    docker_args += [container, *command]

    params = StdioServerParameters(command=rt, args=docker_args, env=None)

    print(f"\n{'='*60}")
    print(f"Domain: {domain}")
    print(f"Command: {rt} {' '.join(docker_args)}")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.list_tools()
            tools = response.tools
            print(f"\nTools ({len(tools)}):")
            print_tools(tools)

            # --- Agent call placeholder ---
            # result = await agent.run(domain=domain, tools=tools, session=session)
            print("\n[placeholder] Agent called with tools")

            # --- Wait for agent response placeholder ---
            # answer = await result
            print("[placeholder] Agent response received")

    print(f"Server stopped for domain: {domain}")


async def main(capability: int, domain_filter: str | None, runtime: str) -> None:
    config = yaml.safe_load(CONFIG_FILE.read_text())
    key = f"capability_{capability}"
    if key not in config:
        print(f"Error: capability {capability} not found in {CONFIG_FILE}.")
        sys.exit(1)

    cfg = config[key]
    rt = get_runtime(runtime)
    domains = cfg["domains"]

    if domain_filter:
        if domain_filter not in domains:
            print(f"Error: domain '{domain_filter}' not found in capability {capability}.")
            sys.exit(1)
        domains = [domain_filter]

    print(f"Capability {capability} runner | container: {cfg['container']} | domains: {len(domains)}")

    for domain in domains:
        await run_domain(domain, cfg, rt)

    print("\nAll domains complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MCP runner (all capabilities)")
    parser.add_argument("--capability", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Which capability to run (1, 2, 3, or 4)")
    parser.add_argument("--domain", default=None, help="Run a single domain only")
    parser.add_argument("--runtime", default="docker", choices=["docker", "podman"])
    args = parser.parse_args()
    asyncio.run(main(args.capability, args.domain, args.runtime))
