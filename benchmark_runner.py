#!/usr/bin/env python3
"""
Simple Benchmark Runner

Takes a task_id as input and iterates over data points in the corresponding directory.
For each file, extracts the domain from the filename, starts the MCP server with that domain,
connects to it, and lists the available tools.

Usage:
    
    export TASK_2_DIR=<path to downloaded task_2 directory from box https://ibm.ent.box.com/folder/364205927270>
    pip install mcp

    python benchmark_runner.py --task_id 2
    python benchmark_runner.py --task_id 2 --container-runtime podman
"""
import os
import argparse
import asyncio
import signal
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Task configurations - maps task_id to input directory path
TASK_PATHS = {
    2: os.environ.get("TASK_2_DIR", "/Users/anu/Documents/GitHub/routing/EnterpriseBenchmark/train/input/input"),
}

# Default settings
DEFAULT_CONTAINER_NAME = "fastapi-mcp-server"
DEFAULT_CONTAINER_RUNTIME = "podman"


def stop_mcp_server(container_runtime: str, container_name: str):
    """Stop any running mcp_server.py processes inside the container."""
    try:
        # Find and kill the mcp_server.py process inside the container
        kill_cmd = [
            container_runtime, "exec", container_name,
            "pkill", "-f", "python mcp_server.py"
        ]
        subprocess.run(kill_cmd, capture_output=True, timeout=5)
        print("  Server stopped.")
    except subprocess.TimeoutExpired:
        print("  Warning: Timeout while stopping server")
    except Exception as e:
        # Ignore errors - process may already be stopped
        pass


async def connect_and_list_tools(domain: str, container_runtime: str, container_name: str) -> list[str]:
    """Connect to MCP server with the given domain and list available tools."""

    exec_args = [
        "exec", "-i",
        "-e", f"MCP_DOMAINS={domain}",
        container_name,
        "python", "mcp_server.py"
    ]
    print(f"  Starting: {container_runtime} {' '.join(exec_args)}")

    server_params = StdioServerParameters(
        command=container_runtime,
        args=exec_args,
        env=None,
    )

    tool_names = []

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                response = await session.list_tools()
                tool_names = [tool.name for tool in response.tools]
    finally:
        # Explicitly stop the server process
        stop_mcp_server(container_runtime, container_name)

    return tool_names


async def process_domain(domain: str, container_runtime: str, container_name: str) -> dict:
    """Process a single domain: connect to MCP server and list tools."""

    print(f"\n{'='*60}")
    print(f"Domain: {domain}")
    print(f"{'='*60}")

    try:
        tools = await connect_and_list_tools(domain, container_runtime, container_name)
        print(f"  Tools loaded: {len(tools)}")

        # Print first few tools as sample
        for tool in tools[:5]:
            print(f"    - {tool}")
        if len(tools) > 5:
            print(f"    ... and {len(tools) - 5} more")

        return {
            "domain": domain,
            "status": "success",
            "tool_count": len(tools),
            "tools": tools,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "domain": domain,
            "status": "error",
            "error": str(e),
            "tool_count": 0,
            "tools": [],
        }


async def run_task(task_id: int, container_runtime: str, container_name: str) -> list[dict]:
    """Run benchmark for a given task_id."""

    if task_id not in TASK_PATHS:
        print(f"Error: Unknown task_id {task_id}")
        print(f"Available task_ids: {list(TASK_PATHS.keys())}")
        sys.exit(1)

    input_path = Path(TASK_PATHS[task_id])

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Get all JSON files and extract domain names
    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        print(f"Error: No JSON files found in {input_path}")
        sys.exit(1)

    domains = [f.stem for f in json_files]  # Strip .json extension

    print(f"Task ID: {task_id}")
    print(f"Input path: {input_path}")
    print(f"Container runtime: {container_runtime}")
    print(f"Container name: {container_name}")
    print(f"Found {len(domains)} domains: {', '.join(domains[:5])}{'...' if len(domains) > 5 else ''}")

    results = []
    for domain in domains:
        result = await process_domain(domain, container_runtime, container_name)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"  Total domains: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        total_tools = sum(r["tool_count"] for r in successful)
        print(f"  Total tools across all domains: {total_tools}")

    if failed:
        print(f"\n  Failed domains:")
        for r in failed:
            print(f"    - {r['domain']}: {r.get('error', 'Unknown error')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner for MCP Server")
    parser.add_argument("--task_id", type=int, required=True, help="Task ID to run")
    parser.add_argument(
        "--container-runtime",
        type=str,
        default=DEFAULT_CONTAINER_RUNTIME,
        help=f"Container runtime to use (default: {DEFAULT_CONTAINER_RUNTIME})"
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default=DEFAULT_CONTAINER_NAME,
        help=f"Container name (default: {DEFAULT_CONTAINER_NAME})"
    )

    args = parser.parse_args()

    print("="*60)
    print("Benchmark Runner")
    print("="*60)

    asyncio.run(run_task(args.task_id, args.container_runtime, args.container_name))


if __name__ == "__main__":
    main()
