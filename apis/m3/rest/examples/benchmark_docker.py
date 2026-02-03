"""
Benchmark: Loop through questions, spin up the correct MCP server inside Docker
per domain, and print how many tools are loaded.

Usage:
    # Start the container first
    docker-compose up -d

    # Run benchmark
    python examples/benchmark_docker.py
"""
import asyncio
import json
import os
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


QUESTIONS_FILE = Path(__file__).parent / "benchmark_questions.json"


def load_questions():
    with open(QUESTIONS_FILE) as f:
        return json.load(f)


async def load_tools_for_domain(domain: str, container_name: str, container_runtime: str):
    """Spawn an MCP server inside Docker for the given domain and return tool count + timing."""
    server_params = StdioServerParameters(
        command=container_runtime,
        args=[
            "exec", "-i",
            "-e", f"MCP_DOMAINS={domain}",
            container_name,
            "python", "mcp_server.py",
        ],
        env=None,
    )

    t0 = time.perf_counter()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            init_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            response = await session.list_tools()
            discovery_time = time.perf_counter() - t1

            tool_count = len(response.tools)
            total_time = time.perf_counter() - t0

    return tool_count, init_time, discovery_time, total_time


async def main():
    container_name = os.getenv("MCP_CONTAINER_NAME", "fastapi-mcp-server")
    container_runtime = os.getenv("CONTAINER_RUNTIME", "docker")
    questions = load_questions()

    print("=" * 90)
    print("BENCHMARK: Docker MCP Server")
    print(f"Container:   {container_name} (via {container_runtime})")
    print(f"Questions:   {len(questions)}")
    print("=" * 90)

    results = []

    for i, item in enumerate(questions):
        domain = item["domain"]
        question = item["question"]

        print(f"\n[{i+1}/{len(questions)}] Domain: {domain}")
        print(f"  Q: {question}")

        try:
            tool_count, init_time, discovery_time, total_time = await load_tools_for_domain(
                domain, container_name, container_runtime
            )
            print(f"  Tools loaded: {tool_count}")
            print(f"  MCP init: {init_time:.3f}s | Discovery: {discovery_time:.3f}s | Total: {total_time:.3f}s")
            # TODO: Load Retrievers as well for a given domain
            results.append({
                "index": i + 1,
                "domain": domain,
                "question": question,
                "tool_count": tool_count,
                "init_s": init_time,
                "discovery_s": discovery_time,
                "total_s": total_time,
                "error": None,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "index": i + 1,
                "domain": domain,
                "question": question,
                "tool_count": 0,
                "init_s": 0,
                "discovery_s": 0,
                "total_s": 0,
                "error": str(e),
            })

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\n  {'#':<4} {'Domain':<25} {'Tools':>6} {'Init':>8} {'Disc':>8} {'Total':>8}  Status")
    print(f"  {'─'*4} {'─'*25} {'─'*6} {'─'*8} {'─'*8} {'─'*8}  {'─'*6}")

    for r in results:
        status = "OK" if not r["error"] else "ERR"
        print(
            f"  {r['index']:<4} {r['domain']:<25} {r['tool_count']:>6} "
            f"{r['init_s']:>7.3f}s {r['discovery_s']:>7.3f}s {r['total_s']:>7.3f}s  {status}"
        )

    successful = [r for r in results if not r["error"]]
    if successful:
        total_tools = sum(r["tool_count"] for r in successful)
        avg_total = sum(r["total_s"] for r in successful) / len(successful)
        total_time = sum(r["total_s"] for r in successful)
        unique_domains = len(set(r["domain"] for r in successful))

        print(f"\n  Unique domains queried: {unique_domains}")
        print(f"  Total tools loaded (across all questions): {total_tools}")
        print(f"  Avg time per question: {avg_total:.3f}s")
        print(f"  Total benchmark time: {total_time:.3f}s")
        print(f"  Errors: {len(results) - len(successful)}/{len(results)}")

    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
