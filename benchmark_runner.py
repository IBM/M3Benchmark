#!/usr/bin/env python3
"""
Simple Benchmark Runner

Iterates over all domain files in the TASK directory.
For each file (e.g., address.json, hockey.json):
  - Extracts domain from filename
  - Starts MCP server with that domain
  - Runs agent on benchmark queries from that file

Usage:

    export TASK_2_DIR=<path to downloaded task_2 directory>
    pip install mcp langchain-anthropic langgraph langchain-ollama

    # List tools only (no agent) - iterates all domains
    python benchmark_runner.py --task_id 2

    # Run benchmark with agent on all domains
    python benchmark_runner.py --task_id 2 --run-agent

    # Run benchmark on specific domain(s) only
    python benchmark_runner.py --task_id 2 --run-agent --domain hockey
    python benchmark_runner.py --task_id 2 --run-agent --domain hockey --domain address

    # Limit samples per domain (e.g., 5 samples from each domain file)
    python benchmark_runner.py --task_id 2 --run-agent --max-samples-per-domain 5

    # Use different provider/model
    python benchmark_runner.py --task_id 2 --run-agent --provider anthropic
    python benchmark_runner.py --task_id 2 --run-agent --provider ollama --model llama3.1:8b

Output:
    Results saved to: benchmark_output_YYYY-MM-DD_HH-MM-SS_<model_name>/
        - address_benchmark_output.json
        - hockey_benchmark_output.json
        - ...
        - summary.json
"""
import json
import os
import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_interface import AgentInterface, AgentResponse, create_agent

# Task configurations - maps task_id to input directory path
TASK_PATHS = {
    2: os.environ.get("TASK_2_DIR", "/Users/anu/Documents/GitHub/routing/EnterpriseBenchmark/train/input/input"),
}

@dataclass
class BenchmarkItem:
    """A single benchmark test case."""
    uuid: str
    domain: str
    query: str
    num_turns: int
    tools: List[Dict[str, Any]]
    additional_instructions: str = ""
    turn_id: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkItem":
        """Create BenchmarkItem from JSON dict."""
        dialogue = data.get("dialogue", {})
        turns = dialogue.get("turns", [])
        # Get first turn's query (for now, single turn support)
        query = turns[0]["query"] if turns else ""
        turn_id = turns[0].get("turn_id", 0) if turns else 0

        return cls(
            uuid=data.get("uuid", ""),
            domain=data.get("domain", ""),
            query=query,
            num_turns=data.get("num_turns", 1),
            tools=data.get("tools", []),
            additional_instructions=data.get("additional_instructions", ""),
            turn_id=turn_id,
        )


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark item."""
    uuid: str
    domain: str
    query: str
    answer: str = ""
    tool_calls: List[Dict] = field(default_factory=list)
    status: str = "pending"
    error: str = ""
    duration_s: float = 0.0


def load_benchmark_file(filepath: Path) -> List[BenchmarkItem]:
    """Load benchmark items from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    items = []
    for item_data in data:
        items.append(BenchmarkItem.from_dict(item_data))
    return items


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save benchmark results to JSON file."""
    output_data = []
    for r in results:
        output_data.append({
            "uuid": r.uuid,
            "domain": r.domain,
            "query": r.query,
            "answer": r.answer,
            "tool_calls": r.tool_calls,
            "status": r.status,
            "error": r.error,
            "duration_s": r.duration_s,
        })

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Results saved to: {output_path}")


def save_results_by_domain(results: List[BenchmarkResult], output_dir: Path):
    """Save benchmark results to domain-specific JSON files."""
    # Group results by domain
    by_domain: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.domain not in by_domain:
            by_domain[r.domain] = []
        by_domain[r.domain].append(r)

    # Save each domain to its own file
    for domain, domain_results in by_domain.items():
        output_file = output_dir / f"{domain}_benchmark_output.json"
        save_results(domain_results, output_file)

    print(f"\nAll results saved to: {output_dir}")


# Timeout for agent execution (seconds)
AGENT_TIMEOUT_SECONDS = 120




# Default settings
DEFAULT_CONTAINER_NAME = "fastapi-mcp-server"
DEFAULT_CONTAINER_RUNTIME = "podman"


def stop_mcp_server(container_runtime: str, container_name: str):
    """Stop any running mcp_server.py processes inside the container."""
    try:
        kill_cmd = [
            container_runtime, "exec", container_name,
            "pkill", "-f", "python mcp_server.py"
        ]
        subprocess.run(kill_cmd, capture_output=True, timeout=5)
        print("  Server stopped.")
    except subprocess.TimeoutExpired:
        print("  Warning: Timeout while stopping server")
    except Exception:
        pass


class MCPToolWrapper:
    """Converts MCP tools to LangChain StructuredTool objects."""

    def __init__(self, session: ClientSession):
        self.session = session
        self._tools_cache: Optional[List[StructuredTool]] = None

    async def get_tools(self) -> List[StructuredTool]:
        """Fetch tools from MCP server and convert to LangChain tools."""
        if self._tools_cache is not None:
            return self._tools_cache

        response = await self.session.list_tools()
        self._tools_cache = [self._create_tool(t) for t in response.tools]
        return self._tools_cache

    def _create_tool(self, mcp_tool) -> StructuredTool:
        """Convert a single MCP tool to LangChain StructuredTool."""
        session = self.session

        async def tool_func(**kwargs) -> str:
            result = await session.call_tool(mcp_tool.name, kwargs)
            if result.content:
                return result.content[0].text
            return "No result"

        return StructuredTool.from_function(
            name=mcp_tool.name,
            description=mcp_tool.description or f"Tool: {mcp_tool.name}",
            coroutine=tool_func,
        )


async def connect_and_get_session(
    domain: str,
    container_runtime: str,
    container_name: str,
):
    """Connect to MCP server and return the session context."""
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

    return stdio_client(server_params)


async def connect_and_list_tools(domain: str, container_runtime: str, container_name: str) -> List[str]:
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
        # Context exited cleanly - server should close on its own
        print("  Server stopped.")
    except ExceptionGroup as eg:
        # Handle Python 3.11+ ExceptionGroup from TaskGroup
        print(f"  Warning: Cleanup error (ignored): {eg}")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            print(f"  Warning: Cleanup error (ignored): {e}")
        else:
            # Force kill on unexpected errors
            stop_mcp_server(container_runtime, container_name)
            raise

    return tool_names


async def run_agent_with_query(
    domain: str,
    query: str,
    container_runtime: str,
    container_name: str,
    agent: AgentInterface,
) -> AgentResponse:
    """Run an agent with tools from the MCP server."""
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

    response = None

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Get tools as LangChain tools
                wrapper = MCPToolWrapper(session)
                tools = await wrapper.get_tools()
                print(f"  Loaded {len(tools)} tools")

                # Run the agent with timeout
                try:
                    response = await asyncio.wait_for(
                        agent.run(query, tools),
                        timeout=AGENT_TIMEOUT_SECONDS
                    )
                    print(f"  Agent completed. Response received: {response is not None}")
                except asyncio.TimeoutError:
                    print(f"  Agent timed out after {AGENT_TIMEOUT_SECONDS}s")
                    raise TimeoutError(f"Agent timed out after {AGENT_TIMEOUT_SECONDS} seconds")
        # Context exited cleanly
        print("  Server stopped.")
    except ExceptionGroup as eg:
        # Handle Python 3.11+ ExceptionGroup from TaskGroup cleanup
        # Response may still be valid if agent completed before cleanup error
        print(f"  Warning: Cleanup error (ignored): {eg}")
        print("  Server stopped.")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            print(f"  Warning: Cleanup error (ignored): {e}")
            print("  Server stopped.")
        else:
            stop_mcp_server(container_runtime, container_name)
            raise

    if response is None:
        raise RuntimeError("Agent did not return a response")
    return response


async def process_domain(
    domain: str,
    container_runtime: str,
    container_name: str,
    agent: Optional[AgentInterface] = None,
    query: Optional[str] = None,
) -> dict:
    """Process a single domain: connect to MCP server and list tools (optionally run agent)."""

    print(f"\n{'='*60}")
    print(f"Domain: {domain}")
    print(f"{'='*60}")

    try:
        if agent and query:
            # Run agent with query
            response = await run_agent_with_query(
                domain, query, container_runtime, container_name, agent
            )
            print(f"  Tool calls: {len(response.tool_calls)}")
            print(f"  Answer: {response.content[:200]}..." if len(response.content) > 200 else f"  Answer: {response.content}")

            return {
                "domain": domain,
                "status": "success",
                "query": query,
                "answer": response.content,
                "tool_calls": response.tool_calls,
            }
        else:
            # Just list tools
            tool_names = await connect_and_list_tools(domain, container_runtime, container_name)
            print(f"  Tools loaded: {len(tool_names)}")

            for tool in tool_names[:5]:
                print(f"    - {tool}")
            if len(tool_names) > 5:
                print(f"    ... and {len(tool_names) - 5} more")

            return {
                "domain": domain,
                "status": "success",
                "tool_count": len(tool_names),
                "tools": tool_names,
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


async def run_benchmark_item(
    item: BenchmarkItem,
    container_runtime: str,
    container_name: str,
    agent: AgentInterface,
) -> BenchmarkResult:
    """Run a single benchmark item."""
    import time

    result = BenchmarkResult(
        uuid=item.uuid,
        domain=item.domain,
        query=item.query,
    )

    start_time = time.perf_counter()

    try:
        response = await run_agent_with_query(
            domain=item.domain,
            query=item.query,
            container_runtime=container_runtime,
            container_name=container_name,
            agent=agent,
        )
        result.answer = response.content
        result.tool_calls = response.tool_calls
        result.status = "success"
    except Exception as e:
        result.status = "error"
        result.error = str(e)

    result.duration_s = time.perf_counter() - start_time
    return result


async def run_benchmark_for_domain(
    domain: str,
    items: List[BenchmarkItem],
    container_runtime: str,
    container_name: str,
    agent: AgentInterface,
    max_samples: Optional[int] = None,
) -> List[BenchmarkResult]:
    """Run benchmark for a single domain - starts MCP server once for all items."""
    import time

    # Limit samples if requested
    if max_samples and max_samples < len(items):
        items = items[:max_samples]

    print(f"\n{'#'*60}")
    print(f"# DOMAIN: {domain} ({len(items)} items)")
    print(f"{'#'*60}")

    results: List[BenchmarkResult] = []

    # Start MCP server ONCE for this domain
    exec_args = [
        "exec", "-i",
        "-e", f"MCP_DOMAINS={domain}",
        container_name,
        "python", "mcp_server.py"
    ]
    print(f"  Starting MCP server: {container_runtime} {' '.join(exec_args)}")

    server_params = StdioServerParameters(
        command=container_runtime,
        args=exec_args,
        env=None,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Get tools ONCE for this domain
                wrapper = MCPToolWrapper(session)
                tools = await wrapper.get_tools()
                print(f"  Loaded {len(tools)} tools for domain '{domain}'")

                # Run all queries for this domain
                for i, item in enumerate(items):
                    print(f"\n  [{i+1}/{len(items)}] Query: {item.query[:80]}{'...' if len(item.query) > 80 else ''}")

                    result = BenchmarkResult(
                        uuid=item.uuid,
                        domain=domain,
                        query=item.query,
                    )

                    start_time = time.perf_counter()

                    try:
                        # Run agent with timeout
                        response = await asyncio.wait_for(
                            agent.run(item.query, tools),
                            timeout=AGENT_TIMEOUT_SECONDS
                        )
                        result.answer = response.content
                        result.tool_calls = response.tool_calls
                        result.status = "success"
                        elapsed = time.perf_counter() - start_time
                        print(f"    Status: success | Tools: {len(result.tool_calls)} | Time: {elapsed:.2f}s")
                        # Log the answer
                        answer_preview = result.answer[:200] if result.answer else "(empty)"
                        print(f"    Answer: {answer_preview}{'...' if len(result.answer) > 200 else ''}")
                        # Log tool calls if any
                        if result.tool_calls:
                            print(f"    Tool calls:")
                            for tc in result.tool_calls:
                                print(f"      - {tc.get('tool_name', 'unknown')}: {tc.get('arguments', {})}")
                                tc_result = str(tc.get('result', ''))[:100]
                                print(f"        Result: {tc_result}{'...' if len(str(tc.get('result', ''))) > 100 else ''}")
                    except asyncio.TimeoutError:
                        result.status = "error"
                        result.error = f"Agent timed out after {AGENT_TIMEOUT_SECONDS} seconds"
                        print(f"    Status: timeout after {AGENT_TIMEOUT_SECONDS}s")
                    except Exception as e:
                        result.status = "error"
                        result.error = str(e)
                        print(f"    Status: error | {str(e)[:50]}")

                    result.duration_s = time.perf_counter() - start_time
                    results.append(result)

        print(f"\n  Server stopped for domain '{domain}'")
    except ExceptionGroup as eg:
        print(f"  Warning: Cleanup error (ignored): {eg}")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            print(f"  Warning: Cleanup error (ignored): {e}")
        else:
            stop_mcp_server(container_runtime, container_name)
            raise

    return results


async def run_task(
    task_id: int,
    container_runtime: str,
    container_name: str,
    run_agent: bool = False,
    provider: str = "ollama",
    model: Optional[str] = None,
    max_samples_per_domain: Optional[int] = None,
    output_file: Optional[str] = None,
    domains: Optional[List[str]] = None,
) -> List[BenchmarkResult]:
    """Run benchmark for a given task_id, iterating over all domain files."""

    if task_id not in TASK_PATHS:
        print(f"Error: Unknown task_id {task_id}")
        print(f"Available task_ids: {list(TASK_PATHS.keys())}")
        sys.exit(1)

    input_path = Path(TASK_PATHS[task_id])

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Get all JSON files - each file represents a domain
    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        print(f"Error: No JSON files found in {input_path}")
        sys.exit(1)

    # Filter to specific domains if provided
    if domains:
        filtered_files = []
        for json_file in json_files:
            if json_file.stem in domains:
                filtered_files.append(json_file)
        if not filtered_files:
            available = [f.stem for f in json_files]
            print(f"Error: None of the specified domains found: {domains}")
            print(f"Available domains: {available[:10]}{'...' if len(available) > 10 else ''}")
            sys.exit(1)
        json_files = filtered_files
        print(f"Filtered to {len(json_files)} domains: {[f.stem for f in json_files]}")

    print(f"Task ID: {task_id}")
    print(f"Input path: {input_path}")
    print(f"Container runtime: {container_runtime}")
    print(f"Container name: {container_name}")
    print(f"Processing {len(json_files)} domain files")

    if not run_agent:
        # Just list tools for each domain (original behavior)
        results = []
        for json_file in json_files:
            domain = json_file.stem  # Domain from filename
            result = await process_domain(domain, container_runtime, container_name)
            results.append(result)
        return results

    # Create agent once for all domains
    agent = create_agent(provider=provider, model=model)
    print(f"Agent: {provider} / {model or 'default'}")
    if max_samples_per_domain:
        print(f"Max samples per domain: {max_samples_per_domain}")

    # Process each domain file
    all_results: List[BenchmarkResult] = []
    print("jsn files", json_files)
    for json_file in json_files:
        domain = json_file.stem  # Extract domain from filename (e.g., "address" from "address.json")

        # Load benchmark items from this file
        items = load_benchmark_file(json_file)
        print(f"\nLoaded {len(items)} items from {json_file.name}")

        # Run benchmark for this domain
        domain_results = await run_benchmark_for_domain(
            domain=domain,
            items=items,
            container_runtime=container_runtime,
            container_name=container_name,
            agent=agent,
            max_samples=max_samples_per_domain,
        )
        all_results.extend(domain_results)

    results = all_results

    # Summary
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "error"]

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "error"]

    print(f"  Total items: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        total_time = sum(r.duration_s for r in successful)
        avg_time = total_time / len(successful)
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per item: {avg_time:.2f}s")

    if failed:
        print(f"\n  Failed items:")
        for r in failed[:10]:  # Show first 10 failures
            print(f"    - [{r.domain}] {r.query[:50]}...: {r.error[:50]}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    # Save results to timestamped directory with domain-specific files
    if output_file:
        # Use specified output file (single file mode)
        save_results(results, Path(output_file))
    else:
        # Create human-readable timestamped directory with model name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Sanitize model name for directory (replace special chars)
        model_name = (model or "default").replace(":", "_").replace("/", "_")
        output_dir = Path(f"benchmark_output_{timestamp}_{model_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save domain-specific output files (e.g., address_benchmark_output.json)
        save_results_by_domain(results, output_dir)

        # Also save a combined summary file
        summary_file = output_dir / "summary.json"
        summary = {
            "task_id": task_id,
            "timestamp": timestamp,
            "provider": provider,
            "model": model or "default",
            "total_items": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_time_s": sum(r.duration_s for r in results),
            "domains": list(set(r.domain for r in results)),
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to: {summary_file}")

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
    parser.add_argument(
        "--domain",
        type=str,
        action="append",
        default=None,
        help="Domain(s) to process (can specify multiple times, default: all domains)"
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Run the agent on benchmark queries (default: just list tools)"
    )
    parser.add_argument(
        "--max-samples-per-domain",
        type=int,
        default=None,
        help="Maximum number of benchmark items per domain (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: timestamped directory with per-domain files)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["anthropic", "openai", "ollama"],
        help="LLM provider to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: provider-specific default)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Benchmark Runner")
    print("="*60)

    asyncio.run(run_task(
        task_id=args.task_id,
        container_runtime=args.container_runtime,
        container_name=args.container_name,
        run_agent=args.run_agent,
        provider=args.provider,
        model=args.model,
        max_samples_per_domain=args.max_samples_per_domain,
        output_file=args.output,
        domains=args.domain,
    ))


if __name__ == "__main__":
    main()
