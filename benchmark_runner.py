#!/usr/bin/env python3
"""
Benchmark Runner
================

Runs LLM agents against MCP tool servers and records trajectories + answers.

Each task_id maps to a TaskConfig:
  - input_dir:       where the domain JSON files live
  - container_name:  which Docker container to exec into
  - mcp_domain_env:  env var the MCP server reads for domain filtering

Tasks:
  Task 2  -> fastapi-mcp-server   (M3 SQL tools)
  Task 5  -> retriever-mcp-server (ChromaDB retriever)

Setup:
  pip install mcp langchain-anthropic langgraph langchain-ollama

Usage:
  # Single task
  python benchmark_runner.py --task_id 2 --run-agent --domain hockey
  python benchmark_runner.py --task_id 5 --run-agent --domain address

  # Multiple tasks (sequential, default)
  python benchmark_runner.py --task_id 2 5 --run-agent --domain address

  # Multiple tasks (parallel via asyncio.gather)
  python benchmark_runner.py --task_id 2 5 --run-agent --domain address --parallel

  # Limit samples, choose provider/model
  python benchmark_runner.py --task_id 5 --run-agent --domain address --max-samples-per-domain 5
  python benchmark_runner.py --task_id 5 --run-agent --provider anthropic --model claude-sonnet-4-5-20250929

  # List tools only (no agent run)
  python benchmark_runner.py --task_id 5 --list-tools --domain address

  # Tool shortlisting (top-k most relevant tools per query)
  python benchmark_runner.py --task_id 2 --run-agent --domain hockey --top-k-tools 5

Options:
  --task_id ID [ID ...]       Task ID(s) to run (e.g. 2, 5, or 2 5)
  --parallel                  Run multiple task_ids concurrently (default: sequential)
  --run-agent                 Run the agent on benchmark queries
  --list-tools                List available tools and exit
  --domain DOMAIN             Filter to specific domain(s), repeatable
  --max-samples-per-domain N  Cap number of queries per domain
  --provider PROVIDER         LLM provider: ollama, anthropic, openai, litellm, watsonx, rits
  --model MODEL               Model name (default: provider-specific)
  --top-k-tools K             Keep top-K tools per query via embedding similarity
  --container-runtime RT      Container runtime: docker or podman (default: auto-detect)
  --container-name NAME       Override container name from task config

Output:
  Results saved to: <task_input_dir>/../output/<domain>.json
"""
import json
import os
import argparse
from dotenv import load_dotenv
load_dotenv()
import asyncio
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.agent_interface import AgentInterface, AgentResponse, LangGraphReActAgent, create_agent
from agents.llm import create_llm
from agents.mcp_tool_wrapper import MCPToolWrapper


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    """Per-task settings: where the data is, which container to use, and
    what environment variable name the MCP server reads for domain filtering."""
    input_dir: str
    container_name: str
    mcp_domain_env: str = "MCP_DOMAIN"


TASK_CONFIGS: Dict[int, TaskConfig] = {
    2: TaskConfig(
        input_dir=os.environ.get(
            "TASK_2_DIR",
            "/Users/anu/Documents/GitHub/routing/EnterpriseBenchmark/train/input/",
        ),
        container_name="fastapi-mcp-server",
        # NOTE: existing task_2 passes MCP_DOMAIN (plural) which the m3
        # server doesn't actually read — it reads MCP_DOMAIN.  Keeping the
        # legacy value here to avoid changing current behaviour.
        mcp_domain_env="MCP_DOMAIN",
    ),
    5: TaskConfig(
        input_dir=os.environ.get(
            "TASK_5_DIR",
            "/Users/anu/Desktop/data/task_5/train/input",
        ),
        container_name="retriever-mcp-server",
        mcp_domain_env="MCP_DOMAIN",
    ),
}

# Back-compat helper used by run_task / list_tools_for_domains
TASK_PATHS = {tid: cfg.input_dir for tid, cfg in TASK_CONFIGS.items()}

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
    trajectory: List[Dict] = field(default_factory=list)  # Agent trajectory
    turn_id: int = 0
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


def _extract_tool_response_values(result_str: str):
    """Extract only the values from a tool response JSON string.

    Tool responses come as JSON dicts like '{"description": "Foo"}' or
    '{"codes": []}'. This extracts just the values ("Foo" or []) so the
    output contains the data without the key names.
    """
    try:
        parsed = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return result_str

    if isinstance(parsed, dict):
        values = list(parsed.values())
        if len(values) == 1:
            return values[0]
        return values

    # Already a plain value (list, int, string, etc.)
    return parsed


def save_results_ground_truth(results: List[BenchmarkResult], output_dir: Path):
    """Save benchmark results in ground truth format to per-domain files.

    Writes one file per domain to output_dir/<domain>.json matching the
    structure of the example output files (uuid, domain, ground_truth with
    turn_id, query, answer, and gold_sequence).
    """
    # Group results by domain
    by_domain: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.domain not in by_domain:
            by_domain[r.domain] = []
        by_domain[r.domain].append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    for domain, domain_results in by_domain.items():
        records = []
        for r in domain_results:
            # Build gold_sequence from tool_calls.
            # Each tool call becomes its own entry so retries are preserved.
            gold_sequence = []
            # Internal LangChain parameters that leak into tool schemas
            _INTERNAL_KEYS = {"args", "config", "kwargs"}
            for tc in r.tool_calls:
                raw_args = tc.get("arguments", {})
                filtered_args = {
                    k: v for k, v in raw_args.items() if k not in _INTERNAL_KEYS
                }
                gold_sequence.append({
                    "tool_call": [[{
                        "name": tc.get("tool_name", ""),
                        "arguments": filtered_args,
                    }]],
                    "tool_response": [_extract_tool_response_values(tc.get("result", ""))],
                })

            record = {
                "uuid": r.uuid,
                "domain": r.domain,
                "status": r.status,
                "error": r.error,
                "duration_s": r.duration_s,
                "ground_truth": [
                    {
                        "turn_id": r.turn_id,
                        "query": r.query,
                        "answer": r.answer,
                        "gold_sequence": gold_sequence,
                    }
                ],
            }
            records.append(record)

        output_file = output_dir / f"{domain}.json"
        with open(output_file, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Ground truth results saved to: {output_file}")


# Timeout for agent execution (seconds)
AGENT_TIMEOUT_SECONDS = 120


# Default settings
DEFAULT_CONTAINER_NAME = "fastapi-mcp-server"
DEFAULT_CONTAINER_RUNTIME = "podman"


def _build_exec_args(
    domain: str,
    container_name: str,
    mcp_domain_env: str = "MCP_DOMAIN",
) -> list:
    """Build the docker/podman exec args for starting an MCP server."""
    return [
        "exec", "-i",
        "-e", f"{mcp_domain_env}={domain}",
        container_name,
        "python", "mcp_server.py",
    ]


def detect_container_runtime() -> str:
    """
    Detect available container runtime (podman or docker).
    Returns 'podman' if available, otherwise 'docker'.
    """
    import shutil
    
    # Check if podman is available
    if shutil.which("podman"):
        return "podman"
    
    # Fall back to docker
    if shutil.which("docker"):
        print("  Note: podman not found, using docker instead")
        return "docker"
    
    # Neither found
    raise RuntimeError("Neither podman nor docker found in PATH. Please install one of them.")




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


async def connect_and_get_session(
    domain: str,
    container_runtime: str,
    container_name: str,
    mcp_domain_env: str = "MCP_DOMAIN",
):
    """Connect to MCP server and return the session context."""
    exec_args = _build_exec_args(domain, container_name, mcp_domain_env)
    print(f"  Starting: {container_runtime} {' '.join(exec_args)}")

    server_params = StdioServerParameters(
        command=container_runtime,
        args=exec_args,
        env=None,
    )

    return stdio_client(server_params)


async def connect_and_list_tools(
    domain: str, container_runtime: str, container_name: str,
    mcp_domain_env: str = "MCP_DOMAIN",
) -> List[str]:
    """Connect to MCP server with the given domain and list available tools."""
    exec_args = _build_exec_args(domain, container_name, mcp_domain_env)
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


async def connect_and_get_tools_detailed(
    domain: str, container_runtime: str, container_name: str,
    mcp_domain_env: str = "MCP_DOMAIN",
) -> List[Dict[str, Any]]:
    """Connect to MCP server and get detailed tool information including parameters."""
    exec_args = _build_exec_args(domain, container_name, mcp_domain_env)
    print(f"  Starting: {container_runtime} {' '.join(exec_args)}")

    server_params = StdioServerParameters(
        command=container_runtime,
        args=exec_args,
        env=None,
    )

    tools_detailed = []

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                
                for tool in response.tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                    }
                    tools_detailed.append(tool_info)
        
        print("  Server stopped.")
    except ExceptionGroup as eg:
        print(f"  Warning: Cleanup error (ignored): {eg}")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            print(f"  Warning: Cleanup error (ignored): {e}")
        else:
            stop_mcp_server(container_runtime, container_name)
            raise

    return tools_detailed


async def run_agent_with_query(
    domain: str,
    query: str,
    container_runtime: str,
    container_name: str,
    agent: AgentInterface,
    mcp_domain_env: str = "MCP_DOMAIN",
) -> AgentResponse:
    """Run an agent with tools from the MCP server."""
    exec_args = _build_exec_args(domain, container_name, mcp_domain_env)
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
    mcp_domain_env: str = "MCP_DOMAIN",
) -> dict:
    """Process a single domain: connect to MCP server and list tools (optionally run agent)."""

    print(f"\n{'='*60}")
    print(f"Domain: {domain}")
    print(f"{'='*60}")

    try:
        if agent and query:
            # Run agent with query
            response = await run_agent_with_query(
                domain, query, container_runtime, container_name, agent,
                mcp_domain_env=mcp_domain_env,
            )
            print(f"  Tool calls: {len(response.tool_calls)}")
            print(f"  Answer: {response.content[:200]}..." if len(response.content) > 200 else f"  Answer: {response.content}")

            return {
                "domain": domain,
                "status": "success",
                "query": query,
                "answer": response.content,
                "tool_calls": response.tool_calls,
                "trajectory": response.trajectory,
            }
        else:
            # Just list tools
            tool_names = await connect_and_list_tools(domain, container_runtime, container_name, mcp_domain_env=mcp_domain_env)
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
        result.trajectory = response.trajectory
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
    shortlister=None,
    mcp_domain_env: str = "MCP_DOMAIN",
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
    exec_args = _build_exec_args(domain, container_name, mcp_domain_env)
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

                # Pre-compute tool embeddings for shortlisting
                if shortlister:
                    shortlister.encode_tools(tools)

                # Run all queries for this domain
                for i, item in enumerate(items):
                    print(f"\n  [{i+1}/{len(items)}] Query: {item.query}")

                    result = BenchmarkResult(
                        uuid=item.uuid,
                        domain=domain,
                        query=item.query,
                        turn_id=item.turn_id,
                    )

                    start_time = time.perf_counter()

                    try:
                        # Shortlist tools per query if enabled
                        if shortlister:
                            query_tools = shortlister.shortlist(item.query, tools)
                            print(f"    Shortlisted {len(query_tools)}/{len(tools)} tools")
                        else:
                            query_tools = tools

                        # Run agent with timeout
                        response = await asyncio.wait_for(
                            agent.run(item.query, query_tools),
                            timeout=AGENT_TIMEOUT_SECONDS
                        )
                        result.answer = response.content
                        result.tool_calls = response.tool_calls
                        result.trajectory = response.trajectory
                        result.status = "success"
                        elapsed = time.perf_counter() - start_time
                        print(f"    Status: success | Tools: {len(result.tool_calls)} | Trajectory steps: {len(result.trajectory)} | Time: {elapsed:.2f}s")
                        # Log the answer
                        print(f"    Answer: {result.answer or '(empty)'}")
                        # Log trajectory summary
                        if result.trajectory:
                            print(f"    Trajectory ({len(result.trajectory)} steps):")
                            for i, step in enumerate(result.trajectory):
                                step_type = step.get('type', 'unknown')
                                if step_type == 'HumanMessage':
                                    print(f"      [{i+1}] User: {step.get('content', '')}")
                                elif step_type == 'AIMessage':
                                    tool_calls = step.get('tool_calls', [])
                                    if tool_calls:
                                        print(f"      [{i+1}] AI: Calling {len(tool_calls)} tool(s)")
                                        for tc in tool_calls:
                                            print(f"          - {tc.get('name', 'unknown')}({tc.get('args', {})})")
                                    else:
                                        print(f"      [{i+1}] AI: {step.get('content', '')}")
                                elif step_type == 'ToolMessage':
                                    tool_name = step.get('tool_name', 'unknown')
                                    result_text = str(step.get('result', ''))
                                    print(f"      [{i+1}] Tool ({tool_name}):\n{result_text}")
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
    top_k_tools: int = 0,
    mcp_domain_env: str = "MCP_DOMAIN",
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
            result = await process_domain(domain, container_runtime, container_name, mcp_domain_env=mcp_domain_env)
            results.append(result)
        return results

    # Create LLM and agent once for all domains
    llm = create_llm(provider=provider, model=model)
    agent = LangGraphReActAgent(llm=llm, model=model or "", provider=provider)
    print(f"Agent: {provider} / {model or 'default'}")

    # Create tool shortlister if requested
    shortlister = None
    if top_k_tools > 0:
        from agents.components.tool_shortlister import ToolShortlister
        shortlister = ToolShortlister(top_k=top_k_tools)
        print(f"Tool shortlister enabled: top_k={top_k_tools}")

    if max_samples_per_domain:
        print(f"Max samples per domain: {max_samples_per_domain}")

    # Process each domain file, writing output incrementally per domain
    all_results: List[BenchmarkResult] = []
    gt_output_dir = input_path.parent / "output"
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
            shortlister=shortlister,
            mcp_domain_env=mcp_domain_env,
        )
        all_results.extend(domain_results)

        # Write output immediately after each domain completes
        save_results_ground_truth(domain_results, gt_output_dir)

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

async def list_tools_for_domains(
    task_id: int,
    container_runtime: str,
    container_name: str,
    domains: Optional[List[str]] = None,
    mcp_domain_env: str = "MCP_DOMAIN",
):
    """List all available tools for specified domains using MCP protocol with detailed parameters."""
    
    print(f"Task ID: {task_id}")
    print(f"Container runtime: {container_runtime}")
    print(f"Container name: {container_name}\n")

    # If domains are specified, use them directly
    # Otherwise, try to discover from JSON files in task directory
    domains_to_process = []
    
    if domains:
        # Use specified domains directly
        domains_to_process = domains
        print(f"Listing tools for {len(domains)} specified domain(s): {domains}")
    else:
        # Try to discover domains from task directory
        if task_id in TASK_PATHS:
            input_path = Path(TASK_PATHS[task_id])
            if input_path.exists():
                json_files = sorted(input_path.glob("*.json"))
                if json_files:
                    domains_to_process = [f.stem for f in json_files]
                    print(f"Discovered {len(domains_to_process)} domains from task directory")
        
        if not domains_to_process:
            print("Error: No domains specified and could not discover from task directory")
            print("Please specify domains using --domain flag")
            sys.exit(1)

    # Collect all tools for OpenAPI spec
    all_tools_by_domain = {}

    # Process each domain
    for domain in domains_to_process:
        
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")
        
        try:
            tools_detailed = await connect_and_get_tools_detailed(domain, container_runtime, container_name, mcp_domain_env=mcp_domain_env)
            print(f"  Total tools: {len(tools_detailed)}\n")
            
            all_tools_by_domain[domain] = tools_detailed
            
            for i, tool in enumerate(tools_detailed, 1):
                print(f"  {i:3d}. {tool['name']}")
                if tool['description']:
                    print(f"       Description: {tool['description'][:100]}{'...' if len(tool['description']) > 100 else ''}")
                
                # Show parameters
                input_schema = tool.get('inputSchema', {})
                properties = input_schema.get('properties', {})
                required = input_schema.get('required', [])
                
                if properties:
                    print(f"       Parameters:")
                    for param_name, param_info in properties.items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', '')
                        required_marker = " (required)" if param_name in required else ""
                        print(f"         - {param_name}: {param_type}{required_marker}")
                        if param_desc:
                            print(f"           {param_desc[:80]}{'...' if len(param_desc) > 80 else ''}")
                print()
                
        except Exception as e:
            print(f"  ERROR: {e}")

    # Save as OpenAPI-like spec
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Include domain names in filename if specified
    if domains and len(domains) <= 3:
        # For 1-3 domains, include them in the filename
        domain_str = "_".join(domains)
        output_file = Path(f"tools_spec_{domain_str}_{timestamp}.json")
    elif domains and len(domains) > 3:
        # For more than 3 domains, just indicate "multiple"
        output_file = Path(f"tools_spec_multiple_domains_{timestamp}.json")
    else:
        # No specific domains (all domains)
        output_file = Path(f"tools_spec_all_{timestamp}.json")
    
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "MCP Tools Specification",
            "version": "1.0.0",
            "description": f"Tools available for task {task_id}"
        },
        "paths": {},
        "components": {
            "schemas": {}
        }
    }
    
    # Convert tools to OpenAPI paths
    for domain, tools in all_tools_by_domain.items():
        for tool in tools:
            path = f"/v1/{domain}/{tool['name']}"
            openapi_spec["paths"][path] = {
                "get": {
                    "summary": tool['description'],
                    "operationId": tool['name'],
                    "parameters": [],
                    "responses": {
                        "200": {
                            "description": "Successful response"
                        }
                    }
                }
            }
            
            # Add parameters
            input_schema = tool.get('inputSchema', {})
            properties = input_schema.get('properties', {})
            required = input_schema.get('required', [])
            
            for param_name, param_info in properties.items():
                openapi_spec["paths"][path]["get"]["parameters"].append({
                    "name": param_name,
                    "in": "query",
                    "required": param_name in required,
                    "schema": {
                        "type": param_info.get('type', 'string'),
                        "description": param_info.get('description', '')
                    }
                })
    
    with open(output_file, 'w') as f:
        json.dump(openapi_spec, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Tool listing complete")
    print(f"{'='*60}")
    print(f"OpenAPI specification saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner for MCP Server")
    parser.add_argument("--task_id", type=int, nargs="+", required=True, help="Task ID(s) to run (e.g. --task_id 2 5)")
    parser.add_argument(
        "--container-runtime",
        type=str,
        default=None,
        help=f"Container runtime to use (default: auto-detect, prefers podman over docker)"
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default=None,
        help="Container name (default: auto-resolved from task config)"
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
        "--list-tools",
        action="store_true",
        help="List available tools for the specified domain(s) and exit"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run multiple task_ids in parallel using asyncio.gather (default: sequential)"
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
        choices=["anthropic", "openai", "ollama", "litellm", "watsonx", "rits"],
        help="LLM provider to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: provider-specific default)"
    )
    parser.add_argument(
        "--top-k-tools",
        type=int,
        default=0,
        help="Enable tool shortlisting: keep top-k tools per query"
    )
    parser.add_argument(
        "--litellm-base-url",
        type=str,
        default=None,
        help="LiteLLM proxy base URL (e.g. http://localhost:4000, or set LITELLM_BASE_URL env var)"
    )
    parser.add_argument(
        "--litellm-api-key",
        type=str,
        default=None,
        help="API key for LiteLLM (or set LITELLM_API_KEY env var)"
    )
    parser.add_argument(
        "--watsonx-project-id",
        type=str,
        default=None,
        help="watsonx.ai project ID (required for watsonx provider, or set WATSONX_PROJECT_ID env var)"
    )
    parser.add_argument(
        "--watsonx-space-id",
        type=str,
        default=None,
        help="watsonx.ai space ID (alternative to project-id, or set WATSONX_SPACE_ID env var)"
    )
    parser.add_argument(
        "--watsonx-api-key",
        type=str,
        default=None,
        help="watsonx.ai API key (or set WATSONX_APIKEY env var)"
    )

    args = parser.parse_args()
    task_ids = args.task_id  # list of ints now

    # Validate all task IDs upfront
    task_cfgs = {}
    for tid in task_ids:
        cfg = TASK_CONFIGS.get(tid)
        if not cfg:
            print(f"Error: Unknown task_id {tid}")
            print(f"Available task_ids: {list(TASK_CONFIGS.keys())}")
            sys.exit(1)
        task_cfgs[tid] = cfg

    # Auto-detect container runtime if not specified
    container_runtime = args.container_runtime
    if not container_runtime:
        container_runtime = detect_container_runtime()
        print(f"Auto-detected container runtime: {container_runtime}")

    # Set provider environment variables if provided via command line
    if args.provider == "watsonx":
        import os
        if args.watsonx_project_id:
            os.environ["WATSONX_PROJECT_ID"] = args.watsonx_project_id
        if args.watsonx_space_id:
            os.environ["WATSONX_SPACE_ID"] = args.watsonx_space_id
        if args.watsonx_api_key:
            os.environ["WATSONX_APIKEY"] = args.watsonx_api_key
    elif args.provider == "litellm":
        import os
        if args.litellm_base_url:
            os.environ["LITELLM_BASE_URL"] = args.litellm_base_url
        if args.litellm_api_key:
            os.environ["LITELLM_API_KEY"] = args.litellm_api_key

    mode = "parallel" if args.parallel and len(task_ids) > 1 else "sequential"
    print("="*60)
    print(f"Benchmark Runner ({mode}, tasks: {task_ids})")
    print("="*60)

    def _make_run_task_coro(tid: int):
        cfg = task_cfgs[tid]
        cname = args.container_name or cfg.container_name
        return run_task(
            task_id=tid,
            container_runtime=container_runtime,
            container_name=cname,
            run_agent=args.run_agent,
            provider=args.provider,
            model=args.model,
            max_samples_per_domain=args.max_samples_per_domain,
            output_file=args.output,
            domains=args.domain,
            top_k_tools=args.top_k_tools,
            mcp_domain_env=cfg.mcp_domain_env,
        )

    def _make_list_tools_coro(tid: int):
        cfg = task_cfgs[tid]
        cname = args.container_name or cfg.container_name
        return list_tools_for_domains(
            task_id=tid,
            container_runtime=container_runtime,
            container_name=cname,
            domains=args.domain,
            mcp_domain_env=cfg.mcp_domain_env,
        )

    # Handle --list-tools mode
    if args.list_tools:
        async def _list_all():
            coros = [_make_list_tools_coro(tid) for tid in task_ids]
            if args.parallel and len(coros) > 1:
                await asyncio.gather(*coros)
            else:
                for c in coros:
                    await c
        asyncio.run(_list_all())
        return

    # Run tasks
    async def _run_all():
        coros = [_make_run_task_coro(tid) for tid in task_ids]
        if args.parallel and len(coros) > 1:
            await asyncio.gather(*coros)
        else:
            for c in coros:
                await c

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
