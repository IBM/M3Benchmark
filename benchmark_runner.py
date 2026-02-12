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

MCP connection settings are read from a YAML config file
(default: apis/configs/mcp_connection_config.yaml). Override the path
with --mcp-config.

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

    # Run benchmark on specific domain(s) only
    python benchmark_runner.py --task_id 2 --run-agent --domain hockey
    python benchmark_runner.py --task_id 2 --run-agent --domain hockey \
        --domain address

    # Limit samples per domain (e.g., 5 samples from each domain file)
    python benchmark_runner.py --task_id 2 --run-agent \
        --max-samples-per-domain 5

    # Use different provider/model
    python benchmark_runner.py --task_id 2 --run-agent --provider anthropic
    python benchmark_runner.py --task_id 2 --run-agent \
        --provider ollama --model llama3.1:8b

    # Use a custom MCP connection config
    python benchmark_runner.py --task_id 2 --run-agent \
        --mcp-config my_mcp_config.yaml

Output:
  Results saved to: output/task_{id}_{timestamp}/<domain>.json
  e.g. output/task_5_feb_13_11_21am/address.json
"""
import asyncio
import contextlib
import json
import os
import argparse
import subprocess
import sys
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.agent_interface import (
    AgentInterface,
    AgentResponse,
    LangGraphReActAgent,
)
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
from agents.tool_calling_agent import ToolCallingAgent

load_dotenv()
# Task configurations - maps task_id to input directory path
TASK_PATHS = {
    1: os.environ.get(
        "TASK_1_DIR",
        str(Path(__file__).parent / "data" / "tasks" / "task_1"),
    ),
    2: os.environ.get(
        "TASK_2_DIR",
        str(Path(__file__).parent / "data" / "tasks" / "task_2"),
    ),
}

# Default MCP connection config file path
DEFAULT_MCP_CONFIG = str(
    Path(__file__).parent / "apis" / "configs" / "mcp_connection_config.yaml"
)

# Default settings
DEFAULT_CONTAINER_NAME = "fastapi-mcp-server"


@dataclass
class MCPConnectionConfig:
    """Connection settings for a single task's MCP server."""
    mode: str = "stdio"
    container_name: str = DEFAULT_CONTAINER_NAME
    container_runtime: Optional[str] = None  # None = auto-detect
    container_command: Optional[List[str]] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    server_url: Optional[str] = None


def load_mcp_config(config_path: str) -> Dict[int, MCPConnectionConfig]:
    """Load MCP connection config from YAML file.

    Returns a dict mapping task_id (int) to MCPConnectionConfig.
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    tasks = data.get("tasks", {})
    result = {}
    for k, v in tasks.items():
        cmd = v.get("command", None)
        # "python" in config means the current Python executable (virtualenv-safe)
        if cmd == "python":
            cmd = sys.executable
        result[int(k)] = MCPConnectionConfig(
            mode=v.get("mode", "stdio"),
            container_name=v.get("container_name", DEFAULT_CONTAINER_NAME),
            container_runtime=v.get("container_runtime", None),
            container_command=v.get("container_command", None),
            command=cmd,
            args=v.get("args", None),
            server_url=v.get("server_url", None),
        )
    return result


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


@contextlib.asynccontextmanager
async def connect_to_mcp_server(cfg: MCPConnectionConfig, domain: str = ""):
    """Async context manager yielding (read_stream, write_stream).

    Connection mode is determined by cfg.mode:
    - "stdio" with cfg.command: local subprocess (SlotFilling/SelectionMCPServer)
    - "stdio" without cfg.command: container exec (FastAPIMCPServer)
    - "websocket": WebSocket connection

    Yields:
        (read_stream, write_stream) — pass to ClientSession(read, write)
    """
    if cfg.mode == "stdio":
        if cfg.command:
            # Local subprocess mode
            server_params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args or [],
                env=os.environ.copy(),
            )
        else:
            # Container exec mode (FastAPIMCPServer)
            runtime = cfg.container_runtime
            if not runtime:
                runtime = detect_container_runtime()
                print(f"  Auto-detected container runtime: {runtime}")
            if not cfg.container_name:
                raise ValueError(
                    "stdio mode requires either command or container_name"
                )
            exec_env = {"MCP_DOMAIN": domain} if domain else {}
            env_args = []
            for k, v in exec_env.items():
                env_args += ["-e", f"{k}={v}"]
            cmd = cfg.container_command or ["python", "mcp_server.py"]
            server_params = StdioServerParameters(
                command=runtime,
                args=["exec", "-i"] + env_args + [cfg.container_name] + cmd,
                env=None,
            )
        async with stdio_client(server_params) as (read, write):
            yield read, write

    elif cfg.mode == "websocket":
        if not cfg.server_url:
            raise ValueError("websocket mode requires server_url")
        from mcp.client.websocket import websocket_client
        async with websocket_client(cfg.server_url) as (read, write):
            yield read, write

    else:
        raise ValueError(
            f"Unknown mode: {cfg.mode!r}. Must be 'stdio' or 'websocket'"
        )


def load_benchmark_data(
    task_id: int,
    domains: Optional[List[str]] = None,
    domain_names_only: bool = False,
) -> Tuple[List[BenchmarkItem], List[str]]:
    """Load all benchmark items for a task, optionally filtered by domain.

    Searches <task_dir>/*/input/ for *.json files (one file per domain,
    named <domain>.json). Output directories are ignored.

    Args:
        task_id: Task ID to load data for.
        domains: Optional list of domain names to filter by.
        domain_names_only: If True, skip reading JSON files and return only
            the list of domain names found (items list will be empty).

    Returns:
        Tuple of (items, domain_names) where items is a flat list of
        BenchmarkItem objects (empty when domain_names_only=True) and
        domain_names is a sorted list of domain name strings.
    """
    if task_id not in TASK_PATHS:
        print(f"Error: Unknown task_id {task_id}")
        sys.exit(1)

    input_path = Path(TASK_PATHS[task_id])
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    json_files = sorted(input_path.glob("*/input/*.json"))
    if not json_files:
        print(f"Error: No JSON files found under {input_path}/*/input/")
        sys.exit(1)

    if domains:
        json_files = [f for f in json_files if f.stem in domains]
        if not json_files:
            available = sorted(
                {f.stem for f in input_path.glob("*/input/*.json")}
            )
            print(f"Error: No files found for domains: {domains}")
            suffix = "..." if len(available) > 10 else ""
            print(f"Available domains: {available[:10]}{suffix}")
            sys.exit(1)

    domain_names = sorted({f.stem for f in json_files})

    if domain_names_only:
        return [], domain_names

    items: List[BenchmarkItem] = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        for item_data in data:
            items.append(BenchmarkItem.from_dict(item_data))
    return items, domain_names


def extract_tool_calling_agent_response(agent, answer: str) -> AgentResponse:
    """Extract AgentResponse-compatible data from ToolCallingAgent after run().

    The ToolCallingAgent stores its message history internally but only returns
    a string. This function extracts tool_calls and trajectory from
    agent._messages to create a BenchmarkResult-compatible response.

    Args:
        agent: ToolCallingAgent instance after run() completed
        answer: The string returned by agent.run()

    Returns:
        AgentResponse with tool_calls and trajectory extracted from
        agent._messages
    """
    tool_calls = []
    trajectory = []
    tool_call_args = {}  # Map tool_call_id -> {name, args}

    for msg in agent._messages:
        msg_class = msg.__class__.__name__

        trajectory_entry = {
            "type": msg_class,
            "content": getattr(msg, "content", ""),
        }

        if msg_class == "HumanMessage":
            trajectory.append(trajectory_entry)

        elif msg_class == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                trajectory_entry["tool_calls"] = []
                for tc in msg.tool_calls:
                    tc_id = tc.get("id", "") or tc.get("tool_call_id", "")
                    tool_call_args[tc_id] = {
                        "name": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                    }
                    trajectory_entry["tool_calls"].append({
                        "id": tc_id,
                        "name": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                    })
            trajectory.append(trajectory_entry)

        elif msg_class == "ToolMessage":
            tool_call_id = getattr(msg, "tool_call_id", "")
            tool_info = tool_call_args.get(tool_call_id, {})
            tool_name = (
                getattr(msg, "name", None) or tool_info.get("name", "unknown")
            )
            tool_calls.append({
                "tool_name": tool_name,
                "arguments": tool_info.get("args", {}),
                "result": msg.content,
            })
            trajectory_entry["tool_name"] = tool_name
            trajectory_entry["tool_call_id"] = tool_call_id
            trajectory_entry["result"] = msg.content
            trajectory.append(trajectory_entry)

        elif msg_class == "SystemMessage":
            trajectory.append(trajectory_entry)

    return AgentResponse(
        content=answer,
        tool_calls=tool_calls,
        messages=[],
        metadata={},
        trajectory=trajectory,
    )


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


def save_results_ground_truth(
    results: List[BenchmarkResult], output_dir: Path
):
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
                    k: v for k, v in raw_args.items()
                    if k not in _INTERNAL_KEYS
                }
                gold_sequence.append({
                    "tool_call": [[{
                        "name": tc.get("tool_name", ""),
                        "arguments": filtered_args,
                    }]],
                    "tool_response": [
                        _extract_tool_response_values(tc.get("result", ""))
                    ],
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
    raise RuntimeError(
        "Neither podman nor docker found in PATH. "
        "Please install one of them."
    )


def stop_mcp_server(cfg: MCPConnectionConfig):
    """Stop a running MCP server process.

    For container stdio mode, force-kills the mcp_server.py process inside the
    container.  For subprocess stdio and websocket modes the transport's own
    context manager handles teardown, so this is a no-op.
    """
    if cfg.mode == "websocket":
        # WebSocket connection is closed by the context manager; nothing to do.
        return

    if cfg.mode == "stdio" and not cfg.command and cfg.container_name:
        # Container exec mode: pkill the server process inside the container.
        runtime = cfg.container_runtime or detect_container_runtime()
        try:
            kill_cmd = [
                runtime, "exec", cfg.container_name,
                "pkill", "-f", "python mcp_server.py"
            ]
            subprocess.run(kill_cmd, capture_output=True, timeout=5)
            print("  Server stopped.")
        except subprocess.TimeoutExpired:
            print("  Warning: Timeout while stopping server")
        except Exception:
            pass
    # Local subprocess stdio: the stdio_client context manager terminates the
    # child process on exit, so no explicit kill is needed here.


async def connect_and_list_tools(
    domain: str,
    cfg: MCPConnectionConfig,
) -> List[str]:
    """Connect to MCP server with the given domain and list available tools."""
    runtime = cfg.container_runtime or "(auto-detect)"
    print(
        f"  Starting: {runtime} exec -i"
        f" -e MCP_DOMAIN={domain} {cfg.container_name} python mcp_server.py"
    )

    tool_names = []

    try:
        async with connect_to_mcp_server(cfg, domain) as (read, write):
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
            stop_mcp_server(cfg)
            raise

    return tool_names


async def connect_and_get_tools_detailed(
    domain: str,
    cfg: MCPConnectionConfig,
) -> List[Dict[str, Any]]:
    """Connect to MCP server and get detailed tool info."""
    if cfg.mode == "websocket":
        print(f"  Connecting via websocket: {cfg.server_url}")
    elif cfg.command:
        print(
            f"  Starting: {cfg.command}"
            f" {' '.join(cfg.args or [])}"
        )
    else:
        runtime = cfg.container_runtime or "(auto-detect)"
        print(
            f"  Starting: {runtime} exec -i"
            f" -e MCP_DOMAIN={domain} {cfg.container_name} python mcp_server.py"
        )

    tools_detailed = []

    try:
        async with connect_to_mcp_server(cfg, domain) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()

                for tool in response.tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": (
                            tool.inputSchema
                            if hasattr(tool, 'inputSchema') else {}
                        ),
                    }
                    tools_detailed.append(tool_info)

        print("  Server stopped.")
    except ExceptionGroup as eg:
        print(f"  Warning: Cleanup error (ignored): {eg}")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            print(f"  Warning: Cleanup error (ignored): {e}")
        else:
            stop_mcp_server(cfg)
            raise

    return tools_detailed


async def run_agent_with_query(
    domain: str,
    query: str,
    cfg: MCPConnectionConfig,
    agent: AgentInterface,
) -> AgentResponse:
    """Run an agent with tools from the MCP server."""
    if cfg.mode == "stdio" and not cfg.command and cfg.container_name:
        runtime = cfg.container_runtime or "(auto-detect)"
        print(
            f"  Starting: {runtime} exec -i"
            f" -e MCP_DOMAIN={domain} {cfg.container_name} python mcp_server.py"
        )
    elif cfg.mode == "websocket":
        print(f"  Connecting via websocket: {cfg.server_url}")
    else:
        print(
            f"  Starting: {cfg.command}"
            f" {' '.join(cfg.args or [])}"
        )

    response = None

    try:
        async with connect_to_mcp_server(cfg, domain) as (read, write):
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
                    print(
                        f"  Agent completed."
                        f" Response received: {response is not None}"
                    )
                except asyncio.TimeoutError:
                    print(
                        f"  Agent timed out after {AGENT_TIMEOUT_SECONDS}s"
                    )
                    raise TimeoutError(
                        f"Agent timed out after {AGENT_TIMEOUT_SECONDS}"
                        " seconds"
                    )
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
            stop_mcp_server(cfg)
            raise

    if response is None:
        raise RuntimeError("Agent did not return a response")
    return response


async def process_domain(
    domain: str,
    cfg: MCPConnectionConfig,
    agent: Optional[AgentInterface] = None,
    query: Optional[str] = None,
    mcp_domain_env: str = "MCP_DOMAIN",
) -> dict:
    """Process a single domain: connect to MCP server and list tools."""

    print("\n" + "=" * 60)
    print(f"Domain: {domain}")
    print("=" * 60)

    try:
        if agent and query:
            # Run agent with query
            response = await run_agent_with_query(domain, query, cfg, agent)
            print(f"  Tool calls: {len(response.tool_calls)}")
            if len(response.content) > 200:
                print(f"  Answer: {response.content[:200]}...")
            else:
                print(f"  Answer: {response.content}")

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
            tool_names = await connect_and_list_tools(domain, cfg)
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
    cfg: MCPConnectionConfig,
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
            cfg=cfg,
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
    cfg: MCPConnectionConfig,
    agent: AgentInterface,
    max_samples: Optional[int] = None,
    shortlister=None,
) -> List[BenchmarkResult]:
    """Run benchmark for a single domain - starts MCP server once."""
    import time

    # Limit samples if requested
    if max_samples and max_samples < len(items):
        items = items[:max_samples]

    print("\n" + "#" * 60)
    print(f"# DOMAIN: {domain} ({len(items)} items)")
    print("#" * 60)

    results: List[BenchmarkResult] = []

    if cfg.mode == "stdio" and not cfg.command and cfg.container_name:
        runtime = cfg.container_runtime or "(auto-detect)"
        print(
            f"  Starting MCP server: {runtime} exec -i"
            f" -e MCP_DOMAIN={domain} {cfg.container_name} python mcp_server.py"
        )
    elif cfg.mode == "websocket":
        print(
            f"  Connecting to MCP server via websocket: {cfg.server_url}"
        )
    else:
        print(
            f"  Starting MCP server: {cfg.command}"
            f" {' '.join(cfg.args or [])}"
        )

    try:
        async with connect_to_mcp_server(cfg, domain) as (read, write):
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
                    query_suffix = (
                        "..." if len(item.query) > 80 else ""
                    )
                    print(
                        f"\n  [{i+1}/{len(items)}]"
                        f" Query: {item.query[:80]}{query_suffix}"
                    )

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
                            query_tools = shortlister.shortlist(
                                item.query, tools
                            )
                            print(
                                f"    Shortlisted"
                                f" {len(query_tools)}/{len(tools)} tools"
                            )
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
                        print(
                            f"    Status: success"
                            f" | Tools: {len(result.tool_calls)}"
                            f" | Trajectory steps:"
                            f" {len(result.trajectory)}"
                            f" | Time: {elapsed:.2f}s"
                        )
                        # Log the answer
                        answer_preview = (
                            result.answer[:200]
                            if result.answer else "(empty)"
                        )
                        ans_suffix = (
                            "..." if len(result.answer) > 200 else ""
                        )
                        print(
                            f"    Answer: {answer_preview}{ans_suffix}"
                        )
                        # Log trajectory summary
                        if result.trajectory:
                            traj_len = len(result.trajectory)
                            print(
                                f"    Trajectory ({traj_len} steps):"
                            )
                            for i, step in enumerate(result.trajectory):
                                step_type = step.get('type', 'unknown')
                                if step_type == 'HumanMessage':
                                    content_preview = (
                                        step.get('content', '')[:80]
                                    )
                                    c_len = len(
                                        step.get('content', '')
                                    )
                                    c_suffix = (
                                        "..." if c_len > 80 else ""
                                    )
                                    print(
                                        f"      [{i+1}] User:"
                                        f" {content_preview}{c_suffix}"
                                    )
                                elif step_type == 'AIMessage':
                                    content_preview = (
                                        step.get('content', '')[:80]
                                    )
                                    tool_calls = step.get(
                                        'tool_calls', []
                                    )
                                    if tool_calls:
                                        print(
                                            f"      [{i+1}] AI: Calling"
                                            f" {len(tool_calls)} tool(s)"
                                        )
                                        for tc in tool_calls:
                                            tc_name = tc.get(
                                                'name', 'unknown'
                                            )
                                            tc_args = tc.get('args', {})
                                            print(
                                                f"          - {tc_name}"
                                                f"({tc_args})"
                                            )
                                    else:
                                        c_len = len(
                                            step.get('content', '')
                                        )
                                        c_suffix = (
                                            "..." if c_len > 80 else ""
                                        )
                                        print(
                                            f"      [{i+1}] AI:"
                                            f" {content_preview}"
                                            f"{c_suffix}"
                                        )
                                elif step_type == 'ToolMessage':
                                    tool_name = step.get(
                                        'tool_name', 'unknown'
                                    )
                                    result_preview = str(
                                        step.get('result', '')
                                    )[:80]
                                    r_len = len(
                                        str(step.get('result', ''))
                                    )
                                    r_suffix = (
                                        "..." if r_len > 80 else ""
                                    )
                                    print(
                                        f"      [{i+1}] Tool"
                                        f" ({tool_name}):"
                                        f" {result_preview}{r_suffix}"
                                    )
                    except asyncio.TimeoutError:
                        result.status = "error"
                        result.error = (
                            f"Agent timed out after"
                            f" {AGENT_TIMEOUT_SECONDS} seconds"
                        )
                        print(
                            f"    Status: timeout after"
                            f" {AGENT_TIMEOUT_SECONDS}s"
                        )
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
            stop_mcp_server(cfg)
            raise

    return results


async def run_task1_benchmark(
    items: List[BenchmarkItem],
    provider: str,
    model: Optional[str],
    cfg: MCPConnectionConfig,
    max_samples: Optional[int] = None,
) -> List[BenchmarkResult]:
    """Run task_id=1 benchmark using ToolCallingAgent + SlotFillingMCPServer.

    Key differences from task_id=2:
    - Single MCP server for all instances (universe switching via get_data)
    - ToolCallingAgent with handle-based result management

    Args:
        items: List of BenchmarkItem objects to process
        provider: LLM provider name
        model: Model name (optional)
        cfg: MCP connection configuration
        max_samples: Maximum number of instances to process

    Returns:
        List of BenchmarkResult objects
    """
    import time

    results: List[BenchmarkResult] = []

    # Limit samples if requested
    if max_samples:
        items = items[:max_samples]

    print("\n" + "#" * 60)
    print("# TASK 1: ToolCallingAgent + SlotFillingMCPServer")
    print(f"# Processing {len(items)} instances")
    print("#" * 60)

    # Create LLM once
    llm = create_llm(provider=provider, model=model)
    print(f"  LLM: {provider}" + (f" / {model}" if model else ""))

    # Connect to single MCP server for all instances
    try:
        async with connect_to_mcp_server(cfg, domain="") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("  MCP session initialized")

                # Get tools once (they don't change between universes)
                wrapper = MCPToolWrapper(
                    session=session,
                    use_openai_restrictions=(provider in ("openai", "rits"))
                )
                tools = await wrapper.get_tools()
                print(f"  Loaded {len(tools)} tools")

                # Bind tools to LLM once
                llm_with_tools = llm.bind_tools(tools)

                # Process each instance
                for i, item in enumerate(items):
                    print(f"\n  [{i+1}/{len(items)}] Instance: {item.uuid}")
                    q_suffix = "..." if len(item.query) > 80 else ""
                    print(
                        f"  Query: {item.query[:80]}{q_suffix}"
                    )

                    result = BenchmarkResult(
                        uuid=item.uuid,
                        domain=item.domain,
                        query=item.query,
                        turn_id=item.turn_id,
                    )

                    start_time = time.perf_counter()

                    try:
                        # Step 1: Switch universe via get_data tool
                        get_data_tool = next(
                            (t for t in tools if t.name == "get_data"),
                            None,
                        )
                        if not get_data_tool:
                            raise RuntimeError(
                                "get_data tool not found in available tools"
                            )

                        print(
                            f"    Switching to universe: {item.uuid}"
                        )
                        data_result = await get_data_tool.ainvoke(
                            {"tool_universe_id": item.uuid}
                        )
                        parsed_data = json.loads(data_result)

                        # Handle MCP TextContent format
                        if isinstance(parsed_data, list) and parsed_data:
                            first_item = parsed_data[0]
                            if (
                                isinstance(first_item, dict)
                                and "text" in first_item
                            ):
                                parsed_data = json.loads(first_item["text"])
                            else:
                                parsed_data = first_item

                        if (
                            isinstance(parsed_data, dict)
                            and "error" in parsed_data
                        ):
                            raise RuntimeError(
                                f"Universe switch failed:"
                                f" {parsed_data['error']}"
                            )

                        print("    Universe loaded successfully")

                        # Step 2: Create agent for this instance
                        agent = ToolCallingAgent(
                            llm_with_tools=llm_with_tools,
                            mcp_tools=tools,
                            initial_data_handle="placeholder",
                            max_iterations=10
                        )

                        # Store initial data and update handle
                        initial_handle = (
                            agent.handle_manager.store_initial_data(
                                parsed_data
                            )
                        )
                        agent._initial_data_handle = initial_handle
                        print(
                            f"    Initial data stored as: {initial_handle}"
                        )

                        # Step 3: Run agent with timeout
                        answer = await asyncio.wait_for(
                            agent.run(item.query),
                            timeout=AGENT_TIMEOUT_SECONDS
                        )

                        # Step 4: Extract response data from agent
                        response = extract_tool_calling_agent_response(
                            agent, answer
                        )

                        result.answer = response.content
                        result.tool_calls = response.tool_calls
                        result.trajectory = response.trajectory
                        result.status = "success"

                        elapsed = time.perf_counter() - start_time
                        print(
                            f"    Status: success"
                            f" | Tools: {len(result.tool_calls)}"
                            f" | Time: {elapsed:.2f}s"
                        )
                        answer_preview = (
                            result.answer[:200]
                            if result.answer else "(empty)"
                        )
                        ans_suffix = (
                            "..." if len(result.answer) > 200 else ""
                        )
                        print(
                            f"    Answer: {answer_preview}{ans_suffix}"
                        )

                    except asyncio.TimeoutError:
                        result.status = "error"
                        result.error = (
                            f"Agent timed out after"
                            f" {AGENT_TIMEOUT_SECONDS}s"
                        )
                        print(
                            f"    Status: timeout after"
                            f" {AGENT_TIMEOUT_SECONDS}s"
                        )
                    except Exception as e:
                        result.status = "error"
                        result.error = str(e)
                        print(f"    Status: error | {str(e)[:80]}")
                        import traceback
                        traceback.print_exc()

                    result.duration_s = time.perf_counter() - start_time
                    results.append(result)

        print("\n  Server stopped")
    except ExceptionGroup as eg:
        print(f"  Warning: Cleanup error (ignored): {eg}")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            print(f"  Warning: Cleanup error (ignored): {e}")
        else:
            raise

    return results


def _make_output_dir(task_id: int, output_dir: Optional[str] = None) -> Path:
    """Create a timestamped output directory for a task under CWD.

    Format: output/task_{id}_{Mon}_{dd}_{hh}_{mm}{am|pm}/
    e.g.    output/task_5_Feb_13_11_21am/
    """
    if output_dir:
        p = Path(output_dir)
    else:
        from datetime import datetime
        now = datetime.now()
        ts = now.strftime("%b_%d_%I_%M%p").lower()  # e.g. feb_13_11_21am
        p = Path("output") / f"task_{task_id}_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


async def run_task(
    task_id: int,
    cfg: MCPConnectionConfig,
    run_agent: bool = False,
    provider: str = "ollama",
    model: Optional[str] = None,
    max_samples_per_domain: Optional[int] = None,
    output_dir: Optional[str] = None,
    domains: Optional[List[str]] = None,
    top_k_tools: int = 0,
) -> List[BenchmarkResult]:
    """Run benchmark for a given task_id, iterating over all domain files."""

    if task_id not in TASK_PATHS:
        print(f"Error: Unknown task_id {task_id}")
        print(f"Available task_ids: {list(TASK_PATHS.keys())}")
        sys.exit(1)

    # ============================================================
    # TASK ID 1: ToolCallingAgent + SlotFillingMCPServer
    # ============================================================
    if task_id == 1:
        items, _ = load_benchmark_data(task_id=1, domains=domains)
        print(f"Loaded {len(items)} items")

        if not run_agent:
            # Just list available items for task 1
            print(f"\nAvailable items ({len(items)}):")
            for item in items[:10]:
                print(f"  {item.uuid}: {item.query[:60]}...")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")
            return []

        # Run task 1 benchmark
        results = await run_task1_benchmark(
            items=items,
            provider=provider,
            model=model,
            cfg=cfg,
            max_samples=max_samples_per_domain,
        )

        # Save results
        if results:
            output_dir = Path(TASK_PATHS[task_id]).parent / "output"
            save_results_ground_truth(results, output_dir)

        # Print summary
        successful = [r for r in results if r.status == "success"]
        failed = [r for r in results if r.status == "error"]
        print("\n" + "=" * 60)
        print("TASK 1 BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"  Total items: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        return results

    # ============================================================
    # TASK ID 2: LangGraphReActAgent + FastAPIMCPServer (existing)
    # ============================================================
    all_items, _ = load_benchmark_data(task_id=task_id, domains=domains)

    # Group items by domain
    items_by_domain: Dict[str, List[BenchmarkItem]] = {}
    for item in all_items:
        items_by_domain.setdefault(item.domain, []).append(item)

    domain_list = sorted(items_by_domain)
    print(f"Task ID: {task_id}")
    print(f"Mode: {cfg.mode}")
    if not cfg.command and cfg.mode == "stdio":
        print(f"Container name: {cfg.container_name}")
    print(f"Processing {len(domain_list)} domain(s): {domain_list}")

    if not run_agent:
        # Just list tools for each domain (original behavior)
        results = []
        for domain in domain_list:
            result = await process_domain(domain, cfg)
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

    # Process each domain, writing output incrementally
    all_results: List[BenchmarkResult] = []
    gt_output_dir = Path(TASK_PATHS[task_id]).parent / "output"
    for domain in domain_list:
        items = items_by_domain[domain]
        print(f"\nLoaded {len(items)} items for domain '{domain}'")

        # Run benchmark for this domain
        domain_results = await run_benchmark_for_domain(
            domain=domain,
            items=items,
            cfg=cfg,
            agent=agent,
            max_samples=max_samples_per_domain,
            shortlister=shortlister,
        )
        all_results.extend(domain_results)

        # Write output immediately after each domain completes
        save_results_ground_truth(domain_results, gt_output_dir)

    results = all_results

    # Summary
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "error"]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Total items: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    return results


async def list_tools_for_domains(
    task_id: int,
    cfg: MCPConnectionConfig,
    domains: Optional[List[str]] = None,
    mcp_domain_env: str = "MCP_DOMAIN",
):
    """List all available tools for specified domains via MCP protocol."""

    print(f"Task ID: {task_id}")
    if cfg.mode == "websocket":
        print(f"Server URL: {cfg.server_url}\n")
    elif cfg.command:
        print(
            f"Subprocess: {cfg.command}"
            f" {' '.join(cfg.args or [])}\n"
        )
    else:
        runtime = cfg.container_runtime or "(auto-detect)"
        print(f"Container runtime: {runtime}")
        print(f"Container name: {cfg.container_name}\n")

    _, domains_to_process = load_benchmark_data(
        task_id=task_id, domains=domains, domain_names_only=True
    )
    print(
        f"Listing tools for {len(domains_to_process)} domain(s):"
        f" {domains_to_process}"
    )

    # Collect all tools for OpenAPI spec
    all_tools_by_domain = {}

    if task_id == 1:
        # Task 1: single subprocess server serves all domains (same tools
        # regardless of domain). Connect once and report under "task_1".
        print("\n" + "=" * 60)
        print("Task 1 tools (single server)")
        print("=" * 60)
        try:
            tools_detailed = await connect_and_get_tools_detailed(
                domain="",
                cfg=cfg,
            )
            all_tools_by_domain["task_1"] = tools_detailed
            print(f"  Total tools: {len(tools_detailed)}\n")
            for i, tool in enumerate(tools_detailed, 1):
                print(f"  {i:3d}. {tool['name']}")
                if tool['description']:
                    desc = tool['description']
                    d_suffix = "..." if len(desc) > 100 else ""
                    print(
                        f"       Description: {desc[:100]}{d_suffix}"
                    )
                input_schema = tool.get('inputSchema', {})
                properties = input_schema.get('properties', {})
                required = input_schema.get('required', [])
                if properties:
                    print("       Parameters:")
                    for param_name, param_info in properties.items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', '')
                        req_marker = (
                            " (required)"
                            if param_name in required else ""
                        )
                        print(
                            f"         - {param_name}:"
                            f" {param_type}{req_marker}"
                        )
                        if param_desc:
                            pd_suffix = (
                                "..." if len(param_desc) > 80 else ""
                            )
                            print(
                                f"           {param_desc[:80]}{pd_suffix}"
                            )
                print()
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        # Task 2: per-domain connections.
        for domain in domains_to_process:
            print("\n" + "=" * 60)
            print(f"Domain: {domain}")
            print("=" * 60)
            try:
                tools_detailed = await connect_and_get_tools_detailed(
                    domain=domain,
                    cfg=cfg,
                )
                print(f"  Total tools: {len(tools_detailed)}\n")
                all_tools_by_domain[domain] = tools_detailed
                for i, tool in enumerate(tools_detailed, 1):
                    print(f"  {i:3d}. {tool['name']}")
                    if tool['description']:
                        desc = tool['description']
                        d_suffix = "..." if len(desc) > 100 else ""
                        print(
                            f"       Description: {desc[:100]}{d_suffix}"
                        )
                    input_schema = tool.get('inputSchema', {})
                    properties = input_schema.get('properties', {})
                    required = input_schema.get('required', [])
                    if properties:
                        print("       Parameters:")
                        for param_name, param_info in properties.items():
                            param_type = param_info.get('type', 'unknown')
                            param_desc = param_info.get('description', '')
                            req_marker = (
                                " (required)"
                                if param_name in required else ""
                            )
                            print(
                                f"         - {param_name}:"
                                f" {param_type}{req_marker}"
                            )
                            if param_desc:
                                pd_suffix = (
                                    "..." if len(param_desc) > 80 else ""
                                )
                                print(
                                    f"           "
                                    f"{param_desc[:80]}{pd_suffix}"
                                )
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
        output_file = Path(
            f"tools_spec_multiple_domains_{timestamp}.json"
        )
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

    print("\n" + "=" * 60)
    print("Tool listing complete")
    print("=" * 60)
    print(f"OpenAPI specification saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Runner for MCP Server"
    )
    parser.add_argument(
        "--task_id", type=int, nargs="+", required=True, help="Task ID to run"
    )
    parser.add_argument(
        "--domain",
        type=str,
        action="append",
        default=None,
        help=(
            "Domain(s) to process"
            " (can specify multiple times, default: all domains)"
        ),
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
        help="Output directory (default: output/task_{id}_{timestamp}/ in CWD)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=[
            "anthropic", "openai", "ollama",
            "litellm", "watsonx", "rits",
        ],
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
        "--mcp-config",
        type=str,
        default=DEFAULT_MCP_CONFIG,
        help=(
            f"Path to MCP connection config YAML file"
            f" (default: {DEFAULT_MCP_CONFIG})"
        ),
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

    mode = "parallel" if args.parallel and len(task_ids) > 1 else "sequential"
    print("="*60)
    print(f"Benchmark Runner ({mode}, tasks: {task_ids})")
    print("="*60)
    print("=" * 60)
    print("Benchmark Runner")
    print("=" * 60)

    # Load MCP connection config from YAML
    mcp_configs = load_mcp_config(args.mcp_config)
    cfg = mcp_configs.get(args.task_id, MCPConnectionConfig())

    def _make_run_task_coro(tid: int):
        task_cfg = task_cfgs[tid]
        cname = cfg.container_name
        return run_task(
            task_id=tid,
            container_runtime=cfg.container_runtime,
            container_name=cname,
            run_agent=args.run_agent,
            provider=args.provider,
            model=args.model,
            max_samples_per_domain=args.max_samples_per_domain,
            output_dir=args.output,
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
    subprocess_args = args.subprocess_args.split() if args.subprocess_args else None

        asyncio.run(list_tools_for_domains(
            task_id=args.task_id,
            cfg=cfg,
            domains=args.domain,
        ))
        return

    asyncio.run(run_task(
        task_id=args.task_id,
        cfg=cfg,
        run_agent=args.run_agent,
        provider=args.provider,
        model=args.model,
        max_samples_per_domain=args.max_samples_per_domain,
        output_file=args.output,
        domains=args.domain,
        top_k_tools=args.top_k_tools,
    ))


if __name__ == "__main__":
    main()
