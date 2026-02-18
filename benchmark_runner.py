#!/usr/bin/env python3
"""
Benchmark Runner
================

Runs LLM agents against MCP tool servers and records trajectories + answers.

Tasks:
  Task 2  -> fastapi-mcp-server   (M3 SQL tools)
  Task 5  -> retriever-mcp-server (ChromaDB retriever)

Setup:
  pip install langchain-openai langchain mcp langchain-anthropic langgraph langchain-ollama

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
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agents.agent_interface import (
    AgentInterface,
    AgentResponse,
    LangGraphReActAgent,
)
from agents.llm import create_llm
from agents.mcp_tool_wrapper import MCPToolWrapper


from agents.tool_calling_agent import ToolCallingAgent
from benchmark.mcp_client import (
    load_mcp_config,
    create_client_and_connect,
    stop_mcp_server,
    MCPConnectionConfig
    )
from benchmark.processors import (
    save_results_ground_truth,
    load_benchmark_data,
    BenchmarkItem,
    BenchmarkResult
)

load_dotenv()


# Default MCP connection config file path
DEFAULT_MCP_CONFIG = str(
    Path(__file__).parent / "apis" / "configs" / "mcp_connection_config.yaml"
)
# Timeout for agent execution (seconds)
AGENT_TIMEOUT_SECONDS = 120


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


async def connect_and_get_tools(
    domain: str,
    cfg: MCPConnectionConfig,
    verbose: bool = False,
) -> Any:
    """Connect to MCP server and return tool info.

    Args:
        domain: Domain name passed as MCP_DOMAIN to the server.
        cfg: MCP connection configuration.
        verbose: If False (default), return a ``List[str]`` of tool names.
            If True, return a ``List[Dict[str, Any]]`` with ``name``,
            ``description``, and ``inputSchema`` for each tool.
    """

    tools_info: List[Dict[str, Any]] = []
    try:
        async with create_client_and_connect(cfg, domain) as session:
            response = await session.list_tools()
            for tool in response.tools:
                tools_info.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": (
                        tool.inputSchema
                        if hasattr(tool, "inputSchema") else {}
                    ),
                })
        print("  Server stopped.")
    except ExceptionGroup as eg:
        print(f"  Warning: Cleanup error (ignored): {eg}")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            print(f"  Warning: Cleanup error (ignored): {e}")
        else:
            stop_mcp_server(cfg)
            raise

    if verbose:
        return tools_info
    return [t["name"] for t in tools_info]


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
        async with create_client_and_connect(cfg, domain) as session:
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
            tool_names = await connect_and_get_tools(domain, cfg)
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


async def run_benchmark_for_domain(
    domain: str,
    items: List[BenchmarkItem],
    cfg: MCPConnectionConfig,
    make_item_runner,
    max_samples: Optional[int] = None,
    shortlister=None,
) -> List[BenchmarkResult]:
    """Run benchmark for a single domain - starts MCP server once.

    Args:
        make_item_runner: Callable[[tools], async (item, tools) -> AgentResponse]
            Factory called once per domain after tools are loaded.
    """
    import time

    # Limit samples if requested
    if max_samples and max_samples < len(items):
        items = items[:max_samples]

    print("\n" + "#" * 60)
    print(f"# DOMAIN: {domain} ({len(items)} items)")
    print("#" * 60)

    results: List[BenchmarkResult] = []

    try:
        async with create_client_and_connect(cfg, domain) as session:
            # Get tools ONCE for this domain
            wrapper = MCPToolWrapper(session)
            tools = await wrapper.get_tools()
            print(f"  Loaded {len(tools)} tools for domain '{domain}'")

            # Build the per-item runner for this domain's tools
            item_runner = make_item_runner(tools)

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

                    # Run item via the factory-provided runner
                    response = await asyncio.wait_for(
                        item_runner(item, query_tools),
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


def _make_task1_item_runner(llm):
    """Return a domain-level factory for task_id=1 (ToolCallingAgent)."""
    def factory(tools):
        llm_with_tools = llm.bind_tools(tools)
        get_data_tool = next(
            (t for t in tools if t.name == "get_data"), None
        )

        async def item_runner(item, query_tools):
            if not get_data_tool:
                raise RuntimeError(
                    "get_data tool not found in available tools"
                )

            print(f"    Switching to universe: {item.uuid}")
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
                    f"Universe switch failed: {parsed_data['error']}"
                )

            print("    Universe loaded successfully")

            agent = ToolCallingAgent(
                llm_with_tools=llm_with_tools,
                mcp_tools=query_tools,
                initial_data_handle="placeholder",
                max_iterations=10,
            )
            handle = agent.handle_manager.store_initial_data(parsed_data)
            agent._initial_data_handle = handle
            print(f"    Initial data stored as: {handle}")

            answer = await agent.run(item.query)
            return extract_tool_calling_agent_response(agent, answer)

        return item_runner
    return factory


def _make_task2_item_runner(agent):
    """Return a domain-level factory for task_id=2 (LangGraphReActAgent)."""
    def factory(_tools):
        async def item_runner(item, query_tools):
            return await agent.run(item.query, query_tools)
        return item_runner
    return factory


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
        # Just list tools/items without running the agent
        results = []
        for domain in domain_list:
            result = await process_domain(domain, cfg)
            results.append(result)
        return results

    # Create tool shortlister if requested
    shortlister = None
    if top_k_tools > 0:
        from agents.components.tool_shortlister import ToolShortlister
        shortlister = ToolShortlister(top_k=top_k_tools)
        print(f"Tool shortlister enabled: top_k={top_k_tools}")

    if max_samples_per_domain:
        print(f"Max samples per domain: {max_samples_per_domain}")

    # Build the per-domain item runner factory based on task_id
    llm = create_llm(provider=provider, model=model)
    if task_id == 1:
        print(f"Agent: ToolCallingAgent / {provider} / {model or 'default'}")
        item_runner_factory = _make_task1_item_runner(llm)
    else:
        agent = LangGraphReActAgent(
            llm=llm, model=model or "", provider=provider
        )
        print(
            f"Agent: LangGraphReActAgent / {provider} / {model or 'default'}"
        )
        item_runner_factory = _make_task2_item_runner(agent)

    # Process each domain, writing output incrementally
    out_dir = _make_output_dir(task_id, output_dir)
    all_results: List[BenchmarkResult] = []
    for domain in domain_list:
        items = items_by_domain[domain]
        print(f"\nLoaded {len(items)} items for domain '{domain}'")

        domain_results = await run_benchmark_for_domain(
            domain=domain,
            items=items,
            cfg=cfg,
            make_item_runner=item_runner_factory,
            max_samples=max_samples_per_domain,
            shortlister=shortlister,
        )
        all_results.extend(domain_results)
        save_results_ground_truth(domain_results, out_dir / f"{domain}.json")

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
):
    """List all available tools for specified domains via MCP protocol."""

    print(f"Task ID: {task_id}")
    # Collect all tools for OpenAPI spec
    all_tools_by_domain = {}
    if task_id == 2:
        # Task 2: per-domain connections.
        _, domains_to_process = load_benchmark_data(
            task_id=task_id, domains=domains, domain_names_only=True
        )
        print(
            f"Listing tools for {len(domains_to_process)} domain(s):"
            f" {domains_to_process}"
        )
    else:
        # Task 1: same connection for all domains
        domains_to_process = [""]

    for domain in domains_to_process:
        print("\n" + "=" * 60)
        print(f"Domain: {domain}")
        print("=" * 60)
        try:
            tools_detailed = await connect_and_get_tools(
                domain=domain,
                cfg=cfg,
                verbose=True,
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
        "--task_id",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Task ID to run, must be one of [1, 2, 3, 4]"
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

    mode = "parallel" if args.parallel and len(task_ids) > 1 else "sequential"
    print("="*60)
    print(f"Benchmark Runner ({mode}, tasks: {task_ids})")
    print("="*60)

    # Load MCP connection config from YAML
    mcp_configs = load_mcp_config(args.mcp_config)

    def _make_run_task_coro(tid: int):
        task_cfg = mcp_configs.get(tid, MCPConnectionConfig())
        return run_task(
            task_id=tid,
            cfg=task_cfg,
            run_agent=args.run_agent,
            provider=args.provider,
            model=args.model,
            max_samples_per_domain=args.max_samples_per_domain,
            output_dir=args.output,
            domains=args.domain,
            top_k_tools=args.top_k_tools,
        )

    def _make_list_tools_coro(tid: int):
        task_cfg = mcp_configs.get(tid, MCPConnectionConfig())
        return list_tools_for_domains(
            task_id=tid,
            cfg=task_cfg,
            domains=args.domain,
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
