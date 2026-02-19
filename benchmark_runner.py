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
(default: apis/configs/mcp_connection_config.yaml). Override with --mcp-config.

Usage:
  # Single task, single domain
  python benchmark_runner.py --task_id 2 --domain hockey

  # Single task, multiple domains
  python benchmark_runner.py --task_id 2 --domain hockey --domain address

  # Multiple tasks (sequential, default)
  python benchmark_runner.py --task_id 2 5

  # Multiple tasks in parallel
  python benchmark_runner.py --task_id 2 5 --parallel

  # Limit samples per domain
  python benchmark_runner.py --task_id 2 --max-samples-per-domain 5

  # Choose provider/model
  python benchmark_runner.py --task_id 2 --provider anthropic --model claude-sonnet-4-5-20250929
  python benchmark_runner.py --task_id 2 --provider ollama --model llama3.1:8b

  # Enable tool shortlisting (top-k tools per query)
  python benchmark_runner.py --task_id 2 --top-k-tools 10

  # Custom output directory
  python benchmark_runner.py --task_id 2 --output my_results/

  # Use a custom MCP connection config
  python benchmark_runner.py --task_id 2 --mcp-config my_mcp_config.yaml

  # List available tools for a domain (does not run the benchmark)
  python benchmark_runner.py --task_id 2 --domain hockey --list-tools

Output:
  Results saved to: output/task_{id}_{timestamp}/<domain>.json
  e.g. output/task_2_feb_18_11_21am/hockey.json
"""
import asyncio
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)                                         
logging.getLogger("langchain").setLevel(logging.WARNING)                                      
logging.getLogger("langgraph").setLevel(logging.WARNING)
logging.getLogger("ibm_watsonx_ai").setLevel(logging.WARNING)

from agents.agent_interface import (
    AgentInterface,
    LangGraphReActAgent,
)
from agents.llm import create_llm
from agents.mcp_tool_wrapper import MCPToolWrapper


from benchmark.mcp_client import (
    load_mcp_config,
    create_client_and_connect,
    stop_mcp_server,
    MCPConnectionConfig
    )
from benchmark.runner_helpers import (
    save_results_ground_truth,
    load_benchmark_data,
    log_trajectory,
    log_message_history,
    make_output_dir,
    BenchmarkItem,
    BenchmarkResult,
)
from benchmark.utils import generate_openapi_spec

load_dotenv()


# Default MCP connection config file path
DEFAULT_MCP_CONFIG = str(
    Path(__file__).parent / "apis" / "configs" / "mcp_connection_config.yaml"
)
# Timeout for agent execution (seconds)
AGENT_TIMEOUT_SECONDS = 120


async def run_benchmark_for_domain(
    domain: str,
    items: List[BenchmarkItem],
    cfg: MCPConnectionConfig,
    task_id: int,
    llm,
    max_samples: Optional[int] = None,
    top_k_tools: int = 0,
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

    try:
        async with create_client_and_connect(cfg, domain) as session:
            # Get tools ONCE for this domain
            wrapper = MCPToolWrapper(session)
            tools = await wrapper.get_tools()
            print(f"  Loaded {len(tools)} tools for domain '{domain}'")

            agent = _get_agent(task_id, llm, tools, top_k_tools)

            get_data_tool = next(
                (t for t in tools if t.name == "get_data"), None
            )

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
                    if get_data_tool:
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
                        assert isinstance(agent, LangGraphReActAgent)
                        assert agent.handle_manager is not None
                        handle = agent.handle_manager.store_initial_data(parsed_data)
                        agent._initial_data_handle = handle
                        print(f"    Initial data stored as: {handle}")

                    response = await asyncio.wait_for(
                        agent.run(item.query),
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
                    log_trajectory(result)
                    log_message_history(result)
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


def _get_agent(task_id: int, llm, tools, top_k_tools: int = 0) -> AgentInterface:
    """Return the appropriate agent for the given task_id."""
    if task_id == 1:
        return LangGraphReActAgent(
            llm=llm,
            tools=tools,
            use_handle_manager=True,
            initial_data_handle="placeholder",
            max_iterations=10,
        )
    return LangGraphReActAgent(llm=llm, tools=tools, top_k_tools=top_k_tools)


async def run_task(
    task_id: int,
    cfg: MCPConnectionConfig,
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

    if max_samples_per_domain:
        print(f"Max samples per domain: {max_samples_per_domain}")

    llm = create_llm(provider=provider, model=model)

    # Process each domain, writing output incrementally
    out_dir = make_output_dir(task_id, output_dir)
    all_results: List[BenchmarkResult] = []
    for domain in domain_list:
        items = items_by_domain[domain]
        print(f"\nLoaded {len(items)} items for domain '{domain}'")

        domain_results = await run_benchmark_for_domain(
            domain=domain,
            items=items,
            cfg=cfg,
            task_id=task_id,
            llm=llm,
            max_samples=max_samples_per_domain,
            top_k_tools=top_k_tools,
        )
        all_results.extend(domain_results)
        save_results_ground_truth(domain_results, out_dir)

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
            tools_detailed: List[Dict[str, Any]] = []
            try:
                async with create_client_and_connect(cfg, domain) as session:
                    response = await session.list_tools()
                    for tool in response.tools:
                        tools_detailed.append({
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
            except Exception as exc:
                if "TaskGroup" in str(type(exc).__name__) or "TaskGroup" in str(exc):
                    print(f"  Warning: Cleanup error (ignored): {exc}")
                else:
                    stop_mcp_server(cfg)
                    raise
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

    openapi_spec = generate_openapi_spec(all_tools_by_domain, task_id)
    with open(output_file, "w") as f:
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
