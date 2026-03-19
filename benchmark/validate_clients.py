"""
MCP Client Validation and Tools Listing
========================================

Utilities for validating MCP connections and listing available tools across
all configured tasks.

Usage:
    # Validate all MCP connections
    python -m benchmark.validate_clients

    # Or directly
    python benchmark/validate_clients.py

    # List tools for specific domains (from benchmark_runner.py)
    python benchmark_runner.py --capability_id 2 --domain hockey --list-tools
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.mcp_client import (
    create_client_and_connect,
    load_mcp_config,
    stop_mcp_server,
    MCPConnectionConfig,
)
from benchmark.runner_helpers import load_benchmark_data
from benchmark.utils import generate_openapi_spec

# Suppress noisy logging from MCP servers and related modules
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("benchmark.mcp_client").setLevel(logging.WARNING)


DEFAULT_CONFIG_PATH = str(
    Path(__file__).parent / "mcp_connection_config.yaml"
)


async def list_tools_for_domains(
    capability_id: int,
    cfg: MCPConnectionConfig,
    domains: Optional[List[str]] = None,
):
    """List all available tools for specified domains via MCP protocol."""

    print(f"Capability ID: {capability_id}")
    # Collect all tools for OpenAPI spec
    all_tools_by_domain = {}
    if capability_id in (2, 3, 4):
        # Capabilities 2, 3, 4: per-domain connections — MCP_DOMAIN must be set so
        # the server filters to the requested domain's tools only.
        # Capability 3: bpo_router.py selects BPO or M3 REST based on MCP_DOMAIN.
        # Capability 4: capability_4_mcp_server.py filters both M3 REST and retriever tools.
        _, domains_to_process = load_benchmark_data(
            capability_id=capability_id, domains=domains, domain_names_only=True
        )
        print(
            f"Listing tools for {len(domains_to_process)} domain(s):"
            f" {domains_to_process}"
        )
    else:
        # Task 1: tool universe is domain-independent at the connection level,
        # but if the user specified domains, honour them so MCP_DOMAIN is set
        # and the server loads the correct database.
        if domains:
            domains_to_process = domains
        else:
            domains_to_process = [""]

    for domain in domains_to_process[:5]:
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

    openapi_spec = generate_openapi_spec(all_tools_by_domain, capability_id)
    with open(output_file, "w") as f:
        json.dump(openapi_spec, f, indent=2)

    print("\n" + "=" * 60)
    print("Tool listing complete")
    print("=" * 60)
    print(f"OpenAPI specification saved to: {output_file}")


async def validate_mcp_connection(
    capability_id: int,
    cfg: MCPConnectionConfig,
    domain: str = "superhero",
) -> Dict[str, Any]:
    """
    Validate MCP connection for a single task.

    Args:
        capability_id: Capability ID to validate
        cfg: MCP connection configuration
        domain: Domain to use for testing connection (default: "superhero")

    Returns:
        Dict with keys:
        - capability_id: int
        - status: "connected" | "failed"
        - tool_count: int (0 if failed)
        - error: Optional[str]
        - duration_ms: float
    """
    start_time = time.perf_counter()
    result = {
        "capability_id": capability_id,
        "status": "failed",
        "tool_count": 0,
        "error": None,
        "duration_ms": 0.0,
    }

    try:
        async with create_client_and_connect(cfg, domain=domain) as session:
            response = await session.list_tools()
            result["tool_count"] = len(response.tools)
            result["status"] = "connected"
    except RuntimeError as e:
        # Container not running or not found
        result["error"] = str(e)
    except ConnectionError as e:
        # Server not reachable
        result["error"] = f"Connection error: {e}"
    except TimeoutError as e:
        # Connection timeout
        result["error"] = f"Timeout: {e}"
    except ExceptionGroup as eg:
        # Cleanup errors (Python 3.11+)
        result["error"] = f"Cleanup error: {eg}"
    except Exception as e:
        # Catch-all for other errors
        error_str = str(e)
        # Truncate very long error messages
        if len(error_str) > 500:
            error_str = error_str[:500] + "..."
        result["error"] = error_str

    result["duration_ms"] = (time.perf_counter() - start_time) * 1000
    return result


async def validate_all_tasks(
    config_path: str = DEFAULT_CONFIG_PATH,
    domain: Optional[str] = None,
):
    """
    Validate MCP connections for all configured tasks.

    Args:
        config_path: Path to MCP connection config YAML
        domain: Optional domain to use for testing. If None, tests all available domains.
    """
    print("\n" + "=" * 60)
    print("MCP Connection Validation Report")
    print("=" * 60)

    # Load configuration
    try:
        mcp_configs = load_mcp_config(config_path)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return

    # Task descriptions
    capability_names = {
        1: "SlotFillingMCPServer",
        2: "FastAPIMCPServer",
        3: "BPO MCP Server",
        4: "RetrieverMCPServer",
    }

    # Validate each task
    all_results = []
    for capability_id in sorted(mcp_configs.keys()):
        cfg = mcp_configs[capability_id]
        task_name = capability_names.get(capability_id, f"Capability {capability_id}")

        # Determine which domains to test
        if domain:
            # Use specified domain
            domains_to_test = [domain]
            print(f"\nTask {capability_id} ({task_name}) - Testing domain: {domain}")
        else:
            # Load all available domains for this task
            try:
                _, available_domains = load_benchmark_data(
                    capability_id=capability_id, domain_names_only=True
                )
                if available_domains:
                    domains_to_test = available_domains
                    print(f"\nTask {capability_id} ({task_name}) - Testing {len(domains_to_test)} domain(s)")
                else:
                    # No domains found, use superhero as fallback
                    domains_to_test = ["superhero"]
                    print(f"\nTask {capability_id} ({task_name}) - No domains found, using 'superhero'")
            except Exception as e:
                # If loading domains fails, use superhero as fallback
                domains_to_test = ["superhero"]
                print(f"\nTask {capability_id} ({task_name}) - Error loading domains: {e}")
                print("  Using 'superhero' as fallback")

        # Test each domain
        task_results = []
        for test_domain in domains_to_test:
            result = await validate_mcp_connection(capability_id, cfg, domain=test_domain)
            result["domain"] = test_domain
            task_results.append(result)
            all_results.append(result)

        # Print results for this task
        print("  " + "-" * 56)
        for result in task_results:
            domain_label = f"[{result['domain']}]" if len(domains_to_test) > 1 else ""

            if result["status"] == "connected":
                print(f"  {domain_label:15s} ✓ Connected  |  Tools: {result['tool_count']:3d}  |  {result['duration_ms']:5.0f}ms")
            else:
                print(f"  {domain_label:15s} ✗ Failed     |  Tools:   0  |  {result['duration_ms']:5.0f}ms")
                if result["error"]:
                    error_msg = result["error"]
                    if len(error_msg) > 200:
                        error_msg = error_msg[:200] + "..."
                    print(f"                  Error: {error_msg}")

        # Print task summary
        successful = [r for r in task_results if r["status"] == "connected"]
        total_tools = sum(r["tool_count"] for r in successful)
        print(f"  {'-' * 56}")
        print(f"  Task Summary: {len(successful)}/{len(task_results)} domains connected  |  Total tools: {total_tools}")

    # Print summary
    print("\n" + "=" * 60)
    connected_count = sum(1 for r in all_results if r["status"] == "connected")
    total_count = len(all_results)
    print(f"Summary: {connected_count}/{total_count} connection(s) successful")
    print("=" * 60)


def main():
    """CLI entry point for validating MCP connections."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate MCP connections for all configured tasks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to MCP connection config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain to use for testing connections (default: test all available domains)",
    )
    args = parser.parse_args()

    asyncio.run(validate_all_tasks(args.config, args.domain))


if __name__ == "__main__":
    main()
