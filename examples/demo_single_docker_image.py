


import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
import shutil
import sys
import argparse
import contextlib

# Add parent directory to path so we can import agents module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.websocket import websocket_client

from agents.tool_calling_agent import ToolCallingAgent
from agents.llm import create_llm
from agents.mcp_tool_wrapper import MCPToolWrapper


def detect_container_runtime() -> str:
    """Detect available container runtime (podman or docker)."""
    if shutil.which("podman"):
        return "podman"
    if shutil.which("docker"):
        return "docker"
    raise RuntimeError("Neither podman nor docker found in PATH.")


@contextlib.asynccontextmanager
async def connect_to_server(
    mode: str,
    server_url: str | None = None,
    container_runtime: str | None = None,
    container_name: str = "m3_environ",
    domain: str | None = None,
):
    """Connect to MCP server using specified mode.

    Args:
        mode: "stdio" (subprocess), "docker" (container exec), or "websocket"
        server_url: WebSocket URL (required for websocket mode)
        container_runtime: "docker" or "podman" (auto-detected if None)
        container_name: Container name for docker mode (default: m3_environ)
        domain: Domain for MCP_DOMAIN env var (default: from MCP_DOMAIN env or "superhero")

    Yields:
        (read_stream, write_stream): Communication streams
    """
    if mode == "stdio":
        # Local subprocess: launch MCP server directly
        python_exe = sys.executable
        server_params = StdioServerParameters(
            command=python_exe,
            args=["-m", "apis.m3.python_tools.mcp"],
            env=os.environ.copy(),
        )
        async with stdio_client(server_params) as (read, write):
            yield read, write

    elif mode == "docker":
        # Docker/Podman exec: run MCP server inside the unified container
        runtime = container_runtime or detect_container_runtime()
        mcp_domain = domain or os.environ.get("MCP_DOMAIN", "superhero")
        server_params = StdioServerParameters(
            command=runtime,
            args=[
                "exec", "-i",
                "-e", f"MCP_DOMAIN={mcp_domain}",
                container_name,
                "python", "-m", "apis.m3.python_tools.mcp",
            ],
        )
        print(f"  Using {runtime} exec into '{container_name}' (MCP_DOMAIN={mcp_domain})")
        async with stdio_client(server_params) as (read, write):
            yield read, write

    elif mode == "websocket":
        if not server_url:
            raise ValueError("server_url required for websocket mode")
        async with websocket_client(server_url) as (read, write):
            yield read, write

    else:
        raise ValueError(f"Unknown mode: {mode}")


async def run_single_server_with_instances(
    llm,
    instances: dict,
    provider: str,
    mode: str = "stdio",
    server_url: str | None = None,
    max_samples_per_domain: int | None = None,
    container_runtime: str | None = None,
    container_name: str = "m3_environ",
    domain: str | None = None,
):
    """Run queries against a single MCP server, switching universes for each instance.

    Args:
        llm: The language model instance
        instances: Dict mapping instance_id -> {query, init_args}
        provider: LLM provider name (e.g. "rits", "ollama", "anthropic", "openai", "litellm", "watsonx")
        mode: Connection mode ("stdio", "docker", or "websocket")
        server_url: WebSocket URL (required for websocket mode)
        max_samples_per_domain: Maximum number of instances to process (None = all)
        container_runtime: "docker" or "podman" (auto-detected if None; docker mode only)
        container_name: Container name (docker mode only, default: m3_environ)
        domain: Domain for MCP_DOMAIN env var (docker mode only)
    """
    print(f"\n{'='*60}")
    print(f"Connecting to MCP server (mode: {mode})")
    if mode == "websocket":
        print(f"Server URL: {server_url}")
    elif mode == "docker":
        print(f"Container: {container_name}")
    else:
        print(f"Starting SINGLE MCP server for {len(instances)} instances")
    print(f"{'='*60}\n")

    async with connect_to_server(
        mode, server_url,
        container_runtime=container_runtime,
        container_name=container_name,
        domain=domain,
    ) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize session once
            await session.initialize()
            print("MCP session initialized")

            # Use new MCPToolWrapper factory pattern
            wrapper = MCPToolWrapper(
                session=session,
                use_openai_restrictions=(provider in ("openai", "rits"))
            )
            tools = await wrapper.get_tools()

            # Bind tools to LLM
            llm_with_tools = llm.bind_tools(tools)

            print(f"Available tools ({len(tools)}): {[t.name for t in tools]}\n")

            # Loop over each instance
            items = list(instances.items())
            if max_samples_per_domain is not None:
                items = items[:max_samples_per_domain]
            for instance_id, instance in items:
                query = instance['query']
                print(f"\n{'='*60}")
                print(f"Processing Instance: {instance_id}")
                print(f"{'='*60}\n")

                try:
                    # STEP 1: Call get_data with tool_universe_id to switch and load data
                    print(f"Switching to universe: {instance_id} and loading data")

                    get_data_tool = next((t for t in tools if t.name == "get_data"), None)
                    if not get_data_tool:
                        raise RuntimeError("get_data tool not found!")

                    # Call get_data with tool_universe_id parameter
                    # This will switch universe and return the active_data
                    data_result = await get_data_tool.ainvoke({"tool_universe_id": instance_id})

                    # Parse result (it comes back as MCP TextContent list)
                    parsed_data = json.loads(data_result)
                    if isinstance(parsed_data, list) and parsed_data:
                        # Extract text from MCP response format
                        first_item = parsed_data[0]
                        if isinstance(first_item, dict) and "text" in first_item:
                            parsed_data = json.loads(first_item["text"])
                        else:
                            parsed_data = first_item

                    if isinstance(parsed_data, dict) and "error" in parsed_data:
                        raise RuntimeError(f"Failed to switch universe: {parsed_data['error']}")

                    print(f"Successfully loaded universe '{instance_id}'")

                    # STEP 2: Create agent and run the query
                    # Note: No toolkit refresh needed - key_name validation happens at runtime
                    agent = ToolCallingAgent(
                        llm_with_tools=llm_with_tools,
                        mcp_tools=tools,
                        initial_data_handle="initial_data_1",  # Temporary, will be updated
                        max_iterations=10
                    )

                    # Store initial data in agent's handle manager and update handle
                    initial_data_handle = agent.handle_manager.store_initial_data(parsed_data)
                    agent._initial_data_handle = initial_data_handle
                    print(f"Initial data stored as: {initial_data_handle}\n")

                    # Run agent
                    await agent.run(query)

                except Exception as e:
                    print(f"\nError processing instance '{instance_id}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue


async def main(
    provider="rits",
    model=None,
    mode="stdio",
    server_url=None,
    max_samples_per_domain=None,
    container_runtime=None,
    container_name="m3_environ",
    domain=None,
):
    """Main async entry point with single server for all instances.

    Args:
        provider: LLM provider ("rits", "ollama", "anthropic", "openai", "litellm", "watsonx")
        model: Model name (optional)
        mode: Connection mode - "stdio" (subprocess), "docker" (container), or "websocket"
        server_url: WebSocket URL (required for websocket mode)
        max_samples_per_domain: Maximum number of instances to process (None = all)
        container_runtime: "docker" or "podman" (auto-detected if None; docker mode only)
        container_name: Container name (docker mode only, default: m3_environ)
        domain: Domain for MCP_DOMAIN env var (docker mode only)
    """

    # Create LLM instance
    llm = create_llm(provider=provider, model=model)
    print(f"\n{'='*60}")
    print(f"Using LLM: {provider}" + (f" (model: {model})" if model else ""))
    print(f"{'='*60}\n")

    # Load all instances
    with open("apis/configs/mcp_init_mapping.json") as f:
        instances = json.load(f)

    # Run with single server
    await run_single_server_with_instances(
        llm,
        instances,
        provider,
        mode=mode,
        server_url=server_url,
        max_samples_per_domain=max_samples_per_domain,
        container_runtime=container_runtime,
        container_name=container_name,
        domain=domain,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP Demo (unified image) — run LLM agents against the sel/slot MCP server in the m3_environ container.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Connection modes:
  stdio      Launch MCP server as a local subprocess (default)
  docker     Exec into the unified Docker/Podman container (m3_environ)
  websocket  Connect to an external WebSocket server

Examples:
  # Local subprocess (default)
  python examples/demo_single_docker_image.py --provider openai --model gpt-4o

  # Unified Docker container (see docker/Dockerfile.unified for build + run)
  python examples/demo_single_docker_image.py --mode docker --container-name m3_environ --domain superhero \\
      --provider openai --model gpt-4o --max-samples-per-domain 1

  # WebSocket
  python examples/demo_single_docker_image.py --mode websocket --server-url ws://localhost:8000/mcp
        """,
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["rits", "ollama", "anthropic", "openai", "litellm", "watsonx"],
        default="rits",
        help="LLM provider to use (default: rits)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: provider-specific default)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stdio", "docker", "websocket"],
        default="stdio",
        help="Connection mode (default: stdio)"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="WebSocket server URL (required for --mode websocket), e.g., ws://localhost:8000/mcp"
    )
    parser.add_argument(
        "--max-samples-per-domain",
        type=int,
        default=None,
        help="Maximum number of instances to process (default: all)"
    )

    # Docker mode options
    docker_group = parser.add_argument_group("Docker mode options")
    docker_group.add_argument(
        "--container-runtime",
        type=str,
        choices=["docker", "podman"],
        default=None,
        help="Container runtime (default: auto-detect podman, then docker)"
    )
    docker_group.add_argument(
        "--container-name",
        type=str,
        default="m3_environ",
        help="Name of the running container (default: m3_environ)"
    )
    docker_group.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain for MCP_DOMAIN env var, e.g. 'superhero' (default: MCP_DOMAIN env or 'superhero')"
    )

    args = parser.parse_args()
    try:
        asyncio.run(main(
            provider=args.provider,
            model=args.model,
            mode=args.mode,
            server_url=args.server_url,
            max_samples_per_domain=args.max_samples_per_domain,
            container_runtime=args.container_runtime,
            container_name=args.container_name,
            domain=args.domain,
        ))
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
