

import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
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

# Configuration
# Database path is now constructed by MCP server from MCP_DOMAIN environment variable

@contextlib.asynccontextmanager
async def connect_to_server(mode: str, server_url: str = None):
    """Connect to MCP server using specified mode.

    Args:
        mode: "stdio" (subprocess) or "websocket" (external server)
        server_url: WebSocket URL (required for websocket mode)

    Yields:
        (read_stream, write_stream): Communication streams
    """
    if mode == "stdio":
        # Existing subprocess approach
        # MCP server will read MCP_DOMAIN environment variable to construct database path
        # Pass environment variables to subprocess
        python_exe = sys.executable
        server_params = StdioServerParameters(
            command=python_exe,
            args=["-m", "apis.m3.python_tools.mcp"],
            env=os.environ.copy(),  # Pass all environment variables to subprocess
        )
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
    server_url: str = None,
    max_samples_per_domain: int = None
):
    """Run queries against a single MCP server, switching universes for each instance.

    Args:
        llm: The language model instance
        instances: Dict mapping instance_id -> {query, init_args}
        provider: LLM provider name (e.g. "rits", "ollama", "anthropic", "openai", "litellm", "watsonx")
        mode: Connection mode ("stdio" or "websocket")
        server_url: WebSocket URL (required for websocket mode)
        max_samples_per_domain: Maximum number of instances to process (None = all)
    """
    print(f"\n{'='*60}")
    print(f"Connecting to MCP server (mode: {mode})")
    if mode == "websocket":
        print(f"Server URL: {server_url}")
    else:
        print(f"Starting SINGLE MCP server for {len(instances)} instances")
    print(f"{'='*60}\n")

    async with connect_to_server(mode, server_url) as (read, write):
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
    max_samples_per_domain=None
):
    """Main async entry point with single server for all instances.

    Args:
        provider: LLM provider ("rits", "ollama", "anthropic", "openai", "litellm", "watsonx")
        model: Model name (optional)
        mode: Connection mode - "stdio" (subprocess) or "websocket" (external server)
        server_url: WebSocket URL (required for websocket mode)
        max_samples_per_domain: Maximum number of instances to process (None = all)
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
        max_samples_per_domain=max_samples_per_domain
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Demo with LLM options")
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
        choices=["stdio", "websocket"],
        default="stdio",
        help="Connection mode: stdio (launch subprocess) or websocket (connect to external server)"
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

    args = parser.parse_args()
    try:
        asyncio.run(main(
            provider=args.provider,
            model=args.model,
            mode=args.mode,
            server_url=args.server_url,
            max_samples_per_domain=args.max_samples_per_domain,
        ))
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
