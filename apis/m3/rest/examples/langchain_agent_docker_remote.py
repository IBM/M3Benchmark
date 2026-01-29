"""
LangChain ReAct Agent using MCP Server (Docker Remote)
This example connects to an MCP server running inside a Docker container
The agent runs locally with Ollama, connecting to Dockerized MCP+FastAPI
"""
import asyncio
import os
from typing import Any, Dict, List

from langchain_core.tools import StructuredTool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolWrapper:
    """Wrapper to convert MCP tools to LangChain tools"""

    def __init__(self, session: ClientSession, use_openai: bool = False):
        self.session = session
        self.tools_cache = None
        self.use_openai = use_openai

    async def get_tools(self) -> List[StructuredTool]:
        """Fetch tools from MCP server and convert to LangChain tools"""
        if self.tools_cache is None:
            # List tools from MCP server
            response = await self.session.list_tools()
            mcp_tools = response.tools

            langchain_tools = []
            for mcp_tool in mcp_tools:
                # Create a LangChain tool for each MCP tool
                tool = self._create_langchain_tool(mcp_tool)
                langchain_tools.append(tool)

            self.tools_cache = langchain_tools

        return self.tools_cache

    def _create_langchain_tool(self, mcp_tool) -> StructuredTool:
        """Convert a single MCP tool to a LangChain StructuredTool"""

        async def tool_func(**kwargs) -> str:
            """Execute the MCP tool"""
            result = await self.session.call_tool(mcp_tool.name, kwargs)
            # Extract text content from result
            if result.content:
                return result.content[0].text
            return "No result"

        tool_name = mcp_tool.name

        # Apply OpenAI-specific restrictions only when using OpenAI
        if self.use_openai:
            # OpenAI requires tool names to match ^[a-zA-Z0-9_-]+$ and be max 64 chars
            import re
            tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)

            # Shorten if needed
            if len(tool_name) > 64:
                # Try to shorten by removing common prefixes
                tool_name = tool_name.replace("get_", "").replace("_by_", "_").replace("_with_", "_")
                # If still too long, truncate
                if len(tool_name) > 64:
                    tool_name = tool_name[:64]

        return StructuredTool(
            name=tool_name,
            description=mcp_tool.description or f"Tool: {mcp_tool.name}",
            func=lambda **kwargs: asyncio.run(tool_func(**kwargs)),
            coroutine=tool_func,
            args_schema=None,  # We'll let LangChain infer from the function signature
        )


async def main():
    """Main function to run the LangChain agent"""

    # Initialize LLM - Priority: USE_OLLAMA env var > Claude > OpenAI
    use_openai = False
    max_tools = 205  # Default for Ollama and Claude

    # Check if we should use Ollama (prioritize USE_OLLAMA env var)
    skip_claude = os.getenv("SKIP_CLAUDE") or os.getenv("USE_OLLAMA")

    if os.getenv("USE_OLLAMA") or (not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY")):
        # Option 1: Ollama with tool-calling models (local, free)
        try:
            from langchain_ollama import ChatOllama as OllamaTool

            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            print(f"Using Ollama with {ollama_model} (local, FREE, supports 200+ tools)")

            llm = OllamaTool(
                model=ollama_model,
                temperature=0,
                num_ctx=65536,  # Large context window for 200+ tools
            )
            max_tools = 205
            print(f"✅ Ollama configured with 65K context for ALL {max_tools} tools")

        except ImportError:
            print("❌ langchain-ollama not installed")
            print("Install with: pip install langchain-ollama")
            print("Then run: ollama pull llama3.1:8b")
            exit(1)
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            print("Make sure Ollama is running: ollama serve")
            print("And model is available: ollama pull llama3.1:8b")
            exit(1)

    elif os.getenv("ANTHROPIC_API_KEY") and not skip_claude:
        # Option 2: Claude (best for large tool sets)
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        max_tools = 1024
        print("Using Claude 3.5 Sonnet (supports 1000+ tools)")

    else:
        # Option 3: OpenAI (limited by context)
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        use_openai = True
        max_tools = 50
        print("Using OpenAI GPT-4 (limited to 50 tools)")

    # Connect to MCP server running in Docker/Podman via exec
    container_name = os.getenv("MCP_CONTAINER_NAME", "fastapi-mcp-server")
    container_runtime = os.getenv("CONTAINER_RUNTIME", "docker")  # docker or podman

    print(f"\n🐳 Connecting to MCP server in {container_runtime} container: {container_name}")
    print(f"   Architecture: Agent (local) → MCP Server ({container_runtime}) → FastAPI ({container_runtime})")

    server_params = StdioServerParameters(
        command=container_runtime,
        args=[
            "exec",
            "-i",
            container_name,
            "python",
            "mcp_server.py"
        ],
        env=None  # The container already has FASTAPI_BASE_URL set
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()

                # Get tools from MCP server
                mcp_wrapper = MCPToolWrapper(session, use_openai=use_openai)
                all_tools = await mcp_wrapper.get_tools()

                print(f"✅ Loaded {len(all_tools)} tools from MCP server in Docker")

                # Filter tools based on model capabilities
                if len(all_tools) > max_tools:
                    print(f"⚠️  Filtering to top {max_tools} most relevant tools...")
                    priority_keywords = ['hockey', 'player', 'retails', 'customer', 'nation', 'superhero', 'power', 'hall_of_fame', 'alive', 'born']

                    # Sort tools: priority keywords first, then others
                    priority_tools = [t for t in all_tools if any(kw in t.name.lower() for kw in priority_keywords)]
                    other_tools = [t for t in all_tools if not any(kw in t.name.lower() for kw in priority_keywords)]

                    tools = (priority_tools + other_tools)[:max_tools]
                    print(f"   Selected {len(tools)} tools ({len(priority_tools)} priority, {len(tools) - len(priority_tools)} others)")
                else:
                    tools = all_tools
                    print(f"✅ Using ALL {len(tools)} tools!")

                print("\n📋 Sample tools available:")
                for tool in tools[:5]:  # Show first 5 tools
                    print(tool)
                    print(f"   • {tool.name}: {tool.description[:60]}...")
                if len(tools) > 5:
                    print(f"   ... and {len(tools) - 5} more tools\n")

                print("Tool!!!")
                # Create ReAct agent using LangGraph
                agent_executor = create_react_agent(llm, tools)

                # Example queries - using endpoints that exist and work well
                # These queries match actual API endpoints with no/simple parameters
                queries = [
                    # Hockey - no params needed (get_most_recently_born_alive_player)
                    "Who is the most recently born hockey player that is still alive?",

                    # Hockey - no params needed (get_players_not_in_hall_of_fame)
                    "List hockey players who are not in the Hall of Fame",

                    # Retails - no params needed (get_customer_name_by_max_totalprice)
                    "Which customer has the highest total order price?",

                    # Retails - no params needed (get_nation_with_most_customers)
                    "Which nation has the most customers?",

                    # Superhero - simple param (get_power_names_by_superhero)
                    "What powers does Batman have?",
                ]

                for query in queries:
                    print(f"\n{'=' * 80}")
                    print(f"🔍 Query: {query}")
                    print('=' * 80)

                    try:
                        result = await agent_executor.ainvoke({"messages": [("user", query)]})
                        # Extract the final answer from the messages
                        if result and "messages" in result:
                            # Check if any tools were called
                            tool_messages = [m for m in result["messages"] if hasattr(m, '__class__') and m.__class__.__name__ == 'ToolMessage']
                            if tool_messages:
                                print(f"\n✅ Tools called: {(tool_messages)}")

                            final_message = result["messages"][-1]
                            print(f"\n💡 Final Answer:\n{final_message.content}")
                        else:
                            print(f"\n💡 Result: {result}")
                    except Exception as e:
                        print(f"\n❌ Error: {e}")

                    print()

    except Exception as e:
        print(f"\n❌ Failed to connect to Docker container: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Make sure the Docker container is running:")
        print(f"   docker ps | grep {container_name}")
        print(f"2. Start the container if needed:")
        print(f"   docker-compose up -d")
        print(f"3. Check container logs:")
        print(f"   docker logs {container_name}")
        exit(1)


if __name__ == "__main__":
    # Check for Ollama, API keys
    has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    use_ollama = os.getenv("USE_OLLAMA") is not None

    if not has_anthropic and not has_openai and not use_ollama:
        print("❌ Error: Set an API key or use Ollama")
        print("\n🎉 RECOMMENDED - Ollama (local, FREE, handles ALL 205 tools!):")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull model: ollama pull llama3.1:8b")
        print("  3. Set variable: export USE_OLLAMA=true")
        print("  4. Run this script!")
        print("\n📌 Option 2 - Claude (best quality for 200+ tools):")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        print("\n📌 Option 3 - OpenAI (limited to 50 tools):")
        print("  export OPENAI_API_KEY=your-key-here")
        exit(1)

    print("=" * 80)
    print("🚀 LangChain Agent with Dockerized MCP Server")
    print("=" * 80)

    # Run the agent
    asyncio.run(main())
