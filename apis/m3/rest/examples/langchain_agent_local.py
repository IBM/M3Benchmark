"""
LangChain ReAct Agent using MCP Server (Local)
This example connects to a locally running MCP server and FastAPI server
"""
import asyncio
import os
from typing import Any, Dict, List

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
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

    # Initialize LLM - Priority: Claude > Ollama > OpenAI
    use_openai = False
    max_tools = 200  # Default for most models

    # Skip Claude if explicitly disabled or if using Ollama
    skip_claude = os.getenv("SKIP_CLAUDE") or os.getenv("USE_OLLAMA")

    if os.getenv("ANTHROPIC_API_KEY") and not skip_claude:
        # Option 1: Claude (best for large tool sets)
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        max_tools = 1024
        print("Using Claude 3.5 Sonnet (supports 1000+ tools)")

    elif os.getenv("USE_OLLAMA") or not os.getenv("OPENAI_API_KEY"):
        # Option 2: Ollama with tool-calling models (local, free)
        # BREAKTHROUGH: llama3.1:8b can handle ALL 205 tools with 65K context!
        # Best models for tool calling (in order of recommendation):
        # - llama3.1:8b (RECOMMENDED, 200+ tools with num_ctx=65536)
        # - llama3.1:70b (best quality, 200+ tools, requires 40GB+ RAM)
        # - llama3.2:3b (smaller, ~50 tools max)
        # - mistral:7b (alternative, ~50 tools max)
        try:
            from langchain_ollama import ChatOllama as OllamaTool

            # Try to use Ollama with a tool-calling model
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            print(f"Attempting to use Ollama with {ollama_model}...")

            llm = OllamaTool(
                model=ollama_model,
                temperature=0,
                num_ctx=65536,  # Large context window for 200+ tools
            )
            max_tools = 205  # Llama 3.1 can handle all tools with large context
            print(f"Using Ollama with {ollama_model} (local, supports 200+ tools with 65K context)")
            print("Note: If this fails, run: ollama pull llama3.1:8b")

        except ImportError:
            print("Installing langchain-ollama for tool calling support...")
            print("Run: pip install langchain-ollama")

            if os.getenv("OPENAI_API_KEY"):
                print("Falling back to OpenAI...")
                llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                use_openai = True
                max_tools = 50
                print("Using OpenAI GPT-4 (limited to 50 tools)")
            else:
                print("\nPlease either:")
                print("  1. pip install langchain-ollama && ollama pull llama3.1:8b")
                print("  2. Set OPENAI_API_KEY")
                print("  3. Add Anthropic credits and unset USE_OLLAMA")
                exit(1)
        except Exception as e:
            print(f"Ollama error: {e}")
            print("Make sure Ollama is running and you've pulled a tool-calling model:")
            print("  ollama pull llama3.1:8b")

            if os.getenv("OPENAI_API_KEY"):
                print("\nFalling back to OpenAI...")
                llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                use_openai = True
                max_tools = 50
                print("Using OpenAI GPT-4 (limited to 50 tools)")
            else:
                exit(1)

    else:
        # Option 3: OpenAI (limited by context and API restrictions)
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        use_openai = True
        max_tools = 50
        print("Using OpenAI GPT-4 (limited to 50 tools)")

    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env={"FASTAPI_BASE_URL": "http://localhost:8003"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # Get tools from MCP server
            mcp_wrapper = MCPToolWrapper(session, use_openai=use_openai)
            all_tools = await mcp_wrapper.get_tools()

            print(f"Loaded {len(all_tools)} tools from MCP server")

            # Filter tools based on model capabilities
            # For testing, use fewer tools for better performance
            test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
            if test_mode:
                max_tools = 20  # Use fewer tools in test mode
                print(f"TEST MODE: Using only {max_tools} tools")

            if len(all_tools) > max_tools:
                print(f"Filtering to top {max_tools} most relevant tools...")
                priority_keywords = ['hockey', 'player', 'retails', 'customer', 'nation', 'superhero', 'power', 'hall_of_fame', 'alive', 'born']

                # Sort tools: priority keywords first, then others
                priority_tools = [t for t in all_tools if any(kw in t.name.lower() for kw in priority_keywords)]
                other_tools = [t for t in all_tools if not any(kw in t.name.lower() for kw in priority_keywords)]

                tools = (priority_tools + other_tools)[:max_tools]
                print(f"Selected {len(tools)} tools ({len(priority_tools)} priority, {len(tools) - len(priority_tools)} others)")
            else:
                tools = all_tools

            print("Available tools:")
            for tool in tools[:10]:  # Show first 10 tools
                print(f"  - {tool.name}: {tool.description}")
            if len(tools) > 10:
                print(f"  ... and {len(tools) - 10} more tools")

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
                print(f"Query: {query}")
                print('=' * 80)

                try:
                    result = await agent_executor.ainvoke({"messages": [("user", query)]})
                    # Extract the final answer from the messages
                    if result and "messages" in result:
                        # Check if any tools were called
                        tool_messages = [m for m in result["messages"] if hasattr(m, '__class__') and m.__class__.__name__ == 'ToolMessage']
                        if tool_messages:
                            print(f"\n[Tools called: {len(tool_messages)}]")

                        final_message = result["messages"][-1]
                        print(f"\nFinal Answer: {final_message.content}")
                    else:
                        print(f"\nResult: {result}")
                except Exception as e:
                    print(f"Error: {e}")

                print()


if __name__ == "__main__":
    # Check for API keys or Ollama
    has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    use_ollama = os.getenv("USE_OLLAMA") is not None

    if not has_anthropic and not has_openai and not use_ollama:
        print("Error: Set an API key or use Ollama")
        print("\n🎉 RECOMMENDED - Ollama (local, FREE, handles ALL 205 tools!):")
        print("  Install: https://ollama.ai")
        print("  Run: ollama pull llama3.1:8b")
        print("  Then: export USE_OLLAMA=true")
        print("\nOption 2 - Claude (best quality for 200+ tools):")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        print("\nOption 3 - OpenAI (limited to 50 tools):")
        print("  export OPENAI_API_KEY=your-key-here")
        exit(1)

    # Run the agent
    asyncio.run(main())
