"""
LangChain ReAct Agent using MCP Server (Local)

Connects to a locally running MCP server scoped to specific domain(s).
The MCP server talks to a local FastAPI server on port 8000.

Usage:
    # Start FastAPI first
    uvicorn app:app --port 8000

    # Run agent scoped to a single domain
    MCP_DOMAINS="hockey" python examples/langchain_agent_local.py

    # Run agent scoped to multiple domains
    MCP_DOMAINS="hockey,superhero" python examples/langchain_agent_local.py

    # Run agent with all domains (no filtering)
    python examples/langchain_agent_local.py
"""
import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ProfileStats:
    """Collects timing data across the run."""

    def __init__(self):
        self.mcp_init_s: float = 0
        self.tool_discovery_s: float = 0
        self.agent_creation_s: float = 0
        self.queries: List[Dict] = []  # {query, total_s, tool_calls, tool_names}
        self.tool_calls: List[Dict] = []  # {tool_name, duration_s, query_index}

    def print_summary(self):
        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)

        print(f"\n  MCP session init:    {self.mcp_init_s:8.3f}s")
        print(f"  Tool discovery:      {self.tool_discovery_s:8.3f}s")
        print(f"  Agent creation:      {self.agent_creation_s:8.3f}s")

        total_query_time = sum(q["total_s"] for q in self.queries)
        print(f"\n  Queries: {len(self.queries)}   Total query time: {total_query_time:.3f}s")
        print(f"  {'#':<4} {'Time':>8}  {'Tools':>5}  Query")
        print(f"  {'─'*4} {'─'*8}  {'─'*5}  {'─'*50}")
        for i, q in enumerate(self.queries):
            status = "OK" if not q.get("error") else "ERR"
            print(f"  {i+1:<4} {q['total_s']:7.3f}s  {q['tool_calls']:5d}  {q['query'][:50]}")
            if q.get("error"):
                print(f"       ERROR: {q['error']}")

        if self.tool_calls:
            print(f"\n  Individual tool calls: {len(self.tool_calls)}")
            total_tool_time = sum(tc["duration_s"] for tc in self.tool_calls)
            print(f"  Total tool execution time: {total_tool_time:.3f}s")
            print(f"  Avg tool call: {total_tool_time / len(self.tool_calls):.3f}s")
            print(f"\n  {'Tool Name':<45} {'Time':>8}  {'Query #':>7}")
            print(f"  {'─'*45} {'─'*8}  {'─'*7}")
            for tc in self.tool_calls:
                print(f"  {tc['tool_name'][:45]:<45} {tc['duration_s']:7.3f}s  {tc['query_index']+1:>7}")

        overhead = total_query_time - sum(tc["duration_s"] for tc in self.tool_calls)
        print(f"\n  LLM reasoning overhead: {overhead:.3f}s "
              f"({overhead / total_query_time * 100:.1f}% of query time)" if total_query_time > 0 else "")

        grand_total = self.mcp_init_s + self.tool_discovery_s + self.agent_creation_s + total_query_time
        print(f"\n  Grand total: {grand_total:.3f}s")
        print("=" * 80)


# Global stats instance — set per run
_stats: Optional[ProfileStats] = None
_current_query_index: int = 0


class MCPToolWrapper:
    """Wrapper to convert MCP tools to LangChain tools"""

    def __init__(self, session: ClientSession, use_openai: bool = False):
        self.session = session
        self.tools_cache = None
        self.use_openai = use_openai

    async def get_tools(self) -> List[StructuredTool]:
        """Fetch tools from MCP server and convert to LangChain tools"""
        if self.tools_cache is None:
            response = await self.session.list_tools()
            mcp_tools = response.tools
            self.tools_cache = [self._create_langchain_tool(t) for t in mcp_tools]
        return self.tools_cache

    def _create_langchain_tool(self, mcp_tool) -> StructuredTool:
        """Convert a single MCP tool to a LangChain StructuredTool"""

        async def tool_func(**kwargs) -> str:
            global _stats, _current_query_index
            t0 = time.perf_counter()
            result = await self.session.call_tool(mcp_tool.name, kwargs)
            duration = time.perf_counter() - t0
            if _stats is not None:
                _stats.tool_calls.append({
                    "tool_name": mcp_tool.name,
                    "duration_s": duration,
                    "query_index": _current_query_index,
                })
            print(f"  [tool] {mcp_tool.name} -> {duration:.3f}s")
            if result.content:
                return result.content[0].text
            return "No result"

        tool_name = mcp_tool.name

        if self.use_openai:
            import re
            tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)
            if len(tool_name) > 64:
                tool_name = tool_name.replace("get_", "").replace("_by_", "_").replace("_with_", "_")
                if len(tool_name) > 64:
                    tool_name = tool_name[:64]

        return StructuredTool(
            name=tool_name,
            description=mcp_tool.description or f"Tool: {mcp_tool.name}",
            func=lambda **kwargs: asyncio.run(tool_func(**kwargs)),
            coroutine=tool_func,
            args_schema=None,
        )


def create_llm():
    """Initialize LLM based on available API keys / env vars.

    Priority: Claude > Ollama > OpenAI
    Returns (llm, use_openai, max_tools)
    """
    skip_claude = os.getenv("SKIP_CLAUDE") or os.getenv("USE_OLLAMA")

    if os.getenv("ANTHROPIC_API_KEY") and not skip_claude:
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        print("Using Claude 3.5 Sonnet (supports 1000+ tools)")
        return llm, False, 1024

    if os.getenv("USE_OLLAMA") or not os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_ollama import ChatOllama as OllamaTool
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            llm = OllamaTool(model=ollama_model, temperature=0, num_ctx=65536)
            print(f"Using Ollama with {ollama_model} (supports 200+ tools with 65K context)")
            return llm, False, 205
        except ImportError:
            print("langchain-ollama not installed. Run: pip install langchain-ollama")
            if not os.getenv("OPENAI_API_KEY"):
                exit(1)
        except Exception as e:
            print(f"Ollama error: {e}")
            print("Make sure Ollama is running and you've pulled a model: ollama pull llama3.1:8b")
            if not os.getenv("OPENAI_API_KEY"):
                exit(1)

    # Fallback: OpenAI
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print("Using OpenAI GPT-4 (limited to 50 tools)")
    return llm, True, 50


async def main():
    """Main function to run the LangChain agent"""
    global _stats, _current_query_index

    _stats = ProfileStats()
    llm, use_openai, max_tools = create_llm()

    # Domain filtering via MCP_DOMAINS env var.
    domains = os.getenv("MCP_DOMAINS", "")
    fastapi_url = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

    env = {"FASTAPI_BASE_URL": fastapi_url}
    if domains:
        env["MCP_DOMAINS"] = domains
        print(f"\nDomains: {domains}")
    else:
        print("\nDomains: ALL (no MCP_DOMAINS set)")

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # --- MCP init ---
            t0 = time.perf_counter()
            await session.initialize()
            _stats.mcp_init_s = time.perf_counter() - t0
            print(f"MCP session initialized in {_stats.mcp_init_s:.3f}s")

            # --- Tool discovery ---
            t0 = time.perf_counter()
            mcp_wrapper = MCPToolWrapper(session, use_openai=use_openai)
            all_tools = await mcp_wrapper.get_tools()
            _stats.tool_discovery_s = time.perf_counter() - t0
            print(f"Loaded {len(all_tools)} tools in {_stats.tool_discovery_s:.3f}s")

            # Limit tools if the model can't handle them all
            if len(all_tools) > max_tools:
                print(f"Limiting to {max_tools} tools (model constraint)")
                tools = all_tools[:max_tools]
            else:
                tools = all_tools

            print("Sample tools:")
            for tool in tools[:10]:
                print(f"  - {tool.name}: {tool.description}")
            if len(tools) > 10:
                print(f"  ... and {len(tools) - 10} more")

            # --- Agent creation ---
            t0 = time.perf_counter()
            agent_executor = create_react_agent(llm, tools)
            _stats.agent_creation_s = time.perf_counter() - t0
            print(f"Agent created in {_stats.agent_creation_s:.3f}s")

            queries = [
                "Who is the most recently born hockey player that is still alive?",
                "List hockey players who are not in the Hall of Fame",
                "Which customer has the highest total order price?",
                "Which nation has the most customers?",
                "What powers does Batman have?",
            ]

            for i, query in enumerate(queries):
                _current_query_index = i
                print(f"\n{'=' * 80}")
                print(f"Query {i+1}/{len(queries)}: {query}")
                print("=" * 80)

                query_record = {"query": query, "total_s": 0, "tool_calls": 0, "error": None}

                t0 = time.perf_counter()
                try:
                    result = await agent_executor.ainvoke({"messages": [("user", query)]})
                    query_record["total_s"] = time.perf_counter() - t0

                    if result and "messages" in result:
                        tool_messages = [
                            m for m in result["messages"]
                            if getattr(m, "__class__", None) and m.__class__.__name__ == "ToolMessage"
                        ]
                        query_record["tool_calls"] = len(tool_messages)

                        if tool_messages:
                            print(f"\n[Tools called: {len(tool_messages)}]")
                        final_message = result["messages"][-1]
                        print(f"\nAnswer: {final_message.content}")
                    else:
                        print(f"\nResult: {result}")
                except Exception as e:
                    query_record["total_s"] = time.perf_counter() - t0
                    query_record["error"] = str(e)
                    print(f"Error: {e}")

                print(f"\n[Query time: {query_record['total_s']:.3f}s]")
                _stats.queries.append(query_record)

            _stats.print_summary()


if __name__ == "__main__":
    has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    use_ollama = os.getenv("USE_OLLAMA") is not None

    if not has_anthropic and not has_openai and not use_ollama:
        print("Error: Set an API key or use Ollama")
        print("\nOption 1 - Ollama (local, free):")
        print("  ollama pull llama3.1:8b")
        print("  export USE_OLLAMA=true")
        print("\nOption 2 - Claude:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        print("\nOption 3 - OpenAI:")
        print("  export OPENAI_API_KEY=your-key-here")
        exit(1)

    asyncio.run(main())
