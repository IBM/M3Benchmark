"""
Agent Interface and Implementations

Provides a clean abstraction for agents that can answer questions using tools.

Usage:
    from agent_interface import LangGraphReActAgent, Message

    agent = LangGraphReActAgent(model="claude-3-5-sonnet-20241022")

    # Single question
    response = await agent.run("What is 2+2?", tools=my_tools)

    # Conversation
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="What tools do you have?"),
    ]
    response = await agent.run(messages, tools=my_tools)

Benchmark runner examples:
    python benchmark_runner.py --capability_id 2 --run-agent
    python benchmark_runner.py --capability_id 2 --run-agent --domain hockey
    python benchmark_runner.py --capability_id 2 --run-agent --domain hockey --domain address
    python benchmark_runner.py --capability_id 2 --run-agent --domain hockey --max-samples-per-domain 5
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent, ToolNode

from agents.components.result_handle_manager import ResultHandleManager
from agents.components.tool_shortlister import ToolShortlister
from agents.llm import create_llm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class AgentResponse:
    """Response from an agent invocation."""
    content: str
    tool_calls: List[dict] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    trajectory: List[dict] = field(default_factory=list)  # Full agent trajectory


class AgentInterface(ABC):
    """Abstract interface for agents that answer questions using tools."""

    @abstractmethod
    async def run(
        self,
        input: Union[str, List[Message]],
    ) -> AgentResponse:
        """
        Run the agent with the given input.

        Args:
            input: Either a single question string or a list of Messages

        Returns:
            AgentResponse with the final answer and metadata
        """
        pass


class LangGraphReActAgent(AgentInterface):
    """LangGraph ReAct agent with optional handle-based result management and tool shortlisting."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[StructuredTool] | None = None,
        top_k_tools: int = 0,
        use_handle_manager: bool = False,
        initial_data_handle: str | None = None,
        max_iterations: int | None = None,
        **kwargs,
    ):
        """
        Initialize the LangGraph ReAct agent.

        Args:
            llm: Already-constructed LangChain chat model instance
            tools: List of LangChain StructuredTool objects to use
            top_k_tools: If > 0, shortlist to this many tools per query via
                         semantic similarity. 0 disables shortlisting.
            use_handle_manager: If True, enables ResultHandleManager for
                                 handle-based tool result management. Tool
                                 results are stored with handles and the LLM
                                 receives compact references instead of full data.
            initial_data_handle: Handle string for pre-seeded initial dataset.
                                  Only meaningful when use_handle_manager=True.
            max_iterations: Maximum number of agent loop iterations. Maps to
                            LangGraph's recursion_limit (×2 + 1). None uses
                            LangGraph defaults.
        """
        self._llm = llm
        self._tools = tools or []
        self._max_iterations = max_iterations

        logger.debug(
            "Initializing LangGraphReActAgent: llm=%s, tools=%d, top_k_tools=%d, "
            "use_handle_manager=%s, initial_data_handle=%s, max_iterations=%s",
            type(llm).__name__,
            len(self._tools),
            top_k_tools,
            use_handle_manager,
            initial_data_handle,
            max_iterations,
        )

        # Handle manager (optional)
        self.handle_manager = None
        self._initial_data_handle: str | None = initial_data_handle
        if use_handle_manager:
            self.handle_manager = ResultHandleManager()
            logger.debug("ResultHandleManager enabled")

        # Tool shortlister (optional — only shortlist when limit < available tools)
        self._shortlister = None
        if top_k_tools > 0 and top_k_tools < len(self._tools):
            self._shortlister = ToolShortlister(top_k=top_k_tools)
            self._shortlister.encode_tools(self._tools)
            logger.debug("ToolShortlister enabled with top_k=%d over %d tools", top_k_tools, len(self._tools))
        elif top_k_tools > 0:
            logger.debug("top_k_tools=%d >= tool count=%d; shortlisting disabled", top_k_tools, len(self._tools))

        # Build default agent (used when shortlisting is disabled)
        logger.debug("Building default agent with %d tools", len(self._tools))
        self._agent = self._build_agent(self._tools)

    def _build_agent(self, tools):
        """Build a LangGraph ReAct agent for the given tool list."""
        tool_names = [t.name for t in tools]
        logger.debug("_build_agent: building agent with %d tools: %s", len(tools), tool_names)
        if self.handle_manager is not None:
            logger.debug("_build_agent: using handle-wrapped ToolNode")
            tool_node = ToolNode(tools, awrap_tool_call=self._make_handle_wrapper(tools))
            return create_react_agent(self._llm, tool_node)
        return create_react_agent(self._llm, tools)

    def _make_handle_wrapper(self, tools):
        """Return an async awrap_tool_call callable that integrates ResultHandleManager.

        The wrapper resolves handle references in tool args before invocation,
        then stores results and returns compact references to the LLM.
        """
        assert self.handle_manager is not None
        handle_manager = self.handle_manager
        tool_map = {t.name: t for t in tools}

        async def awrap_tool_call(request, execute):
            tool_name = request.tool_call["name"]
            tool_args = dict(request.tool_call["args"])
            tool_call_id = request.tool_call["id"]

            logger.debug("awrap_tool_call: tool=%s id=%s args=%s", tool_name, tool_call_id, tool_args)

            # Unwrap kwargs if needed (fixes LLM schema interpretation issue)
            if len(tool_args) == 1 and "kwargs" in tool_args:
                logger.debug("awrap_tool_call: unwrapping kwargs for %s", tool_name)
                tool_args = tool_args["kwargs"]

            # Resolve any handle references in arguments
            resolved_args = handle_manager.resolve_args(tool_args, resolve_data=True)
            logger.debug("awrap_tool_call: resolved args keys for %s: %s", tool_name, list(resolved_args.keys()))

            # Find and invoke tool directly with resolved args
            tool = tool_map.get(tool_name)
            if not tool:
                logger.warning("awrap_tool_call: tool not found: %s", tool_name)
                return ToolMessage(
                    content=json.dumps({"error": f"Tool {tool_name} not found"}),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )

            try:
                logger.debug("awrap_tool_call: invoking tool %s", tool_name)
                raw_result = await tool.ainvoke(resolved_args)
                logger.debug("awrap_tool_call: raw result type=%s length=%s", type(raw_result).__name__, len(str(raw_result)))
            except Exception as e:
                logger.warning("awrap_tool_call: tool %s raised exception: %s", tool_name, e)
                return ToolMessage(
                    content=json.dumps({"error": str(e)}),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )

            # Parse raw result string
            try:
                parsed_result = json.loads(raw_result)
            except (json.JSONDecodeError, ValueError):
                parsed_result = raw_result

            # Unwrap MCP TextContent format: {"type": ..., "text": ...}
            if (
                isinstance(parsed_result, dict)
                and "type" in parsed_result
                and "text" in parsed_result
            ):
                logger.debug("awrap_tool_call: unwrapping MCP TextContent for %s", tool_name)
                parsed_result = parsed_result["text"]
                if isinstance(parsed_result, str):
                    try:
                        parsed_result = json.loads(parsed_result)
                    except (json.JSONDecodeError, ValueError):
                        pass

            # Return errors directly — don't store in handle manager
            if isinstance(parsed_result, dict) and "error" in parsed_result:
                logger.warning("awrap_tool_call: tool %s returned error: %s", tool_name, parsed_result["error"])
                return ToolMessage(
                    content=json.dumps(parsed_result),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            if isinstance(parsed_result, str) and "error" in parsed_result.lower():
                logger.warning("awrap_tool_call: tool %s returned error string: %s", tool_name, parsed_result[:200])
                return ToolMessage(
                    content=json.dumps({"error": parsed_result}),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )

            # Store result and return compact handle reference
            handle = handle_manager.store_result(tool_name, parsed_result)
            reference = handle_manager.create_reference(handle, parsed_result)
            logger.debug("awrap_tool_call: stored result for %s as handle=%s", tool_name, handle)
            return ToolMessage(
                content=reference,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        return awrap_tool_call

    def _build_system_message(self) -> str:
        """Build system message explaining handle system to the LLM."""
        assert self.handle_manager is not None
        logger.debug("_build_system_message: building system message for handle=%s", self._initial_data_handle)
        initial_data_info = ""
        try:
            if self._initial_data_handle is None:
                raise ValueError("No initial data handle set")
            initial_data = self.handle_manager.get_result(self._initial_data_handle)
            initial_data_reference = self.handle_manager.create_reference(
                self._initial_data_handle,
                initial_data
            )
            ref_dict = json.loads(initial_data_reference)
            if ref_dict.get("type") == "data_table":
                key_names = ref_dict.get("key_names", [])
                preview = ref_dict.get("preview", {})
                initial_data_info = f"""
- Available columns (key_names): {key_names}
- Preview of first few rows: {json.dumps(preview, indent=2)}"""
        except (ValueError, KeyError, json.JSONDecodeError):
            pass

        return f"""
You are a helpful data analysis assistant. Use the available tools and data to answer queries.
Think step by step and provide reasoning for the action that you are planning to take.

IMPORTANT - Tool Result Handles:
- Tool results are returned as handles/references, not full data
- Data tables have: {{"handle": "filtered_data_1", "type": "data_table", "key_names": [...], "preview": {{...}}}}
- Remember that the "preview" is only the first few values of the full data_table.
- To use a result in another tool, pass the handle string as the 'data' argument
- Example: After filter_data returns {{"handle": "filtered_superhero_1", ...}}
  Call: sort_data(data="filtered_superhero_1", key_name="age", ...)
- Scalar results include both handle and value: {{"handle": "count_1", "type": "scalar", "value": 42}}
- Always use handles to reference previous results in tool chains
- Make at most a single tool call per iteration. Once you have the information necessary to answer the query, return the answer with no additional tool calls.
- If you receive an error from the tool, reason over why your previous action resulted in that error and make the appropriate fix.

INITIAL DATA:
- The initial dataset for this task is available as handle: "{self._initial_data_handle}"{initial_data_info}
- Start by using this handle in your first tool call
        """

    def _messages_to_langchain(self, messages: List[Message]) -> list:
        """Convert Message objects to LangChain message objects."""
        logger.debug("_messages_to_langchain: converting %d messages", len(messages))
        lc_messages = []
        for m in messages:
            if m.role == "user":
                lc_messages.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                lc_messages.append(AIMessage(content=m.content))
            elif m.role == "system":
                lc_messages.append(SystemMessage(content=m.content))
            elif m.role == "tool_call":
                lc_messages.append(ToolMessage(content=m.content))                
        return lc_messages

    def _parse_json_tool_call(self, content: str) -> dict | None:
        """Try to parse a JSON tool call from text output (for models that don't use tool calling API)."""
        if not content:
            return None

        # Try to extract JSON from markdown code blocks or raw JSON
        patterns = [
            r'```json\s*(.*?)\s*```',  # ```json ... ```
            r'```\s*(.*?)\s*```',       # ``` ... ```
            r'(\{[^{}]*"name"[^{}]*"arguments"[^{}]*\})',  # raw JSON with name/arguments
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL) 
            if match:
                try:
                    data = json.loads(match.group(1))
                    if "name" in data and "arguments" in data:
                        return data
                except json.JSONDecodeError:
                    continue

        # Try parsing the whole content as JSON
        try:
            data = json.loads(content.strip())
            if "name" in data and "arguments" in data:
                return data
        except json.JSONDecodeError:
            pass

        return None

    async def run(
        self,
        input: Union[str, List[Message]],
    ) -> AgentResponse:
        """Run the ReAct agent with given input."""
        query_preview = input if isinstance(input, str) else next(
            (m.content for m in reversed(input) if m.role == "user"), ""
        )
        logger.debug("run: starting agent run, query_preview=%.200s", query_preview)

        # Determine active tools (shortlisted or full set)
        if self._shortlister:
            if isinstance(input, str):
                query = input
            else:
                query = next(
                    (m.content for m in reversed(input) if m.role == "user"),
                    "",
                )
            active_tools = self._shortlister.shortlist(query, self._tools)
            logger.debug("run: shortlisted to %d tools: %s", len(active_tools), [t.name for t in active_tools])
            agent = self._build_agent(active_tools)
        else:
            active_tools = self._tools
            agent = self._agent
            logger.debug("run: using all %d tools (shortlisting disabled)", len(active_tools))

        # Build tool map for fallback manual execution
        tool_map = {t.name: t for t in active_tools}

        # Convert input to LangChain messages
        if isinstance(input, str):
            lc_messages = [HumanMessage(content=input)]
        else:
            lc_messages = self._messages_to_langchain(input)

        # Inject system message when handle manager is active
        if self.handle_manager is not None:
            has_system = any(m.__class__.__name__ == "SystemMessage" for m in lc_messages)
            if not has_system:
                logger.debug("run: injecting handle-system system message")
                lc_messages = [SystemMessage(content=self._build_system_message())] + lc_messages

        # Build invocation config
        config: RunnableConfig | None = None
        if self._max_iterations is not None:
            config = RunnableConfig(recursion_limit=self._max_iterations * 2 + 1)
            logger.debug("run: recursion_limit=%d (max_iterations=%d)", self._max_iterations * 2 + 1, self._max_iterations)

        # Run the agent
        logger.debug("run: invoking LangGraph agent with %d input messages", len(lc_messages))
        result = await agent.ainvoke({"messages": lc_messages}, config=config)
        logger.debug("run: agent invocation complete, result message count=%d", len(result.get("messages", [])))
        # Extract results
        response_messages = []
        tool_calls = []
        final_content = ""
        trajectory = []

        tool_call_args = {}  # Map tool_call_id → {name, args}

        if result and "messages" in result:
            for msg in result["messages"]:
                msg_class = msg.__class__.__name__

                trajectory_entry = {
                    "type": msg_class,
                    "content": getattr(msg, "content", ""),
                }

                if msg_class == "HumanMessage":
                    response_messages.append(Message(role="user", content=msg.content))
                    trajectory.append(trajectory_entry)

                elif msg_class == "AIMessage":
                    if msg.content:
                        final_content = msg.content
                    response_messages.append(Message(
                        role="assistant",
                        content=msg.content or msg.additional_kwargs.get("reasoning")
                        ))

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        if msg.additional_kwargs.get("reasoning"):
                            trajectory_entry["reasoning"] = msg.additional_kwargs.get("reasoning")
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
                    tool_name = getattr(msg, "name", None) or tool_info.get("name", "unknown")
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

        logger.debug(
            "run: extracted %d tool calls, final_content length=%d",
            len(tool_calls),
            len(final_content),
        )

        # FALLBACK: If no tool calls captured but final_content looks like a JSON tool call,
        # manually parse and execute it (for models that output tool calls as text)
        if not tool_calls and final_content and self.handle_manager is None:
            parsed_call = self._parse_json_tool_call(final_content)
            if parsed_call:
                tool_name = parsed_call.get("name", "")
                tool_args = parsed_call.get("arguments", {})
                logger.debug("run: [FALLBACK] detected text tool call: %s(%s)", tool_name, tool_args)

                if tool_name in tool_map:
                    try:
                        tool_result = await tool_map[tool_name].ainvoke(tool_args)
                        tool_result_str = str(tool_result)
                        logger.debug("run: [FALLBACK] tool result: %.200s", tool_result_str)
                        tool_calls.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": tool_result_str,
                        })
                        final_content = tool_result_str
                    except Exception as e:
                        logger.warning("run: [FALLBACK] tool error for %s: %s", tool_name, e)
                        tool_calls.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": f"Error: {e}",
                        })
                else:
                    logger.warning("run: [FALLBACK] tool '%s' not found in available tools", tool_name)

        logger.debug("run: returning AgentResponse with %d tool calls", len(tool_calls))
        return AgentResponse(
            content=final_content,
            tool_calls=tool_calls,
            messages=response_messages,
            metadata={},
            trajectory=trajectory,
        )

    def restart(self):
        """Clear handle manager state between queries. No-op if handle management is disabled."""
        if self.handle_manager is not None:
            logger.debug("restart: clearing handle manager state")
            self.handle_manager.clear()
        else:
            logger.debug("restart: no-op (handle manager disabled)")


def create_agent(
    provider: str = "anthropic",
    model: str | None = None,
    tools: List[StructuredTool] | None = None,
    **kwargs,
) -> AgentInterface:
    """
    Factory function to create an agent.

    Args:
        provider: "anthropic", "openai", "ollama", "watsonx", or "rits"
        model: Model name (defaults based on provider)
        **kwargs: Additional arguments forwarded to create_llm()
                  e.g. api_key, project_id, space_id, ollama_base_url

    Returns:
        AgentInterface implementation
    """
    default_models = {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4.1",
        "ollama": "llama3.1:8b",
        "watsonx": "openai/gpt-oss-120b",
        "rits": "llama-3-3-70b-instruct",
    }

    resolved_model = model or default_models.get(provider, "")
    llm = create_llm(provider=provider, model=resolved_model, **kwargs)
    return LangGraphReActAgent(llm=llm, tools=tools)
