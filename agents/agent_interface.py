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
"""
"""
# Process all domains
python benchmark_runner.py --task_id 2 --run-agent

# Process only hockey domain
python benchmark_runner.py --task_id 2 --run-agent --domain hockey

# Process hockey and address domains
python benchmark_runner.py --task_id 2 --run-agent --domain hockey --domain address

# Combine with other options
python benchmark_runner.py --task_id 2 --run-agent --domain hockey --max-samples-per-domain 5
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Union

from langchain_core.tools import StructuredTool


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
        tools: List[StructuredTool],
    ) -> AgentResponse:
        """
        Run the agent with the given input and tools.

        Args:
            input: Either a single question string or a list of Messages
            tools: List of LangChain StructuredTool objects

        Returns:
            AgentResponse with the final answer and metadata
        """
        pass


class LangGraphReActAgent(AgentInterface):
    """LangGraph ReAct agent implementation."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0,
        api_key: str | None = None,
        provider: str = "ollama",
        project_id: str | None = None,
        space_id: str | None = None,
    ):
        """
        Initialize the LangGraph ReAct agent.

        Args:
            model: Model name to use
            temperature: Temperature for generation
            api_key: API key (defaults to env var based on provider)
            provider: "anthropic", "openai", "ollama", or "watsonx"
            project_id: watsonx.ai project ID (required for watsonx provider)
            space_id: watsonx.ai space ID (optional, alternative to project_id)
        """
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self.api_key = api_key
        self.project_id = project_id
        self.space_id = space_id
        self._llm = None

    def _get_llm(self):
        """Lazily initialize the LLM."""
        if self._llm is not None:
            return self._llm
        
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"),
            )
        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            )
        elif self.provider == "ollama":
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model=self.model,
                temperature=self.temperature,
                num_ctx=65536,
            )
        elif self.provider == "watsonx":
            from langchain_ibm import ChatWatsonx
            
            # Get credentials from environment or parameters
            api_key = self.api_key or os.getenv("WATSONX_APIKEY")
            project_id = self.project_id or os.getenv("WATSONX_PROJECT_ID")
            space_id = self.space_id or os.getenv("WATSONX_SPACE_ID")
            url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
            
            if not api_key:
                raise ValueError("watsonx.ai API key is required. Set WATSONX_APIKEY environment variable or pass api_key parameter.")
            
            if not project_id and not space_id:
                raise ValueError("Either project_id or space_id is required for watsonx.ai. Set WATSONX_PROJECT_ID or WATSONX_SPACE_ID environment variable.")
            
            params = {
                "model_id": self.model,
                "url": url,
                "apikey": api_key,
                "params": {
                    "temperature": self.temperature,
                    "max_new_tokens": 4096,
                }
            }
            
            if project_id:
                params["project_id"] = project_id
            elif space_id:
                params["space_id"] = space_id
            
            self._llm = ChatWatsonx(**params)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._llm

    def _messages_to_langchain(self, messages: List[Message]) -> List[tuple]:
        """Convert Message objects to LangChain format."""
        return [(m.role, m.content) for m in messages]

    def _parse_json_tool_call(self, content: str) -> dict | None:
        """Try to parse a JSON tool call from text output (for models that don't use tool calling API)."""
        import json
        import re

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
        tools: List[StructuredTool],
    ) -> AgentResponse:
        """Run the ReAct agent with given input and tools."""
        from langgraph.prebuilt import create_react_agent
        llm = self._get_llm()
        agent = create_react_agent(llm, tools)

        # Build tool map for manual execution
        tool_map = {t.name: t for t in tools}

        # Convert input to messages format
        if isinstance(input, str):
            messages = [("user", input)]
        else:
            messages = self._messages_to_langchain(input)

        # Run the agent
        result = await agent.ainvoke({"messages": messages})

        # Extract results
        response_messages = []
        tool_calls = []
        final_content = ""
        trajectory = []  # Capture full trajectory

        # Track tool calls with their arguments
        tool_call_args = {}  # Map tool_call_id to args

        if result and "messages" in result:
            for msg in result["messages"]:
                msg_class = msg.__class__.__name__

                # Build trajectory entry for each message
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
                    response_messages.append(Message(role="assistant", content=msg.content or ""))

                    # Capture tool call arguments from AIMessage (proper tool calling)
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

        # FALLBACK: If no tool calls captured but final_content looks like a JSON tool call,
        # manually parse and execute it (for models that output tool calls as text)
        if not tool_calls and final_content:
            parsed_call = self._parse_json_tool_call(final_content)
            if parsed_call:
                tool_name = parsed_call.get("name", "")
                tool_args = parsed_call.get("arguments", {})
                print(f"    [FALLBACK] Detected text tool call: {tool_name}({tool_args})")

                if tool_name in tool_map:
                    try:
                        tool_result = await tool_map[tool_name].ainvoke(tool_args)
                        tool_result_str = str(tool_result)
                        print(f"    [FALLBACK] Tool result: {tool_result_str[:200]}...")
                        tool_calls.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": tool_result_str,
                        })
                        # Update final_content to be the tool result
                        final_content = tool_result_str
                    except Exception as e:
                        print(f"    [FALLBACK] Tool error: {e}")
                        tool_calls.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": f"Error: {e}",
                        })
                else:
                    print(f"    [FALLBACK] Tool '{tool_name}' not found in available tools")

        return AgentResponse(
            content=final_content,
            tool_calls=tool_calls,
            messages=response_messages,
            metadata={"model": self.model, "provider": self.provider},
            trajectory=trajectory,
        )


def create_agent(
    provider: str = "anthropic",
    model: str | None = None,
    **kwargs,
) -> AgentInterface:
    """
    Factory function to create an agent.

    Args:
        provider: "anthropic", "openai", "ollama", or "watsonx"
        model: Model name (defaults based on provider)
        **kwargs: Additional arguments passed to agent constructor
                  For watsonx: project_id or space_id, api_key

    Returns:
        AgentInterface implementation
    """
    default_models = {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4.1",
        "ollama": "llama3.1:8b",
        "watsonx": "openai/gpt-oss-120b",
    }

    model = model or default_models.get(provider, "claude-3-5-sonnet-20241022")
    return LangGraphReActAgent(
        model=model,
        provider=provider,
        **kwargs,
    )
