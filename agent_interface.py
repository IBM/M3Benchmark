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
    ):
        """
        Initialize the LangGraph ReAct agent.

        Args:
            model: Model name to use
            temperature: Temperature for generation
            api_key: API key (defaults to env var based on provider)
            provider: "anthropic", "openai", or "ollama"
        """
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self.api_key = api_key
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
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._llm

    def _messages_to_langchain(self, messages: List[Message]) -> List[tuple]:
        """Convert Message objects to LangChain format."""
        return [(m.role, m.content) for m in messages]

    async def run(
        self,
        input: Union[str, List[Message]],
        tools: List[StructuredTool],
    ) -> AgentResponse:
        """Run the ReAct agent with given input and tools."""
        from langgraph.prebuilt import create_react_agent
        llm = self._get_llm()
        agent = create_react_agent(llm, tools)
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

        # Track tool calls with their arguments
        tool_call_args = {}  # Map tool_call_id to args

        if result and "messages" in result:
            for msg in result["messages"]:
                msg_class = msg.__class__.__name__

                if msg_class == "HumanMessage":
                    response_messages.append(Message(role="user", content=msg.content))
                elif msg_class == "AIMessage":
                    if msg.content:
                        final_content = msg.content
                    response_messages.append(Message(role="assistant", content=msg.content or ""))

                    # Capture tool call arguments from AIMessage
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tc_id = tc.get("id", "") or tc.get("tool_call_id", "")
                            tool_call_args[tc_id] = {
                                "name": tc.get("name", "unknown"),
                                "args": tc.get("args", {}),
                            }

                elif msg_class == "ToolMessage":
                    tool_call_id = getattr(msg, "tool_call_id", "")
                    tool_info = tool_call_args.get(tool_call_id, {})
                    tool_name = getattr(msg, "name", None) or tool_info.get("name", "unknown")
                    tool_calls.append({
                        "tool_name": tool_name,
                        "arguments": tool_info.get("args", {}),
                        "result": msg.content,
                    })

        return AgentResponse(
            content=final_content,
            tool_calls=tool_calls,
            messages=response_messages,
            metadata={"model": self.model, "provider": self.provider},
        )


def create_agent(
    provider: str = "anthropic",
    model: str | None = None,
    **kwargs,
) -> AgentInterface:
    """
    Factory function to create an agent.

    Args:
        provider: "anthropic", "openai", or "ollama"
        model: Model name (defaults based on provider)
        **kwargs: Additional arguments passed to agent constructor

    Returns:
        AgentInterface implementation
    """
    default_models = {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4.1",
        "ollama": "llama3.1:8b",
    }

    model = model or default_models.get(provider, "claude-3-5-sonnet-20241022")
    return LangGraphReActAgent(
        model=model,
        provider=provider,
        **kwargs,
    )
