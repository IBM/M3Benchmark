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
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Union

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

# ALTK imports for SPARC validation
from altk.pre_tool.sparc import SPARCReflectionComponent
from altk.pre_tool.core import SPARCReflectionRunInput, Track
from altk.core.toolkit import AgentPhase, ComponentConfig
from altk.core.llm import get_llm

# Load environment variables from .env file
load_dotenv()


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
    """LangGraph ReAct agent implementation with optional SPARC validation."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0,
        api_key: str | None = None,
        provider: str = "ollama",
        use_sparc: bool = True,
        sparc_track: str = "FAST_TRACK",
    ):
        """
        Initialize the LangGraph ReAct agent.

        Args:
            model: Model name to use
            temperature: Temperature for generation
            api_key: API key (defaults to env var based on provider)
            provider: "anthropic", "openai", "ollama", or "litellm"
            use_sparc: Enable SPARC validation for tool calls (default: True)
            sparc_track: SPARC validation track - "SYNTAX", "FAST_TRACK", or "SLOW_TRACK" (default: "FAST_TRACK")
        """
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self.api_key = api_key
        self.use_sparc = use_sparc
        self.sparc_track = sparc_track
        self._llm = None
        self._sparc = None

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
        elif self.provider == "litellm":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key or os.getenv("LITELLM_API_KEY"),
                base_url=os.getenv("LITELLM_BASE_URL"),
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._llm

    def _get_sparc(self):
        """Lazily initialize SPARC component."""
        if self._sparc is not None:
            return self._sparc
        
        if not self.use_sparc:
            return None
        
        # Initialize SPARC with ValidatingLLMClient
        try:
            # Use OpenAI-compatible provider for SPARC validation
            sparc_model = os.getenv("LLM_CHAT_MODEL_NAME", self.model)
            llm_client_class = get_llm("openai.sync.output_val")
            
            # Determine base_url based on provider
            if self.provider == "litellm":
                base_url = os.getenv("LITELLM_BASE_URL")
                api_key = self.api_key or os.getenv("LITELLM_API_KEY")
            elif self.provider == "openai":
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            else:
                # For other providers, try to use OpenAI as fallback for SPARC
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print(f"Warning: SPARC requires OpenAI API key. Set OPENAI_API_KEY or USE_SPARC=false")
                    return None
            
            llm_client = llm_client_class(model=sparc_model, base_url=base_url, api_key=api_key)
            
            # Map track string to Track enum
            track_map = {
                "SYNTAX": Track.SYNTAX,
                "FAST_TRACK": Track.FAST_TRACK,
                "SLOW_TRACK": Track.SLOW_TRACK,
            }
            track = track_map.get(self.sparc_track, Track.FAST_TRACK)
            
            config = ComponentConfig(llm_client=llm_client)
            self._sparc = SPARCReflectionComponent(config=config)
            print(f"SPARC initialized with track: {self.sparc_track}")
            
        except Exception as e:
            print(f"Warning: Failed to initialize SPARC: {e}")
            print("Continuing without SPARC validation")
            self._sparc = None
        
        return self._sparc

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

    def _convert_tools_to_openai_format(self, tools: List[StructuredTool]) -> List[dict]:
        """Convert LangChain tools to OpenAI function calling format for SPARC."""
        tool_specs = []
        for tool in tools:
            spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Extract parameters from tool's args_schema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.schema()
                if 'properties' in schema:
                    spec["function"]["parameters"]["properties"] = schema['properties']
                if 'required' in schema:
                    spec["function"]["parameters"]["required"] = schema['required']
            
            tool_specs.append(spec)
        
        return tool_specs

    def _convert_tool_calls_to_openai_format(self, tool_calls: List[dict]) -> List[dict]:
        """Convert LangChain tool calls to OpenAI format for SPARC."""
        openai_calls = []
        for tc in tool_calls:
            # Clean up args - remove schema-like keys if present
            args = tc.get("args", {})
            if isinstance(args, dict):
                # Remove type/properties/required if they exist (schema instead of values)
                cleaned_args = {k: v for k, v in args.items()
                               if k not in ['type', 'properties', 'required']}
                
                # Extract content if args are wrapped
                for key, value in list(cleaned_args.items()):
                    if isinstance(value, dict) and 'content' in value:
                        cleaned_args[key] = value['content']
            else:
                cleaned_args = args
            
            openai_call = {
                "type": "function",
                "function": {
                    "name": tc.get("name", "unknown"),
                    "arguments": json.dumps(cleaned_args)
                }
            }
            openai_calls.append(openai_call)
        
        return openai_calls

    def _validate_with_sparc(self, tool_calls: List[dict], tool_specs: List[dict], context: str = "") -> tuple[bool, str]:
        """
        Validate tool calls using SPARC.
        
        Returns:
            (is_valid, error_message) - is_valid is True if approved, False if rejected
        """
        sparc = self._get_sparc()
        if not sparc:
            return True, ""  # No SPARC, approve by default
        
        try:
            # Convert to OpenAI format
            openai_tool_calls = self._convert_tool_calls_to_openai_format(tool_calls)
            
            # Create SPARC input
            messages = [{"role": "user", "content": context}] if context else []
            
            # Map track string to Track enum
            track_map = {
                "SYNTAX": Track.SYNTAX,
                "FAST_TRACK": Track.FAST_TRACK,
                "SLOW_TRACK": Track.SLOW_TRACK,
            }
            track = track_map.get(self.sparc_track, Track.FAST_TRACK)
            
            sparc_input = SPARCReflectionRunInput(
                track=track,
                tool_calls=openai_tool_calls,
                tool_specs=tool_specs,
                messages=messages
            )
            
            # Run SPARC validation
            result = sparc.process(sparc_input, AgentPhase.RUNTIME)
            reflection = result.output.reflection_result
            
            # Check decision
            if reflection.decision == "approve":
                print(f"    [SPARC] ✓ Tool calls approved")
                return True, ""
            else:
                # Extract error messages from issues
                error_msgs = []
                if reflection.issues:
                    for issue in reflection.issues:
                        if issue.explanation:
                            error_msgs.append(issue.explanation)
                
                error_message = "; ".join(error_msgs) if error_msgs else "Tool call validation failed"
                print(f"    [SPARC] ✗ Tool calls rejected: {error_message}")
                return False, error_message
                
        except Exception as e:
            print(f"    [SPARC] Warning: Validation error: {e}")
            return True, ""  # On error, approve by default

    async def run(
        self,
        input: Union[str, List[Message]],
        tools: List[StructuredTool],
    ) -> AgentResponse:
        """Run the ReAct agent with given input and tools, with optional SPARC validation."""
        from langgraph.prebuilt import create_react_agent
        llm = self._get_llm()
        
        # Convert tool specs to OpenAI format for SPARC (if enabled)
        tool_specs = self._convert_tools_to_openai_format(tools) if self.use_sparc else []
        
        agent = create_react_agent(llm, tools)

        # Build tool map for manual execution
        tool_map = {t.name: t for t in tools}

        # Convert input to messages format
        if isinstance(input, str):
            messages = [("user", input)]
            context = input
        else:
            messages = self._messages_to_langchain(input)
            context = " ".join([m[1] for m in messages if m[0] == "user"])

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
                        
                        # Validate with SPARC if enabled
                        if self.use_sparc and msg.tool_calls:
                            is_valid, error_msg = self._validate_with_sparc(
                                msg.tool_calls,
                                tool_specs,
                                context
                            )
                            trajectory_entry["sparc_validation"] = {
                                "approved": is_valid,
                                "error": error_msg if not is_valid else None
                            }
                            
                            # If rejected, add error to final content
                            if not is_valid:
                                final_content = f"Tool call validation failed: {error_msg}"
                        
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
    use_sparc: bool = True,
    sparc_track: str = "FAST_TRACK",
    **kwargs,
) -> AgentInterface:
    """
    Factory function to create an agent.

    Args:
        provider: "anthropic", "openai", "ollama", or "litellm"
        model: Model name (defaults based on provider)
        use_sparc: Enable SPARC validation for tool calls (default: True)
        sparc_track: SPARC validation track - "SYNTAX", "FAST_TRACK", or "SLOW_TRACK" (default: "FAST_TRACK")
        **kwargs: Additional arguments passed to agent constructor

    Returns:
        AgentInterface implementation
    """
    default_models = {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4.1",
        "ollama": "llama3.1:8b",
        "litellm": "Azure/gpt-5-2025-08-07",
    }

    model = model or default_models.get(provider, "claude-3-5-sonnet-20241022")
    return LangGraphReActAgent(
        model=model,
        provider=provider,
        use_sparc=use_sparc,
        sparc_track=sparc_track,
        **kwargs,
    )
