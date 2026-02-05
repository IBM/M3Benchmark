import os

from langchain_core.language_models.chat_models import BaseChatModel
import asyncio

from langchain.tools import BaseTool
from typing import List, Optional

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration
import httpx
from typing import Any, Sequence, Dict, Callable, Union

from pydantic import Field

class MCPToolWrapper(BaseTool):
    """Wraps an MCPTool to make it compatible with LangChain ChatModels."""

    name: str
    description: str
    _tool: Any = None
    _fields: Dict[str, Any] = {}

    def __init__(self, tool: Any, **kwargs):
        mcp_schema = getattr(tool, "args_schema", {})
        properties = mcp_schema.get("properties", {})
        
        # Initialize the tool
        # set args_schema to None because args are handled manually
        super().__init__(
            name=tool.name,
            description=tool.description,
            args_schema=None, 
            **kwargs
        )
        self._tool = tool
        self._fields = properties

    @property
    def args(self) -> dict:
        """
        Returns the parameters (data, key_name, etc.) directly.
        Overrides defaulting to empty pydantics base.
        Note: Removing this will lead to empty arguments for the tools.
        """
        return self._fields

    def _run(self, **kwargs):
        # MCP tools expect a dict of arguments
        if hasattr(self._tool, "run"):
            return self._tool.run(kwargs)
        # Fallback: try invoke
        if hasattr(self._tool, "invoke"):
            return self._tool.invoke(kwargs)
        raise NotImplementedError(f"Tool {self.name} has no run or invoke method")

    async def _arun(self, **kwargs):
        # MCP tools expect a dict of arguments
        if hasattr(self._tool, "ainvoke"):
            return await self._tool.ainvoke(kwargs)
        elif hasattr(self._tool, "arun"):
            return await self._tool.arun(kwargs)
        elif hasattr(self._tool, "run"):
            return self._tool.run(kwargs)
        elif hasattr(self._tool, "invoke"):
            return self._tool.invoke(kwargs)
        raise NotImplementedError(f"Tool {self.name} has no async method")

    # Remove to_openai_tool - let LangChain's convert_to_openai_tool handle it


class RITSChatModel(BaseChatModel):
    """LangChain-compatible chat model using httpx for internal RITS inference service."""

    # Mapping from endpoint name (short) to payload model name (full)
    MODEL_NAME_MAPPING: Dict[str, str] = {
        "llama-3-3-70b-instruct": "meta-llama/llama-3-3-70b-instruct",
        "gpt-oss-120b": "openai/gpt-oss-120b"
    }

    model_name: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    bound_tools: Optional[List[Dict[str, Any]]] = Field(default=None)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        # Convert LangChain messages to simple dicts
        msgs = []
        for m in messages:
            if isinstance(m, SystemMessage):
                msgs.append({"role": "system", "content": m.content})

            elif isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})

            elif isinstance(m, AIMessage):
                msg_dict = {"role": "assistant", "content": m.content}
                if m.tool_calls:
                    msg_dict["tool_calls"] = m.additional_kwargs.get("tool_calls")
                if "reasoning" in m.additional_kwargs:
                    msg_dict["reasoning"] = m.additional_kwargs["reasoning"]

                msgs.append(msg_dict)

            elif isinstance(m, ToolMessage):
                msgs.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content
                })
            
            else:
                # Fallback for unexpected types
                msgs.append({"role": "user", "content": m.content})

        # Use short name for endpoint URL
        url = f"{self.base_url}/{self.model_name}/v1/chat/completions"
        headers = {"RITS_API_KEY": self.api_key}

        # Use full name for payload if mapping exists, otherwise use model_name
        payload_model_name = self.MODEL_NAME_MAPPING.get(
            self.model_name,
            self.model_name
        )

        # Build request payload
        payload = {
            "model": payload_model_name,
            "messages": msgs,
            "temperature": self.temperature,
            **kwargs
        }

        # Include tools if bound
        if self.bound_tools:
            payload["tools"] = self.bound_tools

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=60.0
            )
            resp.raise_for_status()
            data = resp.json()

        # Handle OpenAI-like structure with optional tool calls
        msg_data = data["choices"][0]["message"]
        content = msg_data.get("content") or ""

        # Check for tool calls
        additional_kwargs = {}
        if "tool_calls" in msg_data:
            additional_kwargs["tool_calls"] = msg_data["tool_calls"]
        if "reasoning" in msg_data and msg_data["reasoning"]:
            additional_kwargs["reasoning"] = msg_data["reasoning"]

        message = AIMessage(
            content=content, additional_kwargs=additional_kwargs
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Synchronous wrapper for _agenerate."""
        try:
            asyncio.get_running_loop()
            # Event loop is running, we shouldn't be here
            raise RuntimeError(
                "Cannot call synchronous _generate from within async context. "
                "Use ainvoke() or agenerate() instead."
            )
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self._agenerate(messages, stop, **kwargs))

    @property
    def _llm_type(self) -> str:
        return "rits-openai-compat"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        **kwargs
    ) -> "RITSChatModel":
        """Bind tools to this chat model.

        Args:
            tools: List of tools to bind (MCPToolWrapper, dicts, etc.)
            **kwargs: Additional arguments to pass to model

        Returns:
            New instance of RITSChatModel with tools bound
        """
        from langchain_core.utils.function_calling import (
            convert_to_openai_tool
        )

        tool_defs = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "args"):
                # Build the tool definition manually 
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.args,
                            "required": list(tool.args.keys())
                        }
                    }
                }
                tool_defs.append(tool_def)
            else:
                try:
                    # Try LangChain's standard conversion
                    tool_defs.append(convert_to_openai_tool(tool))
                except Exception as e:
                    # Fallback: if tool has schema/dict representation
                    if isinstance(tool, dict):
                        tool_defs.append(tool)
                    else:
                        raise ValueError(
                            f"Unable to convert tool {tool} to OpenAI format: {e}"
                        )

        # Create new instance with tools bound
        return self.model_copy(
            update={"bound_tools": tool_defs, **kwargs}
        )


def create_llm(llm_type="openai", model=None, ollama_base_url=None):
    """Create LLM instance based on type

    Args:
        llm_type: Either "openai" or "ollama"
        model: Model name (optional, uses defaults if not provided)
        ollama_base_url: Ollama server URL (default: http://localhost:11434)

    Returns:
        LLM instance
    """
    if llm_type == "openai":
        # Get RITS API key
        try:
            rits_api_key = os.environ["RITS_API_KEY"]
        except BaseException:
            raise ValueError(
                "You need to set the env var RITS_API_KEY to use a "
                "model from RITS."
            )

        base_url = (
            "https://inference-3scale-apicast-production.apps.rits."
            "fmaas.res.ibm.com"
        )
        model_name = model or "llama-3-3-70b-instruct"

        # Use RITSChatModel with the actual RITS inference service
        return RITSChatModel(
            model_name=model_name,
            base_url=base_url,
            api_key=rits_api_key,
            temperature=0
        )
    elif llm_type == "ollama":
        # Lazy import to avoid dependency conflicts when not using Ollama
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is required for Ollama support. "
                "Install with: pip install langchain-ollama"
            )

        model_name = model or "llama3.1:8b"
        base_url = ollama_base_url or "http://localhost:11434"

        print(f"Connecting to Ollama at {base_url}")
        print(
            f"Make sure Ollama is running and the model "
            f"'{model_name}' is pulled."
        )
        print("  To start Ollama: ollama serve")
        print(f"  To pull model: ollama pull {model_name}\n")

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0
        )
    else:
        raise ValueError(
            f"Unknown LLM type: {llm_type}. Must be 'openai' or 'ollama'"
        )
