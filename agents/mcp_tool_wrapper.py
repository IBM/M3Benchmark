"""Unified MCPToolWrapper for converting MCP tools to LangChain StructuredTool.

This module provides a unified implementation that replaces multiple scattered
MCPToolWrapper implementations across the codebase. It properly preserves MCP
tool schemas by converting JSON Schema to Pydantic models, ensuring LLMs can
see complete tool parameter definitions.
"""

import asyncio
import re
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from pydantic import BaseModel, Field, create_model


class MCPToolWrapper:
    """Wrapper to convert MCP tools to LangChain StructuredTool objects.

    This class fetches tools from an MCP ClientSession and converts them to
    LangChain StructuredTool instances with proper schema preservation. It
    supports OpenAI name restrictions, profiling/instrumentation, and caching.

    Key Features:
        - Schema Preservation: Converts MCP JSON Schema to Pydantic
          BaseModel
        - OpenAI Compatibility: Optional name sanitization (64 chars,
          alphanumeric)
        - Profiling Support: Optional timing instrumentation with
          callbacks
        - Caching: Cache tool list for performance
        - Async/Sync Support: Both execution modes

    Example:
        >>> wrapper = MCPToolWrapper(session=mcp_session)
        >>> tools = await wrapper.get_tools()
        >>> # Use tools with LangChain agent
        >>> agent = create_react_agent(llm, tools)

        >>> # With OpenAI restrictions
        >>> wrapper = MCPToolWrapper(
        ...     session=mcp_session,
        ...     use_openai_restrictions=True
        ... )

        >>> # With profiling
        >>> def profile_callback(data):
        ...     print(f"Tool {data['tool_name']} took {data['duration_s']:.3f}s")
        >>> wrapper = MCPToolWrapper(
        ...     session=mcp_session,
        ...     enable_profiling=True,
        ...     profile_callback=profile_callback
        ... )
    """

    def __init__(
        self,
        session: ClientSession,
        use_openai_restrictions: bool = False,
        enable_profiling: bool = False,
        profile_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        current_query_index: Optional[int] = None,
        cache_tools: bool = True,
        capability_id: Optional[int] = None,
        domain: Optional[str] = None,
    ):
        """Initialize MCPToolWrapper.

        Args:
            session: MCP ClientSession for calling tools
            use_openai_restrictions: Apply OpenAI name restrictions
                (64 chars, alphanumeric)
            enable_profiling: Enable timing instrumentation
            profile_callback: Callback function for profiling data with
                signature:
                f(data: Dict[str, Any]) where data contains:
                    - tool_name: str
                    - duration_s: float
                    - query_index: Optional[int]
            current_query_index: Query index for profiling context
            cache_tools: Cache tool list to avoid repeated fetches
            capability_id: Unused; kept for backwards compatibility.
            domain: Unused; kept for backwards compatibility.
        """
        self.session = session
        self.use_openai_restrictions = use_openai_restrictions
        self.enable_profiling = enable_profiling
        self.profile_callback = profile_callback
        self.current_query_index = current_query_index
        self.cache_tools = cache_tools
        self.capability_id = capability_id
        self.domain = domain
        self._tools_cache: Optional[List[StructuredTool]] = None

    async def get_tools(self) -> List[StructuredTool]:
        """Fetch tools from MCP server and convert to LangChain tools.

        Returns:
            List of StructuredTool instances with proper args_schema set
        """
        if self.cache_tools and self._tools_cache is not None:
            return self._tools_cache

        response = await self.session.list_tools()
        tools = [self._create_langchain_tool(t) for t in response.tools]

        if self.cache_tools:
            self._tools_cache = tools

        return tools

    def _create_langchain_tool(self, mcp_tool) -> StructuredTool:
        """Convert single MCP tool to LangChain StructuredTool.

        Args:
            mcp_tool: MCP Tool object with name, description, inputSchema

        Returns:
            StructuredTool with proper args_schema for LLM visibility
        """
        # 1. Convert schema (CRITICAL: This ensures LLM can see parameters)
        args_schema = None
        if hasattr(mcp_tool, 'inputSchema') and mcp_tool.inputSchema:
            args_schema = self._convert_json_schema_to_pydantic(
                mcp_tool.name,
                mcp_tool.inputSchema
            )

        # 2. Sanitize name if needed
        tool_name = mcp_tool.name
        if self.use_openai_restrictions:
            tool_name = self._sanitize_tool_name_for_openai(tool_name)

        # 3. Create tool functions
        async_func = self._create_tool_function_async(mcp_tool)
        sync_func = self._create_tool_function_sync(async_func)

        # 4. Create StructuredTool
        return StructuredTool(
            name=tool_name,
            description=mcp_tool.description or f"Tool: {mcp_tool.name}",
            func=sync_func,
            coroutine=async_func,
            args_schema=args_schema,  # CRITICAL: Preserves schema for LLM!
        )

    def _convert_json_schema_to_pydantic(
        self,
        tool_name: str,
        input_schema: dict
    ) -> Optional[type[BaseModel]]:
        """Convert MCP tool's JSON Schema to Pydantic BaseModel.

        Handles:
        - Basic types: string, integer, number, boolean, array, object
        - Union types (anyOf)
        - Enum constraints
        - Required vs optional fields
        - Field descriptions
        - Default values

        Args:
            tool_name: Name of the tool (used for model class name)
            input_schema: JSON Schema dict from mcp_tool.inputSchema

        Returns:
            Dynamically created Pydantic BaseModel class or None if
            schema is empty
        """
        if not isinstance(input_schema, dict) or "properties" not in input_schema:
            return None

        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        fields = {}

        for prop_name, prop_info in properties.items():
            # Map JSON Schema type to Python type
            python_type = self._map_json_type_to_python(prop_info)

            # Extract metadata
            description = prop_info.get("description", "")
            default = prop_info.get("default", ...)

            # Add enum values to description if present
            enum_values = prop_info.get("enum")
            if enum_values:
                enum_str = ", ".join(map(str, enum_values))
                description = (
                    f"{description} Allowed values: {enum_str}".strip()
                )

            # Create field (required vs optional)
            if prop_name in required and default == ...:
                # Required field
                fields[prop_name] = (
                    python_type, Field(..., description=description)
                )
            else:
                # Optional field
                fields[prop_name] = (
                    Optional[python_type],
                    Field(default=default, description=description)
                )

        if not fields:
            return None

        # Create dynamic Pydantic model
        model_name = f"{tool_name}Input"
        return create_model(model_name, **fields)

    def _map_json_type_to_python(self, prop_info: dict) -> type:
        """Map JSON Schema type to Python type.

        Args:
            prop_info: Property info dict from JSON Schema

        Returns:
            Python type for Pydantic field
        """
        # Handle anyOf (union types)
        if "anyOf" in prop_info:
            return Any

        # Handle enum (string with constraints)
        if "enum" in prop_info:
            return str

        # Handle standard types
        if "type" in prop_info:
            json_type = prop_info["type"]

            # Handle multiple types (e.g., ["string", "null"])
            if isinstance(json_type, list):
                return Any

            # Standard type mapping
            type_map = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            return type_map.get(json_type, Any)

        # Default to Any for unknown types
        return Any

    def _sanitize_tool_name_for_openai(self, name: str) -> str:
        """Sanitize tool name for OpenAI API restrictions.

        OpenAI requires:
        - Max 64 characters
        - Only alphanumeric, underscore, hyphen

        Strategy:
        1. Replace invalid chars with underscore
        2. If > 64 chars, try removing common prefixes/infixes
        3. If still > 64 chars, truncate

        Args:
            name: Original tool name

        Returns:
            Sanitized tool name compatible with OpenAI
        """
        # Replace invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)

        # If too long, try shortening
        if len(sanitized) > 64:
            # Remove common patterns
            sanitized = (
                sanitized
                .replace("get_", "")
                .replace("_by_", "_")
                .replace("_with_", "_")
            )

        # Final truncation if still too long
        if len(sanitized) > 64:
            sanitized = sanitized[:64]

        return sanitized

    def _create_tool_function_async(self, mcp_tool) -> Callable:
        """Create async function that calls MCP tool with optional profiling.

        Args:
            mcp_tool: MCP Tool object

        Returns:
            Async callable that executes the tool
        """
        async def tool_func(**kwargs) -> str:
            # Start profiling timer if enabled
            if self.enable_profiling:
                t0 = time.perf_counter()

            # Call MCP tool
            result = await self.session.call_tool(mcp_tool.name, kwargs)

            # Handle profiling
            if self.enable_profiling:
                duration = time.perf_counter() - t0

                # Call profiling callback if provided
                if self.profile_callback:
                    self.profile_callback({
                        "tool_name": mcp_tool.name,
                        "duration_s": duration,
                        "query_index": self.current_query_index,
                    })

                # Print profiling info
                print(f"  [tool] {mcp_tool.name} -> {duration:.3f}s")

            # Extract result text
            if result.content:
                return result.content[0].text
            return "No result"

        return tool_func

    def _create_tool_function_sync(self, async_func: Callable) -> Callable:
        """Create sync wrapper for async tool function.

        Args:
            async_func: Async function to wrap

        Returns:
            Sync callable that uses asyncio.run()
        """
        def sync_wrapper(**kwargs) -> str:
            return asyncio.run(async_func(**kwargs))

        return sync_wrapper
