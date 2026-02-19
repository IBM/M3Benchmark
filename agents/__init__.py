"""Agent classes for tool-calling workflows"""
from .components.result_handle_manager import ResultHandleManager
from .mcp_tool_wrapper import MCPToolWrapper

# Note: create_llm is available in .llm module but not imported here
# to avoid langchain dependencies when they're not needed

__all__ = [
    "ResultHandleManager",
    "MCPToolWrapper"
]
