"""Agent classes for tool-calling workflows"""
from .result_handle_manager import ResultHandleManager
from .tool_calling_agent import ToolCallingAgent

# Note: create_llm is available in .llm module but not imported here
# to avoid langchain dependencies when they're not needed

__all__ = [
    "ResultHandleManager",
    "ToolCallingAgent",
]
