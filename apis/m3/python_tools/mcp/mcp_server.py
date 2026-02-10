"""Live MCP Server implementations with base class for slot-filling and selection tools."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pathlib

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from pydantic import BaseModel, Field

from ..tools.pydantic_wrapper import PeekInput
from ..tools.dtype_utils import DTYPE_METADATA_KEY
from ..tools.tool_registry import (
    INITIALIZE_ACTIVE_DATA,
    PEEK_FCN,
    SLOT_FILLING_TOOLS,
    SELECTION_TOOLS,
    SlotFillingTools,
    SelectionTools,
)

from .config import MCPServerConfig

logger = logging.getLogger(__name__)

GET_DATA_FCN = "get_data"


class GetDataInput(BaseModel):
    """Input model for get_data tool."""
    tool_universe_id: Optional[str] = Field(
        None,
        description="Optional tool universe ID to switch to before retrieving data. "
        "If provided and different from current universe, the server will reload "
        "active_data for the specified universe before returning it."
    )


class LiveMCPServer(ABC):
    """
    Abstract base class for MCP Servers with live data manipulation tools.

    This base class provides common functionality for universe management,
    tool metadata building, and MCP protocol handlers. Subclasses implement
    specific toolbox types (slot-filling or selection).
    """

    def __init__(self, config: MCPServerConfig, shutdown_event: Optional[asyncio.Event] = None):
        self.config = config
        self.server = Server(name=config.server_name)
        self.shutdown_event = shutdown_event

        # Resolve path to mcp_init_mapping.json in project root
        project_root = pathlib.Path(__file__).parent.parent.parent.parent.parent
        self.universe_configuration_file = str(project_root / "apis" / "configs" / "mcp_init_mapping.json")

        # Load ALL tool configs at startup
        with open(self.universe_configuration_file) as f:
            self.all_tool_configs = json.load(f)

        # Current universe state (mutable)
        self.tool_universe_id = config.tool_universe_id or list(self.all_tool_configs.keys())[0]
        self._universe_lock = asyncio.Lock()  # Thread safety
        self.active_data: Optional[dict] = None  # Will be set by _load_tool_universe

        # Initialize toolbox (subclass-specific)
        self._initialize_toolbox()

        # Load initial universe
        self._load_tool_universe(self.tool_universe_id)

        # Build tool metadata
        self._build_tool_metadata()

        # Register handlers
        self._register_handlers()

    @abstractmethod
    def _initialize_toolbox(self) -> None:
        """Initialize the specific toolbox (SlotFillingTools or SelectionTools)."""
        pass

    @abstractmethod
    def _reload_active_data_for_universe(self, universe_id: str) -> None:
        """
        Load active_data for a specific tool universe.

        Args:
            universe_id: The universe ID to load data for
        """
        pass

    @abstractmethod
    def _get_available_tools(self) -> Dict[str, Any]:
        """
        Get the current available tools dictionary.

        Returns:
            Dictionary mapping tool names to tool functions
        """
        pass

    @abstractmethod
    def _get_tool_specs(self) -> list:
        """
        Get tool specifications for metadata building.

        Returns:
            List of tool specification dicts with name, input_model, output_model, docstring
        """
        pass

    @abstractmethod
    def _on_universe_switch(self, universe_id: str) -> None:
        """
        Hook called after switching universes.

        Subclasses can use this to update dynamic tools (e.g., regenerate getters).

        Args:
            universe_id: The universe ID that was just loaded
        """
        pass

    def _load_tool_universe(self, universe_id: str) -> None:
        """Load a specific tool universe's active_data."""
        if universe_id not in self.all_tool_configs:
            raise ValueError(f"Unknown universe ID: {universe_id}")

        self.tool_universe_id = universe_id
        self.tool_config = self.all_tool_configs[universe_id]["init_args"]
        self.tool_config["database_path"] = self.config.database_path

        logger.info(f"Loading tool universe: {universe_id}")
        logger.info(f"Init args: {self.tool_config}")

        # Reload active_data for this universe (subclass-specific)
        self._reload_active_data_for_universe(universe_id)

        # Call hook for subclass-specific post-switch actions
        self._on_universe_switch(universe_id)

        # Validate active_data structure
        if self.active_data is None:
            raise ValueError(f"active_data is None for universe '{universe_id}'")

        # Filter out _dtypes metadata key to get actual data columns
        valid_keys = [k for k in self.active_data.keys() if k != DTYPE_METADATA_KEY]
        nkeys = len(valid_keys)

        # Get number of entries from first data column (with type checking)
        if nkeys > 0:
            first_value = self.active_data[valid_keys[0]]
            nentries = len(first_value) if isinstance(first_value, (list, tuple)) else 0
        else:
            nentries = 0

        logger.info(f"Loaded universe '{universe_id}' with {nkeys} keys and {nentries} entries")

    def _build_tool_metadata(self) -> None:
        """Build input model and docstring mappings."""
        # Get tool specs from subclass
        tool_specs = self._get_tool_specs()

        self.tool_input_models = {
            spec["name"]: spec["input_model"]
            for spec in tool_specs
            if spec["name"] != INITIALIZE_ACTIVE_DATA
        }
        self.tool_input_models[PEEK_FCN] = PeekInput
        self.tool_input_models[GET_DATA_FCN] = GetDataInput

        self.tool_docstrings = {
            spec["name"]: spec["docstring"]
            for spec in tool_specs
            if spec["name"] != INITIALIZE_ACTIVE_DATA
        }
        self.tool_docstrings[PEEK_FCN] = (
            "Peek at the data to see its structure and content."
        )
        self.tool_docstrings[GET_DATA_FCN] = (
            "Get the data from the server. Optionally switch to a different tool universe "
            "by providing tool_universe_id parameter. If the universe is switched, "
            "the active data for the new universe will be returned."
        )

    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            tools = []
            available_tools = self._get_available_tools()

            for name in available_tools:
                if name == INITIALIZE_ACTIVE_DATA:
                    continue

                # Try to get input model from registry, or use function signature
                input_model = self.tool_input_models.get(name)

                # Get docstring from registry or function
                if name in self.tool_docstrings:
                    description = self.tool_docstrings[name]
                elif hasattr(available_tools[name], '__doc__') and available_tools[name].__doc__:
                    description = available_tools[name].__doc__.strip()
                else:
                    description = f"Tool: {name}"

                if input_model:
                    # Get the input schema
                    input_schema = input_model.model_json_schema()

                    # DEBUG LOGGING: Capture schema for retrieve_data
                    if name == "retrieve_data":
                        logger.debug(
                            "list_tools() - retrieve_data schema: %s\nProperties: %s",
                            json.dumps(input_schema, indent=2),
                            list(input_schema.get('properties', {}).keys()),
                        )

                    # Note: key_name validation happens at runtime in call_tool()
                    # We don't inject allowed values into the schema to avoid client-side caching issues

                    tools.append(
                        Tool(
                            name=name,
                            description=description,
                            inputSchema=input_schema
                        )
                    )

            # Add get_data tool
            tools.append(
                Tool(
                    name=GET_DATA_FCN,
                    description=self.tool_docstrings[GET_DATA_FCN],
                    inputSchema=GetDataInput.model_json_schema()
                )
            )

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]):
            # Handle get_data tool with optional universe switching
            if name == GET_DATA_FCN:
                # Parse input with optional tool_universe_id
                input_data = GetDataInput(**arguments)

                # If tool_universe_id is provided, switch to that universe first
                if input_data.tool_universe_id is not None:
                    async with self._universe_lock:
                        try:
                            universe_id = input_data.tool_universe_id

                            # Validate universe ID exists
                            if universe_id not in self.all_tool_configs:
                                available = list(self.all_tool_configs.keys())[:10]
                                error_msg = (
                                    f"Error: Unknown universe ID '{universe_id}'. "
                                    f"Available IDs (first 10): {available}..."
                                )
                                logger.error(error_msg)
                                return [TextContent(type="text", text=json.dumps({"error": error_msg}))]

                            # Only switch if different from current
                            if universe_id != self.tool_universe_id:
                                logger.info(f"Switching from '{self.tool_universe_id}' to '{universe_id}'")
                                self._load_tool_universe(universe_id)
                            else:
                                logger.info(f"Already in universe '{universe_id}', no switch needed")

                        except Exception as e:
                            logger.error(f"Error switching universe: {e}")
                            error_msg = f"Error switching to universe: {str(e)}"
                            return [TextContent(type="text", text=json.dumps({"error": error_msg}))]

                # Return active data for current universe
                result = self.active_data
            # Handle standard tools
            else:
                available_tools = self._get_available_tools()
                if name not in available_tools:
                    raise ValueError(f"Unknown tool: {name}")

                # Validate key_name if present in arguments
                if "key_name" in arguments:
                    key_name = arguments["key_name"]
                    valid_keys = [k for k in self.active_data.keys() if k != DTYPE_METADATA_KEY]
                    if key_name not in valid_keys:
                        error_msg = (
                            f"Invalid key_name '{key_name}'. "
                            f"Valid keys in current universe ('{self.tool_universe_id}'): {valid_keys}"
                        )
                        logger.error(error_msg)
                        return [TextContent(type="text", text=json.dumps({"error": error_msg}))]

                tool_func = available_tools[name]

                if name == PEEK_FCN:
                    input_model = self.tool_input_models[name]
                    inputs = input_model(**arguments)
                    result = tool_func(inputs)
                else:
                    # Call tool (Pydantic wrapper will handle _dtypes stripping if present)
                    try:
                        result = tool_func(**arguments)
                    except Exception as e:
                        # Catch validation errors and return as JSON error
                        error_msg = f"Input validation error: {str(e)}"
                        logger.error(f"{name} failed: {error_msg}", exc_info=True)

                        # DEBUG LOGGING
                        if name == "filter_data":
                            logger.debug("ERROR calling %s: %s", name, str(e))

                        return [TextContent(type="text", text=error_msg)]

            # Check for None result
            if result is None:
                error_msg = f"Tool {name} returned None"
                logger.error(error_msg)
                return [TextContent(type="text", text=json.dumps({"error": error_msg}))]

            if hasattr(result, "model_dump_json"):
                result_text = result.model_dump_json()
            elif isinstance(result, (dict, list)):
                result_text = json.dumps(result)
            else:
                result_text = str(result)

            return [TextContent(type="text", text=result_text)]

    async def run(self) -> None:
        """Run the MCP server with graceful shutdown support."""
        import signal

        loop = asyncio.get_running_loop()

        def handle_shutdown_signal():
            logger.info(
                "Shutdown signal received, stopping server gracefully..."
            )
            if self.shutdown_event:
                self.shutdown_event.set()

        # Register signal handlers
        if self.shutdown_event:
            loop.add_signal_handler(signal.SIGINT, handle_shutdown_signal)
            loop.add_signal_handler(signal.SIGTERM, handle_shutdown_signal)

        try:
            logger.info(
                f"MCP server: {self.config.server_name} is ready; "
                "waiting for initialize()"
            )

            async with stdio_server() as (read_stream, write_stream):
                # Create server task
                server_task = asyncio.create_task(
                    self.server.run(
                        read_stream,
                        write_stream,
                        self.server.create_initialization_options(),
                    )
                )

                # Wait for either server completion or shutdown signal
                if self.shutdown_event:
                    shutdown_task = asyncio.create_task(
                        self.shutdown_event.wait()
                    )
                    done, pending = await asyncio.wait(
                        [server_task, shutdown_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Check if shutdown was signaled
                    if shutdown_task in done:
                        logger.info("Shutting down server...")
                        # Cancel server task
                        server_task.cancel()
                        # Give cancellation a moment to propagate, but don't
                        # block on I/O completion
                        await asyncio.sleep(0)
                        # Suppress the cancelled task exception and exit
                        raise KeyboardInterrupt()

                    # Server completed naturally, clean up shutdown watcher
                    shutdown_task.cancel()
                    try:
                        await shutdown_task
                    except asyncio.CancelledError:
                        pass
                else:
                    # No shutdown event, just wait for server
                    await server_task

        finally:
            # Cleanup signal handlers
            if self.shutdown_event:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            logger.info("Server stopped")


class SlotFillingMCPServer(LiveMCPServer):
    """
    MCP Server exposing slot-filling data manipulation tools.

    The server auto-loads database tables on startup and provides
    tools for filtering, sorting, aggregating, and transforming data.
    """

    def _initialize_toolbox(self) -> None:
        """Initialize slot filling tools."""
        self.slot_filling_tools = SlotFillingTools(
            io_cache=self.config.cache_dir,
            use_io_wrappers=self.config.use_io_wrappers,
            use_pydantic_signatures=self.config.use_pydantic_signatures,
        )

    def _reload_active_data_for_universe(self, universe_id: str) -> None:
        """Reload active_data using slot-filling initialize_active_data tool."""
        _ = universe_id  # Unused but required by abstract method signature
        self.active_data = self.slot_filling_tools.tools[INITIALIZE_ACTIVE_DATA](**self.tool_config)

    def _get_available_tools(self) -> Dict[str, Any]:
        """Return slot-filling tools dictionary."""
        return self.slot_filling_tools.tools

    def _get_tool_specs(self) -> list:
        """Return slot-filling tool specifications."""
        return SLOT_FILLING_TOOLS

    def _on_universe_switch(self, universe_id: str) -> None:
        """No-op for slot filling (no dynamic tools to update)."""
        _ = universe_id  # Unused but required by abstract method signature


class SelectionMCPServer(LiveMCPServer):
    """
    MCP Server exposing selection-based data manipulation tools.

    The server auto-loads database tables on startup and provides:
    - Specialized filter/sort/aggregate tools
    - Dynamically generated getter functions based on the current data schema
    """

    def _initialize_toolbox(self) -> None:
        """Initialize selection tools."""
        self.selection_tools = SelectionTools(
            io_cache=self.config.cache_dir,
            use_io_wrappers=self.config.use_io_wrappers,
            use_pydantic_signatures=self.config.use_pydantic_signatures,
        )
        # Store base tools before getters are added
        self.base_tools = self.selection_tools.base_tools

    def _reload_active_data_for_universe(self, universe_id: str) -> None:
        """
        Reload active_data and regenerate getter functions for new universe.

        This extracts the schema from the loaded data and creates getter
        functions for each column.
        """
        _ = universe_id  # Unused but required by abstract method signature
        # Initialize active data using base tools
        self.active_data = self.base_tools[INITIALIZE_ACTIVE_DATA](**self.tool_config)

        # Extract schema and generate getters
        self._regenerate_getters()

    def _get_available_tools(self) -> Dict[str, Any]:
        """Return selection tools dictionary (includes dynamically generated getters)."""
        # Type assertion: get_toolbox_with_schema always returns a dict
        tools: Dict[str, Any] = self.selection_tools.tools or {}
        return tools

    def _get_tool_specs(self) -> list:
        """Return selection tool specifications."""
        return SELECTION_TOOLS

    def _on_universe_switch(self, universe_id: str) -> None:
        """Regenerate getter functions when universe switches."""
        _ = universe_id  # Unused but required by abstract method signature
        self._regenerate_getters()

    def _extract_schema_from_active_data(self, active_data: dict) -> list[dict]:
        """
        Extract key_names_and_descriptions from active_data.

        Args:
            active_data: The active data dictionary with optional dtype metadata

        Returns:
            List of dicts with 'key_name', 'description', and 'dtype' fields
        """
        key_names_and_descriptions = []
        dtypes = active_data.get(DTYPE_METADATA_KEY, {})

        for key in active_data.keys():
            if key == DTYPE_METADATA_KEY:
                continue

            key_names_and_descriptions.append({
                'key_name': key,
                'description': f"Column {key}",  # Simple description since we don't have table metadata
                'dtype': dtypes.get(key, 'object')
            })

        return key_names_and_descriptions

    def _regenerate_getters(self) -> None:
        """Regenerate getter functions based on current active_data schema."""
        if self.active_data is None:
            raise ValueError("Cannot regenerate getters: active_data is None")

        # Extract schema from active data
        key_names_and_descriptions = self._extract_schema_from_active_data(self.active_data)

        # Use SelectionTools' get_toolbox_with_schema to create/update tools with getters
        self.selection_tools.tools = self.selection_tools.get_toolbox_with_schema(
            key_names_and_descriptions
        )

        logger.info(f"Regenerated {len(key_names_and_descriptions)} getter functions")


def create_server(config: MCPServerConfig, shutdown_event: Optional[asyncio.Event] = None) -> LiveMCPServer:
    """
    Factory function to create a configured server instance.

    Args:
        config: Server configuration
        shutdown_event: Optional asyncio.Event for graceful shutdown

    Returns:
        SlotFillingMCPServer or SelectionMCPServer based on config.server_type
    """
    # Determine server type from config or server name
    server_type = getattr(config, 'server_type', None)

    if server_type is None:
        # Infer from server name if not explicitly set
        if 'selection' in config.server_name.lower():
            server_type = 'selection'
        else:
            server_type = 'slot_filling'

    if server_type == 'selection':
        return SelectionMCPServer(config, shutdown_event)
    elif server_type == 'slot_filling':
        return SlotFillingMCPServer(config, shutdown_event)
    else:
        raise ValueError(f"Unknown server_type: {server_type}. Must be 'slot_filling' or 'selection'")
