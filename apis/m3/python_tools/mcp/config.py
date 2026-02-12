"""Configuration management for the Slot-Filling MCP Server."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MCPServerConfig:
    """Configuration for the MCP Server (slot-filling or selection mode)."""

    # Required: Path to SQLite database
    database_path: str

    # Optional: Tables to load (default: all tables in database)
    tables: Optional[List[str]] = None

    # Optional: Cache directory for intermediate data
    cache_dir: str = "./mcp_cache"

    # Optional: Enable IO wrappers for caching
    use_io_wrappers: bool = False

    # Optional: Enable Pydantic signatures
    use_pydantic_signatures: bool = True

    # Optional: Server name
    server_name: str = "slot-filling-mcp-server"

    # Optional: Tool universe ID for tracking different server instances
    tool_universe_id: Optional[str] = None

    # Optional: Server type ('slot_filling' or 'selection')
    server_type: Optional[str] = None

    # Optional: Transport protocol ('stdio' or 'websocket')
    transport: str = "stdio"

    # Optional: Host to bind WebSocket server (only used with transport='websocket')
    host: str = "127.0.0.1"

    # Optional: Port for WebSocket server (only used with transport='websocket')
    port: int = 8000


# Standard config file locations to search
DEFAULT_CONFIG_LOCATIONS = [
    "./slot-filling-mcp.json",
    "~/.config/slot-filling-mcp/config.json",
    str(Path(__file__).parent / "default_config.json"),
]


def find_default_config() -> Optional[str]:
    """Search standard locations for a config file."""
    for loc in DEFAULT_CONFIG_LOCATIONS:
        path = Path(loc).expanduser()
        if path.exists():
            return str(path)
    return None


def load_config(
    config_file: Optional[str] = None,
    database_path: Optional[str] = None,
    tables: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    tool_universe_id: Optional[str] = None,
    server_type: Optional[str] = None,
    transport: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> MCPServerConfig:
    """
    Load configuration with priority:
    1. CLI arguments (passed directly)
    2. Environment variables
    3. Config file
    4. Defaults

    Args:
        config_file: Path to JSON config file
        database_path: Path to SQLite database (overrides config/env)
        tables: List of tables to load (overrides config/env)
        cache_dir: Cache directory (overrides config/env)
        tool_universe_id: Unique ID for this server instance (overrides config/env)
        server_type: Server type - 'slot_filling' or 'selection' (overrides config/env)
        transport: Transport protocol - 'stdio' or 'websocket' (overrides config/env)
        host: Host to bind WebSocket server (overrides config/env)
        port: Port for WebSocket server (overrides config/env)

    Returns:
        MCPServerConfig instance

    Raises:
        ValueError: If database_path is not provided through any method
        FileNotFoundError: If config file is specified but doesn't exist
    """
    config_data: dict = {}

    # Load from config file if provided, or search for default
    if not config_file:
        config_file = find_default_config()

    if config_file:
        config_path = Path(config_file).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_path) as f:
            config_data = json.load(f)

    # Environment variable overrides
    env_domain = os.environ.get("MCP_DOMAIN")
    env_cache = os.environ.get("MCP_CACHE_DIR")
    env_tables = os.environ.get("MCP_TABLES")
    env_tool_universe_id = os.environ.get("MCP_TOOL_UNIVERSE_ID")
    env_server_type = os.environ.get("MCP_SERVER_TYPE")
    env_transport = os.environ.get("MCP_TRANSPORT")
    env_host = os.environ.get("MCP_HOST")
    env_port = os.environ.get("MCP_PORT")

    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Config loading: MCP_DOMAIN={env_domain}")
    logger.debug(f"Config loading: config_data before domain processing={config_data.get('database_path', 'NOT SET')}")

    # Construct database path from domain (env variable or default)
    # Priority: env_domain (if set) > config file > default "superhero"
    if env_domain is not None:
        # MCP_DOMAIN explicitly set - override config file
        config_data["database_path"] = f"data/db/{env_domain}/{env_domain}.sqlite"
        logger.debug(f"Config loading: Set database_path from MCP_DOMAIN: {config_data['database_path']}")
    elif "database_path" not in config_data:
        # No config file path and no env - use default superhero
        config_data["database_path"] = "data/db/superhero/superhero.sqlite"
        logger.debug(f"Config loading: Set database_path to default: {config_data['database_path']}")
    if env_cache:
        config_data["cache_dir"] = env_cache
    if env_tables:
        config_data["tables"] = [t.strip() for t in env_tables.split(",")]
    if env_tool_universe_id:
        config_data["tool_universe_id"] = env_tool_universe_id
    if env_server_type:
        config_data["server_type"] = env_server_type
    if env_transport:
        config_data["transport"] = env_transport
    if env_host:
        config_data["host"] = env_host
    if env_port:
        config_data["port"] = int(env_port)

    # CLI argument overrides (highest priority)
    if database_path:
        config_data["database_path"] = database_path
    if tables:
        config_data["tables"] = tables
    if cache_dir:
        config_data["cache_dir"] = cache_dir
    if tool_universe_id:
        config_data["tool_universe_id"] = tool_universe_id
    if server_type:
        config_data["server_type"] = server_type
    if transport:
        config_data["transport"] = transport
    if host:
        config_data["host"] = host
    if port is not None:  # Allow port 0 if explicitly specified
        config_data["port"] = port

    # Validate required fields
    if "database_path" not in config_data:
        raise ValueError(
            "Database path is required. Provide via:\n"
            "  - CLI: slot-filling-mcp --db /path/to/database.sqlite\n"
            "  - Config file: --config config.json\n"
            "  - Environment: MCP_DOMAIN=superhero (constructs db/{domain}/{domain}.sqlite)"
        )

    # Filter out None values and unknown keys
    valid_keys = {f.name for f in MCPServerConfig.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in config_data.items() if k in valid_keys}

    return MCPServerConfig(**filtered_data)
