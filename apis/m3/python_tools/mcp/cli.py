"""CLI entry point for the Slot-Filling and Selection MCP Server."""

import argparse
import asyncio
import logging
import sys
from typing import List, Optional

from apis.mcp_logging import _JsonFormatter
from .config import find_default_config, load_config


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Always writes JSON lines to stderr — stdout is reserved for the MCP protocol.
    Each record includes ``task_id`` and ``domain`` from the environment so logs
    from parallel runs can be filtered with ``jq``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="slot-filling-mcp",
        description="MCP Server for slot-filling and selection data manipulation tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # STDIO mode (default, for subprocess communication)
  slot-filling-mcp --db /path/to/database.sqlite

  # WebSocket mode (for external connections)
  slot-filling-mcp --db /path/to/database.sqlite --transport websocket --port 8000

  # Run with specific tables
  slot-filling-mcp --db /path/to/database.sqlite --tables users,orders

  # Run with config file
  slot-filling-mcp --config ./slot-filling-mcp.json

  # Use environment variables
  MCP_DATABASE=/path/to/db.sqlite slot-filling-mcp

Configuration priority (highest to lowest):
  1. Command line arguments
  2. Environment variables
  3. Config file
  4. Default values

Environment variables:
  MCP_DATABASE  - Path to SQLite database
  MCP_CACHE_DIR - Cache directory for IO operations
  MCP_TABLES    - Comma-separated list of tables to load
        """,
    )

    parser.add_argument(
        "--db",
        "--database",
        dest="database",
        metavar="PATH",
        help="Path to SQLite database file",
    )

    parser.add_argument(
        "--tables",
        metavar="TABLE1,TABLE2,...",
        help="Comma-separated list of tables to load (default: all tables)",
    )

    parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        help="Path to JSON config file",
    )

    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        help="Directory for caching intermediate data (default: ./mcp_cache)",
    )

    parser.add_argument(
        "--mode",
        "--server-type",
        dest="server_type",
        choices=["slot_filling", "selection", "router"],
        metavar="MODE",
        help="Server mode: 'router' (default, routes per query), 'slot_filling' (generic tools), or 'selection' (specialized tools with getters)",
    )

    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "websocket"],
        default="stdio",
        help="Transport protocol: 'stdio' (default, for subprocess) or 'websocket' (for network connections)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind WebSocket server (default: 127.0.0.1, only used with --transport websocket)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for WebSocket server (default: 8000, only used with --transport websocket)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser.parse_args(args)


async def run_server_with_shutdown(config) -> None:
    """Run the server with graceful shutdown support."""
    from .mcp_server import create_server

    # Create shutdown event
    shutdown_event = asyncio.Event()

    # Create and run server
    server = create_server(config, shutdown_event)
    await server.run()


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parsed_args = parse_args(args)

    setup_logging(parsed_args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Parse tables if provided
        tables = None
        if parsed_args.tables:
            tables = [t.strip() for t in parsed_args.tables.split(",")]

        # Find config file
        config_file = parsed_args.config or find_default_config()

        # Load configuration
        config = load_config(
            config_file=config_file,
            database_path=parsed_args.database,
            tables=tables,
            cache_dir=parsed_args.cache_dir,
            server_type=parsed_args.server_type,
            transport=parsed_args.transport,
            host=parsed_args.host,
            port=parsed_args.port,
        )

        # Branch based on transport type
        transport = parsed_args.transport

        if transport == "stdio":
            # STDIO mode: Run as subprocess with stdio_server
            logger.info("Starting MCP server in STDIO mode")
            asyncio.run(run_server_with_shutdown(config))

        elif transport == "websocket":
            # WebSocket mode: Run as FastAPI/uvicorn HTTP server
            logger.info(f"Starting MCP server in WebSocket mode on {parsed_args.host}:{parsed_args.port}")

            try:
                import uvicorn
            except ImportError:
                logger.error("uvicorn not installed. Install with: pip install 'uvicorn[standard]'")
                return 1

            from .server_app import create_app

            # Create FastAPI app with the loaded config
            app = create_app(config)

            # Run uvicorn server
            uvicorn.run(
                app,
                host=parsed_args.host,
                port=parsed_args.port,
                log_level="debug" if parsed_args.verbose else "info"
            )

        else:
            logger.error(f"Unknown transport: {transport}")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except ValueError as e:
        logger.error(str(e))
        return 1
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
