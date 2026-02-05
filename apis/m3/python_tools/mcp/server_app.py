"""FastAPI WebSocket server wrapper for MCP Server.

This module provides a FastAPI application that exposes MCP server functionality
over WebSocket connections, allowing clients to connect to external server processes.
"""

import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mcp.server.websocket import websocket_server

from .config import MCPServerConfig, load_config
from .mcp_server import SlotFillingMCPServer, SelectionMCPServer

logger = logging.getLogger(__name__)


def create_app(config: MCPServerConfig) -> FastAPI:
    """Create FastAPI application with WebSocket MCP endpoint.

    Args:
        config: MCP server configuration

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title=config.server_name,
        description="MCP Server with slot-filling or selection tools",
        version="1.0.0"
    )

    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {
            "status": "ok",
            "server_name": config.server_name,
            "server_type": config.server_type or "slot_filling",
            "database": config.database_path,
        }

    @app.websocket("/mcp")
    async def websocket_endpoint(websocket: WebSocket):
        """MCP protocol endpoint over WebSocket.

        Handles MCP protocol messages using the configured server instance.
        The websocket_server context manager handles the WebSocket accept handshake.
        """
        logger.info(f"WebSocket connection requested for {config.server_name}")

        try:
            # Create appropriate server instance based on config
            server_type = config.server_type or "slot_filling"
            if server_type == "selection":
                mcp_server = SelectionMCPServer(config)
            else:
                mcp_server = SlotFillingMCPServer(config)

            logger.info(f"Created {server_type} MCP server instance")

            # Use MCP's websocket_server context manager to handle protocol
            # NOTE: Do NOT call websocket.accept() - websocket_server handles the entire
            # WebSocket lifecycle including the accept handshake
            async with websocket_server(
                websocket.scope,
                websocket.receive,
                websocket.send
            ) as (read_stream, write_stream):
                logger.info("MCP WebSocket connection established")

                # Run the MCP server protocol
                await mcp_server.server.run(
                    read_stream,
                    write_stream,
                    mcp_server.server.create_initialization_options()
                )

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
            # Try to close cleanly if possible
            try:
                await websocket.close(code=1011, reason=str(e))
            except:
                pass

    return app


def create_app_from_args(
    database_path: str,
    transport: str = "websocket",
    host: str = "127.0.0.1",
    port: int = 8000,
    **kwargs
) -> FastAPI:
    """Create FastAPI app from command-line style arguments.

    This is a convenience wrapper for create_app that accepts
    the same arguments as the CLI.

    Args:
        database_path: Path to SQLite database
        transport: Transport type (ignored, always WebSocket)
        host: Bind address (ignored, handled by uvicorn)
        port: Port number (ignored, handled by uvicorn)
        **kwargs: Additional config parameters (tables, cache_dir, etc.)

    Returns:
        FastAPI application instance
    """
    # Load config with provided arguments
    config = load_config(
        database_path=database_path,
        **{k: v for k, v in kwargs.items() if v is not None}
    )

    return create_app(config)
