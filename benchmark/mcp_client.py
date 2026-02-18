import contextlib
from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
import yaml

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Dict, List, Optional

# Default settings
DEFAULT_CONTAINER_NAME = "fastapi-mcp-server"

def detect_container_runtime() -> str:
    """
    Detect available container runtime (podman or docker).
    Returns 'podman' if available, otherwise 'docker'.
    """
    
    # Check if podman is available
    if shutil.which("podman"):
        return "podman"

    # Fall back to docker
    if shutil.which("docker"):
        print("  Note: podman not found, using docker instead")
        return "docker"

    # Neither found
    raise RuntimeError(
        "Neither podman nor docker found in PATH. "
        "Please install one of them."
    )


@dataclass
class MCPConnectionConfig:
    """Connection settings for a single task's MCP server."""
    mode: str = "stdio"
    container_name: str = DEFAULT_CONTAINER_NAME
    container_runtime: Optional[str] = None  # None = auto-detect
    container_command: Optional[List[str]] = None
    container_env: Optional[Dict[str, str]] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    server_url: Optional[str] = None


def load_mcp_config(config_path: str) -> Dict[int, MCPConnectionConfig]:
    """Load MCP connection config from YAML file.

    Returns a dict mapping task_id (int) to MCPConnectionConfig.
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    tasks = data.get("tasks", {})
    result = {}
    for k, v in tasks.items():
        cmd = v.get("command", None)
        # "python" in config means the current Python executable (virtualenv-safe)
        if cmd == "python":
            cmd = sys.executable
        result[int(k)] = MCPConnectionConfig(
            mode=v.get("mode", "stdio"),
            container_name=v.get("container_name", DEFAULT_CONTAINER_NAME),
            container_runtime=v.get("container_runtime", None),
            container_command=v.get("container_command", None),
            container_env=v.get("container_env", None),
            command=cmd,
            args=v.get("args", None),
            server_url=v.get("server_url", None),
        )
    return result


@contextlib.asynccontextmanager
async def create_client_and_connect(cfg: MCPConnectionConfig, domain: str = ""):
    """Async context manager yielding an initialized ClientSession.

    Connection mode is determined by cfg.mode:
    - "stdio" with cfg.command: local subprocess (SlotFilling/SelectionMCPServer)
    - "stdio" without cfg.command: container exec (FastAPIMCPServer)
    - "websocket": WebSocket connection

    Yields:
        ClientSession — an initialized MCP client session ready to use
    """
    if cfg.mode == "stdio":
        if cfg.command:
            # Local subprocess mode
            print(
                f"  Starting MCP server: {cfg.command}"
                f" {' '.join(cfg.args or [])}"
            )
            env = os.environ.copy()
            if domain:
                env["MCP_DOMAIN"] = domain
            server_params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args or [],
                env=env,
            )
        else:
            # Container exec mode (FastAPIMCPServer)
            runtime = cfg.container_runtime or "(auto-detect)"
            print(
                f"  Starting MCP server: {runtime} exec -i"
                f" -e MCP_DOMAIN={domain} {cfg.container_name} python mcp_server.py"
            )
            runtime = cfg.container_runtime
            if not runtime:
                runtime = detect_container_runtime()
                print(f"  Auto-detected container runtime: {runtime}")
            if not cfg.container_name:
                raise ValueError(
                    "stdio mode requires either command or container_name"
                )
            exec_env = {"MCP_DOMAIN": domain} if domain else {}
            if cfg.container_env:
                exec_env.update(cfg.container_env)
            env_args = []
            for k, v in exec_env.items():
                env_args += ["-e", f"{k}={v}"]
            cmd = cfg.container_command or ["python", "mcp_server.py"]
            server_params = StdioServerParameters(
                command=runtime,
                args=["exec", "-i"] + env_args + [cfg.container_name] + cmd,
                env=None,
            )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    elif cfg.mode == "websocket":
        if not cfg.server_url:
            raise ValueError("websocket mode requires server_url")
        print(f"  Connecting to MCP server via websocket: {cfg.server_url}")
        from mcp.client.websocket import websocket_client
        async with websocket_client(cfg.server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    else:
        raise ValueError(
            f"Unknown mode: {cfg.mode!r}. Must be 'stdio' or 'websocket'"
        )
    
def stop_mcp_server(cfg: MCPConnectionConfig):
    """Stop a running MCP server process.

    For container stdio mode, force-kills the mcp_server.py process inside the
    container.  For subprocess stdio and websocket modes the transport's own
    context manager handles teardown, so this is a no-op.
    """
    if cfg.mode == "websocket":
        # WebSocket connection is closed by the context manager; nothing to do.
        return

    if cfg.mode == "stdio" and not cfg.command and cfg.container_name:
        # Container exec mode: pkill the server process inside the container.
        runtime = cfg.container_runtime or detect_container_runtime()
        try:
            kill_cmd = [
                runtime, "exec", cfg.container_name,
                "pkill", "-f", "python mcp_server.py"
            ]
            subprocess.run(kill_cmd, capture_output=True, timeout=5)
            print("  Server stopped.")
        except subprocess.TimeoutExpired:
            print("  Warning: Timeout while stopping server")
        except Exception:
            pass
    # Local subprocess stdio: the stdio_client context manager terminates the
    # child process on exit, so no explicit kill is needed here.

