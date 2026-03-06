import json
import logging
import shutil
import subprocess
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _extract_tool_response_values(result_str: str):
    """Extract only the values from a tool response JSON string.

    Tool responses come as JSON dicts like '{"description": "Foo"}' or
    '{"codes": []}'. This extracts just the values ("Foo" or []) so the
    output contains the data without the key names.
    """
    try:
        parsed = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return result_str

    if isinstance(parsed, dict):
        values = list(parsed.values())
        if len(values) == 1:
            return values[0]
        return values

    # Already a plain value (list, int, string, etc.)
    return parsed


def generate_openapi_spec(
    all_tools_by_domain: Dict[str, List[Dict[str, Any]]],
    task_id: int,
) -> Dict[str, Any]:
    """Build an OpenAPI-like spec dict from per-domain tool info."""
    spec: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {
            "title": "MCP Tools Specification",
            "version": "1.0.0",
            "description": f"Tools available for task {task_id}",
        },
        "paths": {},
        "components": {"schemas": {}},
    }
    for domain, tools in all_tools_by_domain.items():
        for tool in tools:
            path = f"/v1/{domain}/{tool['name']}"
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            parameters = [
                {
                    "name": param_name,
                    "in": "query",
                    "required": param_name in required,
                    "schema": {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", ""),
                    },
                }
                for param_name, param_info in properties.items()
            ]
            spec["paths"][path] = {
                "get": {
                    "summary": tool["description"],
                    "operationId": tool["name"],
                    "parameters": parameters,
                    "responses": {"200": {"description": "Successful response"}},
                }
            }
    return spec


def _runtime_works(name: str) -> bool:
    """Return True if `name --version` exits successfully."""
    try:
        result = subprocess.run(
            [name, "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _assert_container_running(runtime: str, container_name: str) -> None:
    """Check that container_name exists and is running; raise RuntimeError if not."""
    try:
        result = subprocess.run(
            [runtime, "inspect", "--format", "{{.State.Status}}", container_name],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Container runtime {runtime!r} not found or not executable. "
            "If 'docker' is a symlink to podman, set "
            "'container_runtime: podman' in your MCP connection config."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Container {container_name!r} not found. "
            f"Start it with: {runtime} start {container_name}\n"
            f"({runtime} inspect output: {result.stderr.strip()})"
        )
    status = result.stdout.strip()
    if status != "running":
        raise RuntimeError(
            f"Container {container_name!r} exists but is not running "
            f"(status: {status!r}). "
            f"Start it with: {runtime} start {container_name}"
        )


def detect_container_runtime() -> str:
    """
    Detect available container runtime (podman or docker).
    Returns 'podman' if available and working, otherwise 'docker'.
    Raises RuntimeError if neither runtime works.
    """
    for name in ("podman", "docker"):
        if shutil.which(name) and _runtime_works(name):
            if name == "docker":
                logger.warning("podman not found or not working, using docker instead")
            return name

    raise RuntimeError(
        "Neither podman nor docker found in PATH. "
        "Please install one of them."
    )
