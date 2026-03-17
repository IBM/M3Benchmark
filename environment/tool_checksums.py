"""Shared tool checksum utilities for MCP server/client integrity checks.

Used by:
  - MCP servers (server-side): verify the generated tool list matches the
    stored checksum for (capability_id, domain) after building tools_cache.
  - Benchmark client (client-side): verify the received tools match the
    expected checksum for the intended (capability_id, domain).

Checksums are stored in tool_checksums.json at the project root and committed
to the repository. Run generate_checksums.py to regenerate after intentional
tool changes.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Set MCP_VERIFY_CHECKSUMS=1 (or "true"/"yes") to enable checksum verification.
# Disabled by default so that tool_checksums.json does not need to be populated
# before running benchmarks.
_TRUTHY = {"1", "true", "yes", "on"}


def _verification_enabled() -> bool:
    return os.getenv("MCP_VERIFY_CHECKSUMS", "").lower().strip() in _TRUTHY

# Default path: project_root/tool_checksums.json
# Resolves correctly both locally (environment/ → project root) and inside the
# container (/app/environment/ → /app/).
_DEFAULT_CHECKSUMS_PATH = Path(__file__).parent.parent / "tool_checksums.json"


def compute_tool_checksum(tools: List[Any]) -> str:
    """Compute a stable SHA-256 checksum for a list of MCP tools.

    Tools are sorted by name so insertion order doesn't affect the result.
    Each entry encodes the tool's name and inputSchema.

    Args:
        tools: List of MCP Tool objects (with ``.name`` / ``.inputSchema``
               attributes) or plain dicts with ``"name"`` / ``"inputSchema"``
               keys.

    Returns:
        64-character lowercase hex string (SHA-256).
    """
    entries = []
    for tool in sorted(
        tools,
        key=lambda t: t.name if hasattr(t, "name") else t["name"],
    ):
        name = tool.name if hasattr(tool, "name") else tool["name"]
        schema = (
            tool.inputSchema
            if hasattr(tool, "inputSchema")
            else tool.get("inputSchema", {})
        )
        # Pydantic v2 → plain dict
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        elif hasattr(schema, "dict"):
            schema = schema.dict()  # type: ignore[union-attr]
        entries.append({"name": name, "inputSchema": schema})

    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_checksums(
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    """Load tool checksums from the JSON file.

    Args:
        path: Override path to ``tool_checksums.json``. Defaults to the file
              at the project root.

    Returns:
        Dict mapping ``str(capability_id)`` → ``{domain: checksum_hex}``.
        Returns an empty dict if the file does not exist.
    """
    resolved = path or _DEFAULT_CHECKSUMS_PATH
    try:
        with open(resolved) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug("tool_checksums.json not found at %s", resolved)
        return {}


def verify_checksum(
    capability_id: int,
    domain: str,
    tools: List[Any],
    checksums_path: Optional[Path] = None,
) -> None:
    """Verify that ``tools`` match the stored checksum for (capability_id, domain).

    Behaviour:
    - If no checksum is registered for this (capability_id, domain) pair, a warning
      is logged and the function returns without error. This avoids blocking
      initial setup before ``generate_checksums.py`` has been run.
    - If a checksum IS registered and the computed value does not match, a
      ``ValueError`` is raised immediately (hard error).

    Args:
        capability_id:  Benchmark capability ID (e.g. 2, 3, 4).
        domain:         Domain name (e.g. ``"address"``, ``"airline"``).
        tools:          List of MCP Tool objects returned by the server.
        checksums_path: Override path to ``tool_checksums.json``.

    Raises:
        ValueError: Checksum mismatch — wrong domain was used or tool
                    definitions changed without regenerating checksums.
    """
    if not _verification_enabled():
        return

    checksums = load_checksums(checksums_path)

    task_key = str(capability_id)
    if task_key not in checksums:
        logger.warning(
            "No checksums registered for capability %s — skipping verification. "
            "Run generate_checksums.py to create them.",
            capability_id,
        )
        return

    domain_checksums = checksums[task_key]
    if domain not in domain_checksums:
        logger.warning(
            "No checksum registered for capability %s, domain '%s' — "
            "skipping verification. Run generate_checksums.py to add it.",
            capability_id,
            domain,
        )
        return

    expected = domain_checksums[domain]
    actual = compute_tool_checksum(tools)

    if actual != expected:
        raise ValueError(
            f"Tool checksum mismatch for capability {capability_id}, domain '{domain}'.\n"
            f"  Expected : {expected}\n"
            f"  Got      : {actual}\n"
            "\nThis likely means MCP_DOMAIN is set to the wrong domain, or "
            "the server's tool definitions have changed unexpectedly.\n"
            "If tools were intentionally changed, re-run generate_checksums.py "
            "and commit the updated tool_checksums.json."
        )

    logger.debug(
        "Tool checksum OK for capability %s, domain '%s' (%s)",
        capability_id,
        domain,
        actual[:12],
    )
