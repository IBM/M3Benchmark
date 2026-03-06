"""Shared MCP logging utilities.

All MCP servers write JSON-lines to stderr (stdout is reserved for the MCP
protocol).  Each record carries ``task_id`` and ``domain`` so that logs from
parallel benchmark runs can be filtered with ``jq``.
"""
import json
import logging
import os
import sys


class _JsonFormatter(logging.Formatter):
    """Single-line JSON log records including TASK_ID and MCP_DOMAIN context."""

    def __init__(self) -> None:
        super().__init__()
        self._task_id = os.environ.get("TASK_ID", "")
        self._domain = os.environ.get("MCP_DOMAIN", "")

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "task_id": self._task_id,
                "domain": self._domain,
                "logger": record.name,
                "msg": record.getMessage(),
            },
            ensure_ascii=False,
        )


def _setup_mcp_logging() -> None:
    """Route all logging to stderr as JSON lines (stdout is reserved for MCP)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
