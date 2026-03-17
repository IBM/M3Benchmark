"""
Submission schema for benchmark output files.

These Pydantic models define the required format for result JSON files.
Constructing an OutputRecord with wrong types or missing fields raises
ValidationError immediately — catching schema mistakes before the file
is written.

Expected output structure (one record per benchmark sample):

    [
      {
        "uuid":       "<string>",
        "domain":     "<string>",
        "status":     "success" | "error",
        "error":      "<string>",
        "duration_s": <float>,
        "output": [
          {
            "turn_id":  <int>,
            "query":    "<string>",
            "answer":   "<string>",
            "sequence": {
              "tool_call": [
                {"name": "<string>", "arguments": {<object>}}
              ]
            }
          }
        ]
      }
    ]
"""

from typing import Any, Literal

from pydantic import BaseModel


class ToolCall(BaseModel):
    """A single tool invocation made by the agent."""
    name: str
    arguments: dict[str, Any]


class Sequence(BaseModel):
    """All tool calls the agent made while answering one turn."""
    tool_call: list[ToolCall]


class Turn(BaseModel):
    """One question-answer turn within a benchmark sample."""
    turn_id: int
    query: str
    answer: str
    sequence: Sequence


class OutputRecord(BaseModel):
    """Top-level record for one benchmark sample."""
    uuid: str
    domain: str
    status: Literal["success", "error"]
    error: str
    duration_s: float
    output: list[Turn]
