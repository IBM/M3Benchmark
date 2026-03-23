"""
Leaderboard data layer.

Agents are stored in leaderboard_data.json. Each entry has:
  {
    "model":     "GPT-OSS-120B",
    "agent":     "ReAct (Prompt)",
    "agent_url": "https://...",
    "date":      "Mar 24, 2026",
    "scores": {
      "api_chaining":      0.0,
      "tool_selection":     50.46,
      "multihop": 27.14,
      "multiturn": 40.00
    }
  }

Scores are on a 0-100 scale. Overall is computed as the mean of the four scores.
"""

import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

DATA_FILE = Path(__file__).parent / "leaderboard_data.json"

SCORE_KEYS = ["api_chaining", "tool_selection", "multihop", "multiturn"]


class Scores(BaseModel):
    api_chaining: float = 0.0
    tool_selection: float = 0.0
    multihop: float = 0.0
    multiturn: float = 0.0


class AgentEntry(BaseModel):
    model: str
    agent: str = "ReAct (Prompt)"
    agent_url: str = "https://github.com/IBM/vakra"
    date: str = ""
    scores: Scores


def load_agents() -> list:
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save_agents(agents: list):
    with open(DATA_FILE, "w") as f:
        json.dump(agents, f, indent=2)


def compute_rows(agents: list) -> list:
    """Return leaderboard rows sorted by overall descending, with rank assigned."""
    rows = []
    for entry in agents:
        s = entry.get("scores", {})
        vals = [s.get(k, 0.0) for k in SCORE_KEYS]
        overall = round(sum(vals) / len(vals), 2) if vals else 0.0
        rows.append({
            "model":     entry.get("model", ""),
            "agent":     entry.get("agent", ""),
            "agent_url": entry.get("agent_url", "https://github.com/IBM/vakra"),
            "date":      entry.get("date", ""),
            "scores": {k: s.get(k, 0.0) for k in SCORE_KEYS},
            "overall":   overall,
        })

    rows.sort(key=lambda r: r["overall"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i

    return rows
