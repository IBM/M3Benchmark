"""
Leaderboard data layer.

All agent data is read from / written to ``leaderboard_data.json``.
Score keys use pipe-delimited paths that mirror the column hierarchy:

    "Single Turn|API Styles|Business Intelligence": 90.0

This keeps the JSON simple and avoids tuple-key workarounds.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Column hierarchy  (turn_type, task_type, metric)
# ---------------------------------------------------------------------------
COLUMN_HIERARCHY = [
    ("Single Turn", "API Styles", "Business Intelligence"),
    ("Single Turn", "API Styles", "Dashboard APIs"),
    ("Single Turn", "Reasoning", "Multi-hop"),
    ("Single Turn", "Reasoning", "Policy Adherence"),
    ("Multi-turn", "Joint Structured/ Unstructured Reasoning", "Multi-hop"),
]

DATA_FILE = Path(__file__).parent / "leaderboard_data.json"


def _score_key(turn: str, task: str, metric: str) -> str:
    """Build the pipe-delimited key used in JSON score dicts."""
    return f"{turn}|{task}|{metric}"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------
def load_agents() -> list:
    """Load agents from the JSON file."""
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save_agents(agents: list):
    """Persist agents list to JSON."""
    with open(DATA_FILE, "w") as f:
        json.dump(agents, f, indent=2)


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------
def build_dataframe(agents: Optional[list] = None) -> pd.DataFrame:
    """Build a MultiIndex DataFrame from agent dicts.

    Columns: Rank | Agent | Model | <metrics...> | Overall
    """
    if agents is None:
        agents = load_agents()

    rows = []
    for entry in agents:
        row = {
            "Agent": entry.get("agent", ""),
            "Model": entry.get("model", ""),
        }
        scores = entry.get("scores", {})

        for turn, task, metric in COLUMN_HIERARCHY:
            key = _score_key(turn, task, metric)
            row[(turn, task, metric)] = scores.get(key)

        # Overall = mean of non-null metric scores
        metric_vals = [
            v for k, v in row.items()
            if k not in ("Agent", "Model") and v is not None and isinstance(v, (int, float))
        ]
        row[("", "", "Overall")] = round(sum(metric_vals) / len(metric_vals), 2) if metric_vals else 0.0
        rows.append(row)

    # Sort by Overall descending
    rows.sort(key=lambda r: r.get(("", "", "Overall"), 0), reverse=True)

    # Assign ranks
    for i, row in enumerate(rows, 1):
        row[("", "", "Rank")] = i
        row[("", "", "Agent")] = row.pop("Agent")
        row[("", "", "Model")] = row.pop("Model")

    # MultiIndex columns
    mi_cols = [("", "", "Rank"), ("", "", "Agent"), ("", "", "Model")]
    mi_cols += [(t, task, m) for t, task, m in COLUMN_HIERARCHY]
    mi_cols += [("", "", "Overall")]

    data = [[row.get(col) for col in mi_cols] for row in rows]
    columns = pd.MultiIndex.from_tuples(mi_cols, names=["Turn Type", "Task Type", "Metric"])
    return pd.DataFrame(data, columns=columns)
