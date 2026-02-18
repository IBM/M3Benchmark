import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Task configurations - maps task_id to input directory path
TASK_PATHS = {
    1: os.environ.get(
        "TASK_1_DIR",
        str(Path(__file__).parent.parent / "data" / "tasks" / "task_1"),
    ),
    2: os.environ.get(
        "TASK_2_DIR",
        str(Path(__file__).parent.parent / "data" / "tasks" / "task_2"),
    ),
    5: os.environ.get(
        "TASK_5_DIR",
        str(Path(__file__).parent.parent / "data" / "tasks" / "task_5"),
    ),
}


@dataclass
class BenchmarkItem:
    """A single benchmark test case."""
    uuid: str
    domain: str
    query: str
    num_turns: int
    tools: List[Dict[str, Any]]
    additional_instructions: str = ""
    turn_id: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkItem":
        """Create BenchmarkItem from JSON dict."""
        dialogue = data.get("dialogue", {})
        turns = dialogue.get("turns", [])
        # Get first turn's query (for now, single turn support)
        query = turns[0]["query"] if turns else ""
        turn_id = turns[0].get("turn_id", 0) if turns else 0

        return cls(
            uuid=data.get("uuid", ""),
            domain=data.get("domain", ""),
            query=query,
            num_turns=data.get("num_turns", 1),
            tools=data.get("tools", []),
            additional_instructions=data.get("additional_instructions", ""),
            turn_id=turn_id,
        )


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark item."""
    uuid: str
    domain: str
    query: str
    answer: str = ""
    tool_calls: List[Dict] = field(default_factory=list)
    trajectory: List[Dict] = field(default_factory=list)  # Agent trajectory
    turn_id: int = 0
    status: str = "pending"
    error: str = ""
    duration_s: float = 0.0


def load_benchmark_data(
    task_id: int,
    domains: Optional[List[str]] = None,
    domain_names_only: bool = False,
) -> Tuple[List[BenchmarkItem], List[str]]:
    """Load all benchmark items for a task, optionally filtered by domain.

    Searches <task_dir>/*/input/ for *.json files (one file per domain,
    named <domain>.json). Output directories are ignored.

    Args:
        task_id: Task ID to load data for.
        domains: Optional list of domain names to filter by.
        domain_names_only: If True, skip reading JSON files and return only
            the list of domain names found (items list will be empty).

    Returns:
        Tuple of (items, domain_names) where items is a flat list of
        BenchmarkItem objects (empty when domain_names_only=True) and
        domain_names is a sorted list of domain name strings.
    """
    if task_id not in TASK_PATHS:
        print(f"Error: Unknown task_id {task_id}")
        sys.exit(1)

    input_path = Path(TASK_PATHS[task_id])
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    json_files = sorted(input_path.glob("input/*.json"))
    if not json_files:
        print(f"Error: No JSON files found under {input_path}/*/input/")
        sys.exit(1)

    if domains:
        json_files = [f for f in json_files if f.stem in domains]
        if not json_files:
            available = sorted(
                {f.stem for f in input_path.glob("*/input/*.json")}
            )
            print(f"Error: No files found for domains: {domains}")
            suffix = "..." if len(available) > 10 else ""
            print(f"Available domains: {available[:10]}{suffix}")
            sys.exit(1)

    domain_names = sorted({f.stem for f in json_files})

    if domain_names_only:
        return [], domain_names

    items: List[BenchmarkItem] = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        for item_data in data:
            items.append(BenchmarkItem.from_dict(item_data))
    return items, domain_names


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


def make_output_dir(task_id: int, output_dir: Optional[str] = None) -> Path:
    """Create a timestamped output directory for a task under CWD.

    Format: output/task_{id}_{Mon}_{dd}_{hh}_{mm}{am|pm}/
    e.g.    output/task_5_Feb_13_11_21am/
    """
    if output_dir:
        p = Path(output_dir)
    else:
        now = datetime.now()
        ts = now.strftime("%b_%d_%I_%M%p").lower()  # e.g. feb_13_11_21am
        p = Path("output") / f"task_{task_id}_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_trajectory(result: "BenchmarkResult") -> None:
    """Print a human-readable summary of a result's trajectory to stdout."""
    if not result.trajectory:
        return
    traj_len = len(result.trajectory)
    print(f"    Trajectory ({traj_len} steps):")
    for i, step in enumerate(result.trajectory):
        step_type = step.get("type", "unknown")
        if step_type == "HumanMessage":
            content = step.get("content", "")
            suffix = "..." if len(content) > 80 else ""
            print(f"      [{i+1}] User: {content[:80]}{suffix}")
        elif step_type == "AIMessage":
            content = step.get("content", "")
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                print(f"      [{i+1}] AI: Calling {len(tool_calls)} tool(s)")
                for tc in tool_calls:
                    print(f"          - {tc.get('name', 'unknown')}({tc.get('args', {})})")
            else:
                suffix = "..." if len(content) > 80 else ""
                print(f"      [{i+1}] AI: {content[:80]}{suffix}")
        elif step_type == "ToolMessage":
            tool_name = step.get("tool_name", "unknown")
            result_str = str(step.get("result", ""))
            suffix = "..." if len(result_str) > 80 else ""
            print(f"      [{i+1}] Tool ({tool_name}): {result_str[:80]}{suffix}")


def save_results_ground_truth(
    results: List[BenchmarkResult], output_dir: Path
):
    """Save benchmark results in ground truth format to per-domain files.

    Writes one file per domain to output_dir/<domain>.json matching the
    structure of the example output files (uuid, domain, ground_truth with
    turn_id, query, answer, and gold_sequence).
    """
    # Group results by domain
    by_domain: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.domain not in by_domain:
            by_domain[r.domain] = []
        by_domain[r.domain].append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    for domain, domain_results in by_domain.items():
        records = []
        for r in domain_results:
            # Build gold_sequence from tool_calls.
            # Each tool call becomes its own entry so retries are preserved.
            gold_sequence = []
            # Internal LangChain parameters that leak into tool schemas
            _INTERNAL_KEYS = {"args", "config", "kwargs"}
            for tc in r.tool_calls:
                raw_args = tc.get("arguments", {})
                filtered_args = {
                    k: v for k, v in raw_args.items()
                    if k not in _INTERNAL_KEYS
                }
                gold_sequence.append({
                    "tool_call": [[{
                        "name": tc.get("tool_name", ""),
                        "arguments": filtered_args,
                    }]],
                    "tool_response": [
                        _extract_tool_response_values(tc.get("result", ""))
                    ],
                })

            record = {
                "uuid": r.uuid,
                "domain": r.domain,
                "status": r.status,
                "error": r.error,
                "duration_s": r.duration_s,
                "ground_truth": [
                    {
                        "turn_id": r.turn_id,
                        "query": r.query,
                        "answer": r.answer,
                        "gold_sequence": gold_sequence,
                    }
                ],
            }
            records.append(record)

        output_file = output_dir / f"{domain}.json"
        with open(output_file, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Ground truth results saved to: {output_file}")


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

