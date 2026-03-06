import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from benchmark.utils import _extract_tool_response_values


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
    3: os.environ.get(
        "TASK_3_DIR",
        str(Path(__file__).parent.parent / "data" / "tasks" / "task_3"),
    ),
    5: os.environ.get(
        "TASK_5_DIR",
        str(Path(__file__).parent.parent / "data" / "tasks" / "task_5"),
    ),
}

@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str

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
    context: Optional[List[Message]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkItem":
        """Create BenchmarkItem from JSON dict."""
        dialogue = data.get("dialogue", {})
        turns = dialogue.get("turns", [])
        context = []
        # Get first turn's query — support both dialogue.turns and ground_truth formats
        if turns:
            if len(turns) > 1:
                for turn in turns:
                    context.append(Message(role="user", content=turn["query"]))

                    answer = turn.get("answer")
                    if answer:
                        context.append(Message(role="assistant", content=str(answer)))
                query = turns[-1]["query"]
                turn_id = turns[-1].get("turn_id", 0)
            else:
                query = turns[-1]["query"]
                turn_id = turns[-1].get("turn_id", 0)

        else:
            ground_truth = data.get("ground_truth", [])
            query = ground_truth[0]["query"] if ground_truth else ""
            turn_id = ground_truth[0].get("turn_id", 0) if ground_truth else 0

        return cls(
            uuid=data.get("uuid", ""),
            domain=data.get("domain", ""),
            query=query,
            num_turns=data.get("num_turns", 1),
            tools=data.get("tools", []),
            additional_instructions=data.get("additional_instructions", ""),
            turn_id=turn_id,
            context=context if len(context) !=0 else None
        )


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark item."""
    uuid: str
    domain: str
    query: str
    answer: str = ""
    context: Optional[List[dict]] = None
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
        print(f"Error: No JSON files found under {input_path}/input/")
        sys.exit(1)

    if domains:
        json_files = [f for f in json_files if f.stem in domains]
        if not json_files:
            available = sorted(
                {f.stem for f in input_path.glob("input/*.json")}
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


def log_trajectory(result: "BenchmarkResult", tlog: Callable = print) -> None:
    """Print a human-readable summary of a result's trajectory to stdout."""
    if not result.trajectory:
        return
    traj_len = len(result.trajectory)
    tlog(f"    Trajectory ({traj_len} steps):")
    for i, step in enumerate(result.trajectory):
        step_type = step.get("type", "unknown")
        if step_type == "HumanMessage":
            content = step.get("content", "")
            suffix = "..." if len(content) > 80 else ""
            tlog(f"      [{i+1}] User: {content[:80]}{suffix}")
        elif step_type == "AIMessage":
            content = step.get("content", "")
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                tlog(f"      [{i+1}] AI: Calling {len(tool_calls)} tool(s)")
                for tc in tool_calls:
                    tlog(f"          - {tc.get('name', 'unknown')}({tc.get('args', {})})")
            else:
                suffix = "..." if len(content) > 80 else ""
                tlog(f"      [{i+1}] AI: {content[:80]}{suffix}")
        elif step_type == "ToolMessage":
            tool_name = step.get("tool_name", "unknown")
            result_str = str(step.get("result", ""))
            suffix = "..." if len(result_str) > 80 else ""
            tlog(f"      [{i+1}] Tool ({tool_name}): {result_str[:80]}{suffix}")


def log_message_history(result: "BenchmarkResult", tlog: Callable = print) -> None:
    """Print the full message history for a completed benchmark result.

    Includes system prompts, user messages, tool calls with arguments,
    tool results, and the final answer.
    """
    traj = result.trajectory
    if not traj:
        return

    tlog("\n    " + "-" * 56)
    tlog("    MESSAGE HISTORY")
    tlog("    " + "-" * 56)

    for entry in traj:
        msg_type = entry.get("type", "Unknown")
        content = entry.get("content", "")

        if msg_type == "SystemMessage":
            preview = content[:300] + ("..." if len(content) > 300 else "")
            tlog(f"\n    [SYSTEM]\n      {preview!r}")

        elif msg_type == "HumanMessage":
            tlog(f"\n    [USER]\n      {content}")

        elif msg_type == "AIMessage":
            tool_calls = entry.get("tool_calls", [])
            if content:
                tlog(f"\n    [ASSISTANT]\n      {content}")
            for tc in tool_calls:
                args_str = json.dumps(
                    tc.get("args", {}), ensure_ascii=False
                )
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                tlog(f"\n    [TOOL CALL] {tc.get('name', '?')}({args_str})")

        elif msg_type == "ToolMessage":
            tool_name = entry.get("tool_name", "?")
            res = str(entry.get("result", ""))
            res_preview = res[:300] + ("..." if len(res) > 300 else "")
            tlog(f"\n    [TOOL RESULT] {tool_name}\n      {res_preview}")

    answer = result.answer or "(empty)"
    tlog(f"\n    [FINAL ANSWER]\n      {answer}")
    tlog("    " + "-" * 56)


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
            tool_call_list = []
            # Internal LangChain parameters that leak into tool schemas
            _INTERNAL_KEYS = {"args", "config", "kwargs"}
            for tc in r.tool_calls:
                raw_args = tc.get("arguments", {})
                filtered_args = {
                    k: v for k, v in raw_args.items()
                    if k not in _INTERNAL_KEYS
                }
                tool_call_list.append({
                        "name": tc.get("tool_name", ""),
                        "arguments": filtered_args})

            record = {
                "uuid": r.uuid,
                "domain": r.domain,
                "status": r.status,
                "error": r.error,
                "duration_s": r.duration_s,
                "output": [
                    {
                        "turn_id": r.turn_id,
                        "query": r.query,
                        "answer": r.answer,
                        "sequence": {
                            "tool_call": tool_call_list
                        }, 
                    }
                ],
            }
            records.append(record)

        output_file = output_dir / f"{domain}.json"
        with open(output_file, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Ground truth results saved to: {output_file}")


class TaskLogger:
    """Tee-style logger for a single task run.

    Writes every message to both stdout (prefixed with ``[task_N]`` for easy
    identification in parallel mode) and a per-task ``run.log`` file in the
    output directory (with a millisecond timestamp for post-hoc analysis).

    Usage::

        tlog = TaskLogger(task_id=2, log_path=out_dir / "run.log")
        tlog("Starting domain: hockey")
        tlog.close()

    It is a drop-in replacement for ``print()`` — ``tlog(msg)`` behaves
    identically to ``print(msg)`` except for the side-effects above.
    """

    def __init__(self, task_id: int, log_path: Path) -> None:
        self._prefix = f"[task_{task_id}] "
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("w", buffering=1)  # line-buffered

    def __call__(self, *args, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        end = kwargs.get("end", "\n")
        # stdout: prefix lets you distinguish tasks when running in parallel
        print(f"{self._prefix}{msg}", end=end, flush=True)
        # log file: strip the prefix and add a wall-clock timestamp
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._file.write(f"[{ts}] {msg}{end}")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
