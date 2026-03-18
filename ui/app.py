"""
Agent Performance Leaderboard — Streamlit UI

Run:
    streamlit run ui/app.py
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from data import (
    COLUMN_HIERARCHY,
    build_dataframe,
    load_agents,
    save_agents,
    _score_key,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agent Performance Leaderboard",
    page_icon="🏆",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — explicit light backgrounds so dark-mode can't bleed through
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; }

/* ---- score colours ---- */
.score-excellent { color: #16a34a; font-weight: 600; }
.score-good      { color: #6366f1; font-weight: 600; }
.score-fair      { color: #ca8a04; font-weight: 600; }
.score-poor      { color: #dc2626; font-weight: 600; }

/* ---- rank badges ---- */
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 50%;
    font-weight: 700; font-size: 14px;
}
.rank-1 { background: #fef3c7; color: #92400e; }
.rank-2 { background: #e5e7eb; color: #374151; }
.rank-3 { background: #fed7aa; color: #9a3412; }
.rank-default { background: #f3f4f6; color: #6b7280; }

/* ---- table ---- */
.leaderboard-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    overflow: hidden;
    font-size: 14px;
    background: #ffffff;          /* explicit white base */
}
.leaderboard-table th,
.leaderboard-table td {
    padding: 14px 18px;
    text-align: center;
    border-bottom: 1px solid #e5e7eb;
    background: #ffffff;          /* every cell is white */
    color: #1e293b;
}
.leaderboard-table tr:last-child td { border-bottom: none; }
.leaderboard-table tr:hover td { background: #f1f5f9 !important; }

/* ---- group header: turn type ---- */
.group-header-turn th {
    background: #eef2ff !important;
    font-size: 14px; font-weight: 700; color: #4338ca;
    border-bottom: 2px solid #c7d2fe;
}

/* ---- group header: task type ---- */
.group-header-task th {
    background: #f5f3ff !important;
    font-size: 12px; font-weight: 600; color: #6d28d9;
}

/* ---- metric header row (sortable) ---- */
.metric-header th {
    background: #f8fafc !important;
    font-weight: 600; color: #475569; font-size: 13px;
    cursor: pointer; user-select: none;
    position: relative;
    white-space: nowrap;
}
.metric-header th:hover { background: #eef2ff !important; }
.sort-arrow { margin-left: 4px; font-size: 11px; opacity: 0.4; }
.sort-arrow.active { opacity: 1.0; color: #4338ca; }

/* ---- pills ---- */
.pill {
    display: inline-block; padding: 4px 16px;
    border-radius: 20px; font-size: 13px; font-weight: 600;
}
.pill-turn { background: #e0e7ff; color: #3730a3; }
.pill-task { background: #ede9fe; color: #5b21b6; }

/* ---- special columns ---- */
.agent-name {
    text-align: left !important; font-weight: 600;
}
.col-overall {
    background: #f9fafb !important; font-weight: 700;
}

/* ---- legend ---- */
.legend {
    display: flex; gap: 24px; align-items: center;
    padding: 12px 0; font-size: 13px; color: #64748b;
}
.legend-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score_class(val: float) -> str:
    if val is None:
        return ""
    if val >= 90:
        return "score-excellent"
    elif val >= 80:
        return "score-good"
    elif val >= 70:
        return "score-fair"
    return "score-poor"


def rank_class(rank: int) -> str:
    if rank <= 3:
        return f"rank-{rank}"
    return "rank-default"


def render_score(val) -> str:
    if val is None:
        return "—"
    return f'<span class="{score_class(val)}">{val:.3f}</span>'


def render_overall(val) -> str:
    if val is None:
        return "—"
    return f'<span class="{score_class(val)}" style="font-size:15px">{val:.3f}</span>'


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
TURN_TYPES = ["All"] + list(dict.fromkeys(t for t, _, _ in COLUMN_HIERARCHY))
ALL_TASKS = list(dict.fromkeys(task for _, task, _ in COLUMN_HIERARCHY))


def filter_hierarchy(turn_filter: str, task_filter: str):
    filtered = COLUMN_HIERARCHY[:]
    if turn_filter != "All":
        filtered = [(t, tk, m) for t, tk, m in filtered if t == turn_filter]
    if task_filter != "All Tasks":
        filtered = [(t, tk, m) for t, tk, m in filtered if tk == task_filter]
    return filtered


# ---------------------------------------------------------------------------
# Build sortable column index  (col 0 = Rank, 1 = Agent, 2..N = metrics, N+1 = Overall)
# ---------------------------------------------------------------------------
def _col_index_map(hierarchy):
    """Return {label: column_index} for the data columns in the HTML table."""
    mapping = {"Rank": 0, "Agent": 1, "Model": 2}
    for i, (_, _, metric) in enumerate(hierarchy):
        mapping[metric] = i + 3
    mapping["Overall"] = len(hierarchy) + 3
    return mapping


# ---------------------------------------------------------------------------
# Build HTML table
# ---------------------------------------------------------------------------
def build_leaderboard_html(df: pd.DataFrame, hierarchy: list) -> str:
    turn_groups = {}
    for turn, task, metric in hierarchy:
        turn_groups.setdefault(turn, []).append((task, metric))

    col_map = _col_index_map(hierarchy)
    overall_idx = col_map["Overall"]

    # Compute per-column max values
    col_maxes = {}
    for turn, task, metric in hierarchy:
        col_vals = df[(turn, task, metric)].dropna()
        if len(col_vals) > 0:
            col_maxes[(turn, task, metric)] = col_vals.max()
    overall_vals = df[("", "", "Overall")].dropna()
    overall_max = overall_vals.max() if len(overall_vals) > 0 else None

    # Inline styles — st.components.v1.html renders in its own iframe,
    # so we must embed CSS directly in the HTML fragment.
    html = """<style>
body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: transparent; }
.score-excellent { color: #16a34a; font-weight: 600; }
.score-good      { color: #6366f1; font-weight: 600; }
.score-fair      { color: #ca8a04; font-weight: 600; }
.score-poor      { color: #dc2626; font-weight: 600; }
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 50%;
    font-weight: 700; font-size: 14px;
}
.rank-1 { background: #fef3c7; color: #92400e; }
.rank-2 { background: #e5e7eb; color: #374151; }
.rank-3 { background: #fed7aa; color: #9a3412; }
.rank-default { background: #f3f4f6; color: #6b7280; }
.leaderboard-table {
    width: 100%; border-collapse: separate; border-spacing: 0;
    border: 1px solid #d1d5db; border-radius: 12px;
    overflow: hidden; font-size: 14px; background: #ffffff;
}
.leaderboard-table th, .leaderboard-table td {
    padding: 14px 18px; text-align: center;
    border-bottom: 1px solid #e5e7eb;
    background: #ffffff; color: #1e293b;
}
.leaderboard-table tr:last-child td { border-bottom: none; }
.leaderboard-table tbody tr:hover td { background: #f1f5f9 !important; }
.group-header-turn th {
    background: #eef2ff !important; font-size: 14px;
    font-weight: 700; color: #4338ca; border-bottom: 2px solid #c7d2fe;
}
.group-header-task th {
    background: #f5f3ff !important; font-size: 12px;
    font-weight: 600; color: #6d28d9;
}
.metric-header th {
    background: #f8fafc !important; font-weight: 600;
    color: #475569; font-size: 13px;
    cursor: pointer; user-select: none; white-space: nowrap;
}
.metric-header th:hover { background: #eef2ff !important; }
.sort-arrow { margin-left: 4px; font-size: 11px; opacity: 0.4; }
.sort-arrow.active { opacity: 1.0; color: #4338ca; }
.pill {
    display: inline-block; padding: 4px 16px;
    border-radius: 20px; font-size: 13px; font-weight: 600;
}
.pill-turn { background: #e0e7ff; color: #3730a3; }
.pill-task { background: #ede9fe; color: #5b21b6; }
.agent-name { text-align: left !important; font-weight: 600; }
.model-name { text-align: left !important; color: #64748b; font-size: 13px; }
.col-overall { background: #f9fafb !important; font-weight: 700; }
.col-max { background: #fef9c3 !important; outline: 2px solid #f59e0b; outline-offset: -2px; }
</style>
"""

    html += f'<table class="leaderboard-table" id="lb-table" data-sort-col="{overall_idx}" data-sort-dir="desc">\n'

    # ---- Row 1: turn-type headers ----
    html += '<thead>'
    html += '<tr class="group-header-turn">'
    html += '<th rowspan="3" style="width:60px">Rank</th>'
    html += '<th rowspan="3" style="min-width:120px">Agent</th>'
    html += '<th rowspan="3" style="min-width:120px">Model</th>'
    for turn, cols in turn_groups.items():
        html += f'<th colspan="{len(cols)}"><span class="pill pill-turn">{turn}</span></th>'
    html += f'<th rowspan="3" class="col-overall" style="min-width:90px; cursor:pointer;" '
    html += f'onclick="sortTable({overall_idx})">'
    html += f'Overall <span class="sort-arrow active" id="arrow-{overall_idx}">&#9660;</span></th>'
    html += "</tr>\n"

    # ---- Row 2: task-type headers ----
    html += '<tr class="group-header-task">'
    for turn, cols in turn_groups.items():
        task_groups = {}
        for task, metric in cols:
            task_groups.setdefault(task, []).append(metric)
        for task, metrics in task_groups.items():
            html += f'<th colspan="{len(metrics)}"><span class="pill pill-task">{task}</span></th>'
    html += "</tr>\n"

    # ---- Row 3: metric headers (clickable) ----
    html += '<tr class="metric-header">'
    for turn, cols in turn_groups.items():
        for task, metric in cols:
            idx = col_map[metric]
            html += (
                f'<th onclick="sortTable({idx})">'
                f'{metric} <span class="sort-arrow" id="arrow-{idx}">&#9660;</span></th>'
            )
    html += "</tr>\n"
    html += "</thead>\n"

    # ---- Data rows ----
    html += "<tbody>\n"
    for _, row in df.iterrows():
        rank = int(row[("", "", "Rank")])
        agent = row[("", "", "Agent")]
        model = row[("", "", "Model")] or ""
        overall = row[("", "", "Overall")]

        html += "<tr>"
        html += f'<td><span class="rank-badge {rank_class(rank)}">{rank}</span></td>'
        html += f'<td class="agent-name">{agent}</td>'
        html += f'<td class="model-name">{model}</td>'

        for turn, task, metric in hierarchy:
            val = row.get((turn, task, metric))
            max_val = col_maxes.get((turn, task, metric))
            is_max = val is not None and max_val is not None and val == max_val
            td_class = ' class="col-max"' if is_max else ''
            html += f"<td{td_class}>{render_score(val)}</td>"

        is_max_overall = overall is not None and overall_max is not None and overall == overall_max
        overall_cls = "col-overall col-max" if is_max_overall else "col-overall"
        html += f'<td class="{overall_cls}">{render_overall(overall)}</td>'
        html += "</tr>\n"

    html += "</tbody>\n</table>"

    # ---- JavaScript: client-side column sorting ----
    html += """
<script>
function sortTable(colIdx) {
    const table = document.getElementById("lb-table");
    const tbody = table.querySelector("tbody");
    const rows  = Array.from(tbody.querySelectorAll("tr"));

    // Determine current sort state
    const curCol = parseInt(table.dataset.sortCol);
    let   curDir = table.dataset.sortDir;

    // Toggle direction if same column, else default desc for numbers
    let newDir;
    if (colIdx === curCol) {
        newDir = (curDir === "desc") ? "asc" : "desc";
    } else {
        newDir = "desc";
    }
    table.dataset.sortCol = colIdx;
    table.dataset.sortDir = newDir;

    // Sort rows
    rows.sort(function(a, b) {
        let aText = a.cells[colIdx].innerText.trim();
        let bText = b.cells[colIdx].innerText.trim();

        // Try numeric parse
        let aNum = parseFloat(aText);
        let bNum = parseFloat(bText);

        let cmp;
        if (!isNaN(aNum) && !isNaN(bNum)) {
            cmp = aNum - bNum;
        } else {
            cmp = aText.localeCompare(bText);
        }
        return (newDir === "asc") ? cmp : -cmp;
    });

    // Re-insert rows and update rank badges
    rows.forEach(function(row, i) {
        tbody.appendChild(row);
        // Update the rank cell (first td)
        const rankSpan = row.cells[0].querySelector(".rank-badge");
        if (rankSpan) {
            const newRank = i + 1;
            rankSpan.textContent = newRank;
            rankSpan.className = "rank-badge " + (
                newRank === 1 ? "rank-1" :
                newRank === 2 ? "rank-2" :
                newRank === 3 ? "rank-3" : "rank-default"
            );
        }
    });

    // Update arrow indicators
    document.querySelectorAll("#lb-table .sort-arrow").forEach(function(el) {
        el.classList.remove("active");
        el.innerHTML = "&#9660;";
    });
    const activeArrow = document.getElementById("arrow-" + colIdx);
    if (activeArrow) {
        activeArrow.classList.add("active");
        activeArrow.innerHTML = (newDir === "asc") ? "&#9650;" : "&#9660;";
    }
}
</script>
"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Header
    col_title, col_btn = st.columns([6, 1])
    with col_title:
        st.markdown("# Agent Performance Leaderboard")
        st.caption("Compare agent performance across different task types")
    with col_btn:
        st.write("")
        add_clicked = st.button("+ Agent", type="primary")

    # Filters
    col_turn, col_task, _ = st.columns([1, 2, 5])
    with col_turn:
        turn_filter = st.selectbox("Turn Type", TURN_TYPES, index=0)
    with col_task:
        task_options = ["All Tasks"] + ALL_TASKS
        task_filter = st.selectbox("Task Type", task_options, index=0)

    # Build data
    agents = load_agents()
    df = build_dataframe(agents)

    # Filter columns
    filtered_hierarchy = filter_hierarchy(turn_filter, task_filter)

    # Render table
    html = build_leaderboard_html(df, filtered_hierarchy)
    st.components.v1.html(html, height=max(350, 120 + len(agents) * 65), scrolling=True)

    # Legend
    st.markdown(
        """
    <div class="legend" style="margin-top: 8px;">
        <span><span class="legend-dot" style="background:#16a34a"></span> 90-100: Excellent</span>
        <span><span class="legend-dot" style="background:#6366f1"></span> 80-89: Good</span>
        <span><span class="legend-dot" style="background:#ca8a04"></span> 70-79: Fair</span>
        <span><span class="legend-dot" style="background:#dc2626"></span> &lt;70: Needs Improvement</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---- Add Agent dialog ----
    if add_clicked:
        st.session_state["show_add_form"] = True

    if st.session_state.get("show_add_form"):
        render_add_agent_form(agents)

    # ---- Upload results ----
    st.markdown("---")
    st.subheader("Upload Benchmark Results")
    uploaded = st.file_uploader(
        "Upload a benchmark summary JSON to add an agent",
        type=["json"],
    )
    if uploaded:
        try:
            data = json.load(uploaded)
            st.json(data)
            st.info("Parsing uploaded results — integrate with your evaluation pipeline to auto-populate scores.")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")


def render_add_agent_form(agents: list):
    st.markdown("---")
    st.subheader("Add New Agent")

    col_name, col_model = st.columns(2)
    with col_name:
        agent_name = st.text_input("Agent Name", placeholder="e.g. My Agent v2")
    with col_model:
        model_name = st.text_input("Model", placeholder="e.g. GPT-4o")

    cols = st.columns(len(COLUMN_HIERARCHY))
    scores = {}
    for i, (turn, task, metric) in enumerate(COLUMN_HIERARCHY):
        with cols[i]:
            label = f"{metric}"
            if turn == "Multi-turn":
                label += " (MT)"
            val = st.number_input(label, min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"add_{i}")
            scores[_score_key(turn, task, metric)] = val

    col_save, col_cancel, _ = st.columns([1, 1, 4])
    with col_save:
        if st.button("Save Agent", type="primary"):
            if agent_name.strip():
                agents.append({"agent": agent_name.strip(), "model": model_name.strip(), "scores": scores})
                save_agents(agents)
                st.session_state["show_add_form"] = False
                st.rerun()
            else:
                st.warning("Please enter an agent name.")
    with col_cancel:
        if st.button("Cancel"):
            st.session_state["show_add_form"] = False
            st.rerun()


if __name__ == "__main__":
    main()
