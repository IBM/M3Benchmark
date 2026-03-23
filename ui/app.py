"""
VAKRA Leaderboard — FastAPI backend

Run locally:
    uvicorn app:app --reload --port 8000

The frontend (ui.html) is served at / and fetches leaderboard data from /api/leaderboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from data import load_agents, save_agents, compute_rows, AgentEntry

app = FastAPI(title="VAKRA Leaderboard")

STATIC_DIR = Path(__file__).parent

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(STATIC_DIR / "ui.html")


@app.get("/api/leaderboard")
def get_leaderboard():
    """Return leaderboard rows sorted by overall score descending."""
    agents = load_agents()
    return compute_rows(agents)


@app.post("/api/leaderboard", status_code=201)
def add_agent(entry: AgentEntry):
    """Add a new agent entry and persist it."""
    agents = load_agents()
    agents.append(entry.model_dump())
    save_agents(agents)
    return compute_rows(agents)
