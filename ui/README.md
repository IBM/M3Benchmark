---
title: VAKRA Leaderboard
emoji: 🏆
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# VAKRA Leaderboard UI

FastAPI backend + static HTML frontend for the VAKRA agent leaderboard.
The frontend (`ui.html`) fetches live data from `/api/leaderboard`; no data is hardcoded in the HTML.

## Quick Start (local, no Docker)

```bash
cd ui
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Opens at http://localhost:8000.

## Docker

### 1. Build for linux/amd64 (required for HuggingFace Spaces)

HF Spaces runs `linux/amd64`. Building on Apple Silicon without specifying the platform produces an `arm64` image that won't run on HF.

**Podman:**

```bash
cd ui

podman build --platform linux/amd64 \
  -t docker.io/amurthi44g1wd/vakra-leaderboard:latest .
```

**Docker:**

```bash
cd ui

docker build --platform linux/amd64 \
  -t docker.io/amurthi44g1wd/vakra-leaderboard:latest .
```

### 2. Run locally

```bash
# Podman
podman run -d --name leaderboard -p 7860:7860 vakra-leaderboard:latest
# Docker
docker run -d --name leaderboard -p 7860:7860 vakra-leaderboard:latest

curl http://localhost:7860/api/leaderboard   # expect JSON array
```

Clean up: `podman stop leaderboard && podman rm leaderboard`

### 3. Push to Docker Hub

```bash
# Log in first if needed
podman login docker.io   # or: docker login docker.io

# Push
podman push docker.io/amurthi44g1wd/vakra-leaderboard:latest
# or:
docker push docker.io/amurthi44g1wd/vakra-leaderboard:latest
```

## Deploy to Hugging Face Spaces

### Option A — Push source files (HF builds for you)

```bash
git clone https://huggingface.co/spaces/<your-username>/vakra-leaderboard
cd vakra-leaderboard
cp /path/to/enterprise-benchmark-mar13/ui/{Dockerfile,app.py,data.py,leaderboard_data.json,ui.html,requirements.txt,README.md} .
git add -A && git commit -m "deploy" && git push
```

HF detects the `Dockerfile`, builds on `amd64`, and serves at
`https://huggingface.co/spaces/<your-username>/vakra-leaderboard`.

The `README.md` **must** keep the YAML frontmatter at the top (`sdk: docker`) — already present in this file.

### Option B — Pre-built image from Docker Hub

Create a one-line `Dockerfile` in the Space repo:

```dockerfile
FROM docker.io/amurthi44g1wd/vakra-leaderboard:latest
```

Push it. HF pulls the image directly with no build step.

### Redeploy

- Any `git push` to the Space repo triggers a rebuild.
- **Factory reboot**: Space Settings → Factory reboot (clears build cache).

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `ui.html` |
| `GET` | `/api/leaderboard` | Returns leaderboard rows as JSON, sorted by overall score |
| `POST` | `/api/leaderboard` | Adds a new agent entry, returns updated leaderboard |

### POST body schema

```json
{
  "model":     "My-Model-v2",
  "agent":     "ReAct (Prompt)",
  "agent_url": "https://github.com/...",
  "date":      "Mar 20, 2026",
  "scores": {
    "api":      0.0,
    "tool":     52.3,
    "multihop": 28.1,
    "multiturn": 41.5
  }
}
```

Scores are on a **0–100 scale**. `overall` is computed server-side as the mean of the four scores.

## Data Format

Agent data lives in `leaderboard_data.json`:

```json
[
  {
    "model":     "GPT-OSS-120B",
    "agent":     "ReAct (Prompt)",
    "agent_url": "https://github.com/IBM/vakra",
    "date":      "Mar 24, 2026",
    "scores": {
      "api":       0.0,
      "tool":      50.46,
      "multihop":  27.14,
      "multiturn": 40.00
    }
  }
]
```

## File Structure

```
ui/
  app.py                  # FastAPI app — serves HTML and /api/leaderboard
  data.py                 # Data layer: load/save JSON, compute overall scores
  leaderboard_data.json   # Agent scores (source of truth)
  ui.html                 # Frontend — fetches data from /api/leaderboard at load
  requirements.txt        # Python dependencies (fastapi, uvicorn, pydantic)
  Dockerfile              # Container image (python:3.12-slim, port 7860)
  README.md               # This file
```

## Adding a New Entry

Edit `leaderboard_data.json` directly and restart the server, or `POST /api/leaderboard` with the JSON body above.
To add a new score column, add the key to the `scores` object in the JSON and update the table headers in `ui.html` and the `SCORE_KEYS` list in `data.py`.
