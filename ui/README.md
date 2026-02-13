---
title: Agent Performance Leaderboard
emoji: 🏆
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# Agent Performance Leaderboard UI

A Streamlit dashboard for comparing agent performance across benchmark task types.

## Quick Start (local, no Docker)

```bash
pip install -r ui/requirements.txt
cd ui && streamlit run app.py
```

Opens at http://localhost:8501.

## Quick Start (Docker)

### 1. Build

**Docker (buildx, multi-arch):**
```bash
cd ui
docker buildx build --platform linux/amd64,linux/arm64 \
  -t docker.io/amurthi44g1wd/agent-leaderboard --push .
```

**Podman (manifest, multi-arch):**
```bash
cd ui
podman build --platform linux/amd64 -t docker.io/amurthi44g1wd/agent-leaderboard:amd64 .
podman build --platform linux/arm64 -t docker.io/amurthi44g1wd/agent-leaderboard:arm64 .
podman manifest create docker.io/amurthi44g1wd/agent-leaderboard
podman manifest add docker.io/amurthi44g1wd/agent-leaderboard docker.io/amurthi44g1wd/agent-leaderboard:amd64
podman manifest add docker.io/amurthi44g1wd/agent-leaderboard docker.io/amurthi44g1wd/agent-leaderboard:arm64
```

**Single-platform only (for local testing):**
```bash
docker buildx build --platform linux/amd64 -t docker.io/amurthi44g1wd/agent-leaderboard --load .
# or
podman build --platform linux/amd64 -t docker.io/amurthi44g1wd/agent-leaderboard .
```

### 2. Test locally

```bash
docker run -d --name leaderboard-test -p 7860:7860 docker.io/amurthi44g1wd/agent-leaderboard
curl http://localhost:7860/_stcore/health   # expect: ok
```

Clean up: `docker stop leaderboard-test && docker rm leaderboard-test`

### 3. Push to Docker Hub

```bash
docker login docker.io
# buildx --push already pushes; for podman:
podman manifest push docker.io/amurthi44g1wd/agent-leaderboard
```

## Deploy to Hugging Face Spaces

### Option A — Push source files (HF builds for you)

```bash
git clone https://huggingface.co/spaces/<your-username>/agent-leaderboard
cd agent-leaderboard
cp /path/to/enterprise-benchmark-1/ui/{Dockerfile,app.py,data.py,requirements.txt,leaderboard_data.json,README.md} .
git add -A && git commit -m "Initial deploy" && git push
```

HF detects the Dockerfile, builds on amd64, and serves at `https://huggingface.co/spaces/<your-username>/agent-leaderboard`.

The `README.md` **must** have the YAML frontmatter at the top (`sdk: docker`) — already included in this file.

### Option B — Pre-built image from Docker Hub

Create a one-line `Dockerfile` in your Space repo:
```dockerfile
FROM docker.io/amurthi44g1wd/agent-leaderboard
```

Push it. HF pulls the image directly — no build step.

### Redeploy

- **Any `git push`** to the Space repo triggers a rebuild.
- **Factory reboot**: Space Settings → Factory reboot (clears cache).

## Data Format

Agent data lives in `leaderboard_data.json`:

```json
[
  {
    "agent": "LangGraph React Agent",
    "model": "gpt-4o",
    "scores": {
      "Single Turn|API Styles|Business Intelligence": 90.0,
      "Single Turn|API Styles|Dashboard APIs": 87.0,
      "Single Turn|Reasoning|Multi-hop": 82.0,
      "Single Turn|Reasoning|Policy Adherence": 91.0,
      "Multi-turn|Joint Structured/ Unstructured Reasoning|Multi-hop": 85.0
    }
  }
]
```

Score keys are pipe-delimited: `Turn Type|Task Type|Metric`, matching `COLUMN_HIERARCHY` in `data.py`.

## File Structure

```
ui/
  app.py                  # Streamlit app (entry point)
  data.py                 # Data layer: schema, DataFrame builder, load/save
  leaderboard_data.json   # Agent scores
  requirements.txt        # Python dependencies
  Dockerfile              # Container image
  README.md               # This file
```

## Customization

Add a new metric: append a tuple to `COLUMN_HIERARCHY` in `data.py` and add the matching score key to your agents in `leaderboard_data.json`. Everything else derives from the hierarchy automatically.
