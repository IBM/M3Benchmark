# =============================================================================
# M3 Benchmark — Docker image lifecycle
# =============================================================================
# Targets:
#   make download    Download benchmark data from HuggingFace
#   make build       Build the m3_environ image from source
#   make test        Smoke-test the locally built image (file checks + MCP handshakes)
#   make validate    Validate live MCP connections against running containers
#   make tag         Tag the local image for Docker Hub
#   make push        Push the tagged image to Docker Hub
#   make release     build → test → tag → push  (full publish workflow)
#   make setup       download → build → test → start → validate  (first-time setup)
#   make pull        Pull the m3_environ image from Docker Hub
#   make start       Start all benchmark containers (pulls latest image)
#   make stop        Stop and remove all benchmark containers
#   make logs        Tail logs for all running benchmark containers
#   make clean       Stop containers and remove the local Docker image
#   make e2e         Run end-to-end benchmark tests (requires HF_TOKEN + OPENAI_API_KEY)
#   make e2e-quick   Run e2e tests against already-running containers (no download/restart)
#
# Override variables (optional):
#   DOCKER=podman make build    Force use of podman
#   PYTHON=python make validate Force use of python instead of python3
# =============================================================================

REGISTRY   := docker.io/amurthi44g1wd
IMAGE_NAME := m3_environ
REMOTE     := $(REGISTRY)/$(IMAGE_NAME):latest
DOCKERFILE := docker/Dockerfile.unified

# Auto-detect container runtime: prefer docker, fall back to podman
DOCKER ?= $(shell command -v docker 2>/dev/null | head -1 || command -v podman 2>/dev/null | head -1 || echo docker)

# Auto-detect Python interpreter: prefer python3, fall back to python
PYTHON ?= $(shell command -v python3 2>/dev/null | head -1 || command -v python 2>/dev/null | head -1 || echo python3)

.PHONY: download build test validate tag push release setup pull start stop logs clean e2e e2e-quick

# ---------------------------------------------------------------------------
# Download benchmark data from HuggingFace  (prompts for HF token if not set)
# ---------------------------------------------------------------------------
download:
	$(PYTHON) m3_setup.py --download-data

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build:
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

# ---------------------------------------------------------------------------
# Smoke-test  (spins up a temp container, no volume mounts needed)
# Checks: required files exist, M3 REST /openapi.json, BPO + M3 MCP handshakes
# ---------------------------------------------------------------------------
test:
	bash docker/smoke_test.sh $(IMAGE_NAME)

# ---------------------------------------------------------------------------
# Validate live MCP connections against the running benchmark containers
# Requires: make start (or m3_setup.py --start-containers) run first
# ---------------------------------------------------------------------------
validate:
	$(PYTHON) benchmark/validate_clients.py

# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------
tag:
	$(DOCKER) tag $(IMAGE_NAME) $(REMOTE)

# ---------------------------------------------------------------------------
# Push  (requires: docker login docker.io  or  podman login docker.io)
# ---------------------------------------------------------------------------
push:
	$(DOCKER) push $(REMOTE)

# ---------------------------------------------------------------------------
# Full release workflow  (image only, no data download)
# ---------------------------------------------------------------------------
release: build test tag push
	@echo ""
	@echo "Released $(REMOTE)"
	@echo "Run 'make start && make validate' to restart containers and verify live connections."

# ---------------------------------------------------------------------------
# First-time setup: data + image + containers
# ---------------------------------------------------------------------------
setup: download build test start validate
	@echo ""
	@echo "Setup complete. Run '$(PYTHON) benchmark_runner.py --m3_task_id 3 --domain bpo' to start benchmarking."

# ---------------------------------------------------------------------------
# Container lifecycle  (delegates to m3_setup.py)
# ---------------------------------------------------------------------------
pull:
	$(PYTHON) m3_setup.py --pull-image

start:
	$(PYTHON) m3_setup.py --start-containers

stop:
	$(PYTHON) m3_setup.py --stop-containers

logs:
	@for c in task_1_m3_environ task_2_m3_environ task_3_m3_environ task_5_m3_environ; do \
		echo ""; \
		echo "=== $$c ==="; \
		$(DOCKER) logs --tail 20 $$c 2>/dev/null || echo "  (not running)"; \
	done

# ---------------------------------------------------------------------------
# End-to-end benchmark tests
# Requires: HF_TOKEN and OPENAI_API_KEY env vars set
# ---------------------------------------------------------------------------
e2e:
	@if [ -z "$(HF_TOKEN)" ]; then echo "ERROR: HF_TOKEN is not set."; exit 1; fi
	@if [ -z "$(OPENAI_API_KEY)" ]; then echo "ERROR: OPENAI_API_KEY is not set."; exit 1; fi
	$(PYTHON) -m pytest tests/e2e/test_benchmark_e2e.py -v -s

# ---------------------------------------------------------------------------
# e2e-quick: run e2e tests against already-running containers
# Skips data download and container restart. Requires: make start first.
# ---------------------------------------------------------------------------
e2e-quick:
	@if [ -z "$(OPENAI_API_KEY)" ]; then echo "ERROR: OPENAI_API_KEY is not set."; exit 1; fi
	E2E_SKIP_SETUP=1 $(PYTHON) -m pytest tests/e2e/test_benchmark_e2e.py -v -s

# ---------------------------------------------------------------------------
# Clean — stop & remove containers, then remove the local image
# ---------------------------------------------------------------------------
clean:
	$(PYTHON) m3_setup.py --stop-containers
	$(DOCKER) rmi -f $(IMAGE_NAME) 2>/dev/null || true
	@echo "Removed containers and image '$(IMAGE_NAME)'."
