# =============================================================================
# Enterprise Benchmark — Docker image lifecycle
# =============================================================================
# Targets:
#   make download    Download benchmark data from HuggingFace
#   make build       Build the benchmark_environ image from source
#   make test        Smoke-test the locally built image (file checks + MCP handshakes)
#   make validate    Validate live MCP connections against running containers
#   make validate-output FILES=<path>  Validate output JSON files match submission schema
#   make tag         Tag the local image for Docker Hub
#   make push        Push the tagged image to Docker Hub
#   make release     build → test → tag → push  (full publish workflow)
#   make setup       download → build → test → start → validate  (first-time setup)
#   make pull        Pull the benchmark_environ image from Docker Hub
#   make start       Start all benchmark containers (pulls latest image)
#   make stop        Stop and remove all benchmark containers
#   make restart     Stop and restart all containers without re-pulling the image
#   make logs        Tail logs for all running benchmark containers
#   make clean       Stop containers and remove the local Docker image
#   make e2e              Run end-to-end benchmark tests (requires HF_TOKEN + OPENAI_API_KEY)
#   make e2e-quick        Run e2e tests against already-running containers — OpenAI provider
#   make e2e-quick-rits   Run e2e tests against already-running containers — RITS provider
#   make e2e-quick-watsonx Run e2e tests against already-running containers — WatsonX provider
#   make e2e-quick-litellm Run e2e tests against already-running containers — LiteLLM provider
#   make e2e-quick-anthropic Run e2e tests against already-running containers — Anthropic provider
#
# Override variables (optional):
#   DOCKER=podman make build    Force use of podman
#   PYTHON=python make validate Force use of python instead of python3
# =============================================================================

REGISTRY   := docker.io/amurthi44g1wd
IMAGE_NAME := benchmark_environ
REMOTE     := $(REGISTRY)/$(IMAGE_NAME):latest
DOCKERFILE := docker/Dockerfile.unified

# Auto-detect container runtime: prefer docker, fall back to podman
DOCKER ?= $(shell PATH=$$PATH:/usr/bin command -v docker 2>/dev/null || command -v podman 2>/dev/null || echo docker)

# Auto-detect Python interpreter: prefer the active virtualenv, then python3, then python
PYTHON ?= $(shell \
  if [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python" ]; then \
    echo "$$VIRTUAL_ENV/bin/python"; \
  elif [ -x ".venv/bin/python" ]; then \
    echo ".venv/bin/python"; \
  else \
    command -v python3 2>/dev/null | head -1 || command -v python 2>/dev/null | head -1 || echo python3; \
  fi)

.PHONY: download build test validate validate-output tag push release setup pull start stop restart logs clean e2e \
        e2e-quick e2e-quick-rits e2e-quick-watsonx e2e-quick-litellm e2e-quick-anthropic \
        start-capability1 start-capability2 start-capability3 start-capability4

# ---------------------------------------------------------------------------
# Download benchmark data from HuggingFace  (prompts for HF token if not set)
# ---------------------------------------------------------------------------
download:
	$(PYTHON) benchmark_setup.py --download-data

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
# Requires: make start (or benchmark_setup.py --start-containers) run first
# ---------------------------------------------------------------------------
validate:
	$(PYTHON) benchmark/validate_clients.py

# ---------------------------------------------------------------------------
# Validate output files — check submission JSON files match the required schema
# Usage: make validate-output FILES="results/address.json results/hockey.json"
#        make validate-output FILES=results/   (all .json files in a directory)
# ---------------------------------------------------------------------------
validate-output:
	$(PYTHON) validate_output.py $(FILES)

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
	@echo "Setup complete. Run '$(PYTHON) benchmark_runner.py --capability_id 3 --domain bpo' to start benchmarking."

# ---------------------------------------------------------------------------
# Container lifecycle  (delegates to benchmark_setup.py)
# ---------------------------------------------------------------------------
pull:
	$(PYTHON) benchmark_setup.py --pull-image

start:
	$(PYTHON) benchmark_setup.py --start-containers

stop:
	$(DOCKER) compose down --remove-orphans
	$(PYTHON) benchmark_setup.py --stop-containers

logs:
	@for c in capability_1_bi_apis capability_2_dashboard_apis capability_3_multihop_reasoning capability_4_multiturn; do \
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
# e2e-quick variants: run e2e tests against already-running containers
# Skips data download and container restart. Requires: make start first.
# Each target validates the required credentials for that provider.
# ---------------------------------------------------------------------------
e2e-quick:
	@if [ -z "$(OPENAI_API_KEY)" ]; then echo "ERROR: OPENAI_API_KEY is not set."; exit 1; fi
	E2E_SKIP_SETUP=1 $(PYTHON) -m pytest tests/e2e/test_benchmark_e2e.py -v -s

e2e-quick-rits:
	@if [ -z "$(RITS_API_KEY)" ]; then echo "ERROR: RITS_API_KEY is not set."; exit 1; fi
	E2E_SKIP_SETUP=1 $(PYTHON) -m pytest tests/e2e/test_benchmark_e2e.py -v -s

e2e-quick-watsonx:
	@if [ -z "$(WATSONX_APIKEY)" ]; then echo "ERROR: WATSONX_APIKEY is not set."; exit 1; fi
	@if [ -z "$(WATSONX_PROJECT_ID)" ] && [ -z "$(WATSONX_SPACE_ID)" ]; then \
		echo "ERROR: WATSONX_PROJECT_ID or WATSONX_SPACE_ID must be set."; exit 1; fi
	E2E_SKIP_SETUP=1 $(PYTHON) -m pytest tests/e2e/test_benchmark_e2e.py -v -s

e2e-quick-litellm:
	@if [ -z "$(LITELLM_API_KEY)" ]; then echo "ERROR: LITELLM_API_KEY is not set."; exit 1; fi
	@if [ -z "$(LITELLM_BASE_URL)" ]; then echo "ERROR: LITELLM_BASE_URL is not set."; exit 1; fi
	E2E_SKIP_SETUP=1 E2E_PROVIDER=litellm $(PYTHON) -m pytest tests/e2e/test_benchmark_e2e.py -v -s

e2e-quick-anthropic:
	@if [ -z "$(ANTHROPIC_API_KEY)" ]; then echo "ERROR: ANTHROPIC_API_KEY is not set."; exit 1; fi
	E2E_SKIP_SETUP=1 E2E_PROVIDER=anthropic $(PYTHON) -m pytest tests/e2e/test_benchmark_e2e.py -v -s

# ---------------------------------------------------------------------------
# Start a single container (useful when one fails and you don't want to
# restart all four — uses docker compose so the image must already exist)
# ---------------------------------------------------------------------------
restart:
	$(DOCKER) compose down --remove-orphans
	$(DOCKER) compose up -d

start-capability1:
	-$(DOCKER) rm -f capability_1_bi_apis
	$(DOCKER) compose up -d capability_1_bi_apis

start-capability2:
	-$(DOCKER) rm -f capability_2_dashboard_apis
	$(DOCKER) compose up -d capability_2_dashboard_apis

start-capability3:
	-$(DOCKER) rm -f capability_3_multihop_reasoning
	$(DOCKER) compose up -d capability_3_multihop_reasoning

start-capability4:
	-$(DOCKER) rm -f capability_4_multiturn
	$(DOCKER) compose up -d capability_4_multiturn

# ---------------------------------------------------------------------------
# Clean — stop & remove containers, then remove the local image
# ---------------------------------------------------------------------------
clean:
	$(PYTHON) benchmark_setup.py --stop-containers
	$(DOCKER) rmi -f $(IMAGE_NAME) 2>/dev/null || true
	@echo "Removed containers and image '$(IMAGE_NAME)'."
