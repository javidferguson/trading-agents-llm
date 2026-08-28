# Conventions carried from the ORB+GEX engine's Makefile: `make help` that
# actually lists things, and one-line checks that diagnose a misconfiguration in
# seconds rather than through a stack trace. The targets themselves are new.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# --env-file is not optional here. Compose resolves ${VAR} interpolation against
# a .env in the PROJECT directory, which defaults to the first -f file's parent
# -- docker/, not the repo root. Without this the Langfuse init keys silently
# fall back to their empty defaults and the stack comes up unprovisioned, which
# looks exactly like "the keys in .env are wrong".
COMPOSE := docker compose --env-file .env \
	-f docker/docker-compose.yml \
	-f docker/docker-compose.langfuse.yml

.PHONY: help
help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

.PHONY: setup
setup: .env  ## Create .env and install the pinned dependency set
	uv sync --all-extras
	@echo
	@echo "Next: make langfuse-up, then make doctor"

.env:
	@cp example.env .env
	@echo 'Created .env from example.env -- fill in the keys, then run: make doctor'

.PHONY: lock
lock:  ## Re-resolve uv.lock. LangGraph pins are exact ON PURPOSE (§1) -- upgrade deliberately.
	uv lock

# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

.PHONY: doctor
doctor:  ## Full environment report: config, dirs, ollama, langfuse, gateway
	uv run desk doctor

.PHONY: config-check
config-check:  ## Parse and validate config/, and print the config hash
	uv run desk config-check

.PHONY: check-ollama
check-ollama:  ## Curl Ollama /api/tags FROM INSIDE THE CONTAINER -- see architecture §10
	@echo "Host view:"
	@curl -sS --max-time 5 $${OLLAMA_BASE_URL:-http://127.0.0.1:11434}/api/tags >/dev/null \
		&& echo "  ok   reachable from the host" \
		|| echo "  FAIL not reachable from the host either -- is ollama running?"
	@echo "Container view (this is the one that matters):"
	@$(COMPOSE) run --rm --no-deps --entrypoint sh decide -c \
		'curl -sS --max-time 5 "$$OLLAMA_BASE_URL/api/tags" >/dev/null' \
		&& echo "  ok   reachable from inside the container" \
		|| { \
			echo "  FAIL not reachable from inside the container."; \
			echo "       The daemon is bound to 127.0.0.1. On the HOST:"; \
			echo "         launchctl setenv OLLAMA_HOST 0.0.0.0:11434"; \
			echo "       then restart Ollama and try again."; \
			exit 1; \
		}

.PHONY: check-gateway
check-gateway:  ## One line on whether the IB Gateway is up. This repo never starts one.
	@nc -z -G 3 $${IB_HOST_LOCAL:-127.0.0.1} $${IB_PORT_LOCAL:-4002} 2>/dev/null \
		&& echo "ok   IB Gateway reachable on 127.0.0.1:4002" \
		|| echo "warn IB Gateway not reachable. It is owned by the ORB+GEX repo -- run \`make gateway-start\` THERE. Not needed until Stage 7."

.PHONY: check-network
check-network:  ## Ensure the shared trading-network exists (the ORB+GEX repo creates it)
	@docker network inspect trading-network >/dev/null 2>&1 \
		&& echo "ok   trading-network exists" \
		|| { echo "warn trading-network missing. Start the ORB+GEX stack, or:"; \
		     echo "       docker network create trading-network"; }

# --------------------------------------------------------------------------- #
# Stage 0 exit gate
# --------------------------------------------------------------------------- #

.PHONY: toy-graph
toy-graph:  ## THE STAGE 0 GATE: run two nodes, see two spans in Langfuse
	uv run desk toy-graph --symbol $${SYMBOL:-SPY}

# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

.PHONY: build
build:  ## Build the decide and execute images
	$(COMPOSE) build decide
	$(COMPOSE) --profile execute build execute

.PHONY: langfuse-up
langfuse-up:  ## Start the self-hosted Langfuse stack (6 services -- see the compose file)
	# Named explicitly rather than relying on the profile: `--profile langfuse up`
	# would ALSO start every unprofiled service, which means building the decide
	# image just to look at traces. depends_on pulls in postgres, clickhouse,
	# redis and minio.
	$(COMPOSE) --profile langfuse up -d langfuse-web langfuse-worker
	@echo "Langfuse starting at http://localhost:3000 (first boot runs migrations; give it a minute)"

.PHONY: langfuse-down
langfuse-down:  ## Stop Langfuse, keeping its volumes
	$(COMPOSE) --profile langfuse down

.PHONY: down
down:  ## Stop everything this repo started. Never touches the IB Gateway.
	$(COMPOSE) --profile langfuse --profile execute --profile ollama down

.PHONY: logs
logs:  ## Tail logs from every running service
	$(COMPOSE) logs -f --tail=100

# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

.PHONY: gdelt
gdelt:  ## Collect one day of GDELT tone/volume for the universe. Run it daily.
	uv run python scripts/gdelt_collect.py

.PHONY: gdelt-status
gdelt-status:  ## How many days of sentiment history exist, and where the gaps are
	uv run python scripts/gdelt_collect.py --status

.PHONY: gdelt-probe
gdelt-probe:  ## Measure how far back GDELT's API actually serves (architecture §7.5 asks)
	uv run python scripts/gdelt_collect.py --probe-window

# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

.PHONY: test
test:  ## Run the test suite
	uv run pytest -q

.PHONY: test-layering
test-layering:  ## Just the architectural boundary tests -- fast, and the ones that erode
	uv run pytest -q tests/test_layering.py
