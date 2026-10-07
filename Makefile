# Conventions carried from the ORB+GEX engine's Makefile: `make help` that
# actually lists things, and one-line checks that diagnose a misconfiguration in
# seconds rather than through a stack trace. The targets themselves are new.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# `uv run`, with VIRTUAL_ENV cleared. If you have another venv active (a global
# dev env, say), uv prints a warning on EVERY command:
#
#   warning: `VIRTUAL_ENV=...` does not match the project environment path
#   `.venv` and will be ignored
#
# It is only a warning -- uv already does the right thing -- but it buries real
# output. Clearing the variable for the subprocess silences it without changing
# which environment is used.
UV := VIRTUAL_ENV= uv

# ---------------------------------------------------------------------------
# CONTAINERS ARE THE DEFAULT. `make shell` is the starting point.
#
# Every target below runs inside the `dev` service, which bind-mounts this repo
# so edits take effect with no rebuild. Append HOST=1 to run on the host
# instead -- useful when Docker itself is the thing that is broken:
#
#     make test              # in the container (default)
#     make test HOST=1       # on the host, via uv
#
# The two contexts reach some services at different addresses (Ollama,
# Langfuse, the IB Gateway); the compose file and config.resolve_ib_endpoint()
# handle that, so the same command works either way.
# ---------------------------------------------------------------------------
# Deferred (=) not immediate (:=): COMPOSE is defined below this block, and
# with := these would expand to an empty command.
ifdef HOST
RUN = $(UV) run
IN  = host
else
RUN = $(COMPOSE) run --rm --no-deps dev
IN  = container
endif

# As RUN, but with REQUIRE_SMOKE=1 set, which turns the live-model tests' skip
# into a hard failure (see tests/llm/test_smoke_ollama.py). Compose does not
# forward host environment to `run` unless asked, so the container form needs
# an explicit -e rather than a `VAR=1 make ...` prefix, which would be
# silently dropped at the container boundary -- the same class of mistake the
# gate itself exists to catch.
ifdef HOST
RUN_SMOKE = REQUIRE_SMOKE=1 $(UV) run
else
RUN_SMOKE = $(COMPOSE) run --rm --no-deps -e REQUIRE_SMOKE=1 dev
endif

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
	$(UV) sync --all-extras
	@echo
	@echo "Next: make langfuse-up, then make doctor"

.env:
	@cp example.env .env
	@echo 'Created .env from example.env -- fill in the keys, then run: make doctor'

.PHONY: lock
lock:  ## Re-resolve uv.lock. LangGraph pins are exact ON PURPOSE (§1) -- upgrade deliberately.
	$(UV) lock

# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

.PHONY: shell
shell:  ## THE DEFAULT DEV ENTRY POINT: a bash shell inside the desk container
	@echo "research-desk dev container. The repo is mounted at /app, so edits here"
	@echo "take effect immediately. Try: desk doctor | desk snapshot --symbol AAPL | pytest -q"
	@echo
	$(COMPOSE) run --rm --service-ports dev bash

.PHONY: up
up:  ## Start the long-running services (Langfuse). Ollama stays native -- see §10.
	$(MAKE) --no-print-directory langfuse-up
	@echo
	@echo "Ollama is NOT containerised on purpose (§10: 5-15x slower on macOS)."
	@$(MAKE) --no-print-directory check-ollama || true

.PHONY: doctor
doctor:  ## Full environment report: config, dirs, ollama, langfuse, gateway
	$(RUN) desk doctor

.PHONY: config-check
config-check:  ## Parse and validate config/, and print the config hash
	$(RUN) desk config-check

.PHONY: check-ollama
check-ollama:  ## Curl Ollama /api/tags from inside the container -- the only view that matters
	@echo "Host view:"
	@curl -sS --max-time 5 $${OLLAMA_BASE_URL:-http://127.0.0.1:11434}/api/tags >/dev/null \
		&& echo "  ok   reachable from the host" \
		|| echo "  FAIL not reachable from the host either -- is ollama running?"
	@echo "Container view (this is the one that matters):"
	@$(COMPOSE) run --rm --no-deps --entrypoint sh dev -c \
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
check-gateway:  ## Is our Gateway up? And is the ORB engine's conflicting?
	@$(RUN) python scripts/check_gateway_exclusive.py || true
	@nc -z -G 3 127.0.0.1 $${IB_HOST_PORT:-4012} 2>/dev/null \
		&& echo "ok   desk-ib-gateway reachable on 127.0.0.1:$${IB_HOST_PORT:-4012}" \
		|| echo "warn desk-ib-gateway not running. \`make gateway-start\`. Not needed until Stage 7."

# --------------------------------------------------------------------------- #
# IB Gateway -- OURS. Never run it alongside the ORB+GEX engine's.
# --------------------------------------------------------------------------- #

.PHONY: gateway-start
gateway-start:  ## Start this project's IB Gateway (refuses if the ORB one is up)
	@$(RUN) python scripts/check_gateway_exclusive.py
	@grep -qE '^IB_USERNAME=.+' .env \
		|| { echo "IB_USERNAME is empty in .env -- the Gateway will start and sit"; \
		     echo "on the login screen forever. Fill it in first."; exit 1; }
	$(COMPOSE) --profile execute up -d ib-gateway
	@echo
	@echo "Starting. First login can take a minute, and if IB asks for 2FA you"
	@echo "will only see it over VNC:  open vnc://localhost:5912"
	@echo "Watch progress with: make gateway-logs"

.PHONY: gateway-stop
gateway-stop:  ## Stop this project's IB Gateway. Never touches ajj-ib-gateway.
	$(COMPOSE) --profile execute stop ib-gateway

.PHONY: gateway-logs
gateway-logs:  ## Tail the Gateway log -- where login and 2FA problems surface
	$(COMPOSE) --profile execute logs -f --tail=100 ib-gateway

.PHONY: gateway-vnc
gateway-vnc:  ## Open the Gateway UI (needs VNC_PASSWORD set in .env)
	@grep -qE '^VNC_PASSWORD=.+' .env \
		|| { echo "VNC_PASSWORD is empty in .env, so the image never starts x11vnc"; \
		     echo "and nothing is listening on 5912. Set it and restart the Gateway."; exit 1; }
	open vnc://localhost:5912

.PHONY: models
models:  ## Pull every Ollama model config/models.yaml asks for
	@$(RUN) python -c "import yaml,sys; \
p=yaml.safe_load(open('config/models.yaml'))['profiles']; \
print('\n'.join(sorted({v['model'] for v in p.values() if v.get('provider')=='ollama'})))" \
	| while read -r m; do echo ">> ollama pull $$m"; ollama pull "$$m"; done

# --------------------------------------------------------------------------- #
# Stage 0 exit gate
# --------------------------------------------------------------------------- #

.PHONY: toy-graph
toy-graph:  ## THE STAGE 0 GATE: run two nodes, see two spans in Langfuse
	$(RUN) desk toy-graph --symbol $${SYMBOL:-SPY}

.PHONY: bars
bars:  ## Fetch daily bars from IB into the cache (needs the Gateway up)
	# In the container this reaches desk-ib-gateway:4004 over
	# trading-llm-network; on the host it resolves to 127.0.0.1:$(IB_HOST_PORT).
	# resolve_ib_endpoint() probes both, so the command is identical either way.
	$(RUN) python scripts/fetch_bars.py $(if $(SYMBOLS),--symbols $(SYMBOLS),)

.PHONY: snapshot
snapshot:  ## THE STAGE 2 GATE: every §7.1 metric for a symbol, no LLM
	$(RUN) desk snapshot --symbol $${SYMBOL:-SPY}

.PHONY: decide
decide:  ## THE STAGE 3 GATE: a real decision end to end -> proposal.json
	$(RUN) desk decide --symbol $${SYMBOL:-MSFT}

.PHONY: smoke
smoke:  ## THE STAGE 1 GATE: a real AnalystReport from a local 8B model
	$(RUN) desk smoke

.PHONY: test-smoke
test-smoke:  ## Run the live model tests, failing if they are skipped
	$(RUN_SMOKE) pytest tests/llm/test_smoke_ollama.py -p no:randomly -q -rs -m smoke

# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

.PHONY: build
build:  ## Build the dev, decide and execute images
	$(COMPOSE) build dev
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
down:  ## Stop everything this repo started -- including OUR Gateway, never the ORB one.
	# Scoped to this compose project, so ajj-ib-gateway is untouched by
	# construction: it belongs to a different project and is not in these files.
	$(COMPOSE) --profile langfuse --profile execute --profile ollama down

.PHONY: logs
logs:  ## Tail logs from every running service
	$(COMPOSE) logs -f --tail=100

# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

.PHONY: gdelt
gdelt:  ## Collect one day of GDELT tone/volume for the universe. Run it daily.
	$(RUN) python scripts/gdelt_collect.py

.PHONY: gdelt-status
gdelt-status:  ## How many days of sentiment history exist, and where the gaps are
	$(RUN) python scripts/gdelt_collect.py --status

.PHONY: gdelt-probe
gdelt-probe:  ## Measure how far back GDELT's API actually serves (architecture §7.5 asks)
	$(RUN) python scripts/gdelt_collect.py --probe-window

# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

.PHONY: test
test:  ## Run the test suite
	$(RUN) pytest -q

.PHONY: test-layering
test-layering:  ## Just the architectural boundary tests -- fast, and the ones that erode
	$(RUN) pytest -q tests/test_layering.py

# --------------------------------------------------------------------------- #
# The one command to run after a pull
# --------------------------------------------------------------------------- #

.PHONY: verify
verify:  ## Run the whole Stage 0 gate in order, stopping at the first failure
	@echo "running in: $(IN)"
	@echo
	@echo "=== 1/7  config ============================================"
	@$(MAKE) --no-print-directory config-check
	@echo
	@echo "=== 2/7  tests ============================================="
	@# Via the target, not $(UV) directly: this step used to run on the host
	@# even in container mode, one line after announcing "running in:
	@# container". A gate that reports a context it did not use is the same
	@# defect as one that passes without running.
	@$(MAKE) --no-print-directory test
	@echo
	@echo "=== 3/7  environment ======================================="
	@$(MAKE) --no-print-directory doctor
	@echo
	@echo "=== 4/7  Stage 0 exit gate ================================="
	@$(MAKE) --no-print-directory toy-graph
	@echo
	@echo "=== 5/7  Stage 1 exit gate ================================="
	@$(MAKE) --no-print-directory smoke
	@echo
	@echo "=== 6/7  Stage 2 exit gate ================================="
	@$(MAKE) --no-print-directory snapshot
	@echo
	@echo "=== 7/7  Stage 3 exit gate ================================="
	@$(MAKE) --no-print-directory decide
