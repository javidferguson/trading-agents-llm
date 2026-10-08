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

# ALWAYS the host, container default or not. For the few commands that inspect
# the host itself -- the Docker daemon, or a port on the host's loopback -- and
# are therefore meaningless anywhere else.
#
# `make gateway-start` was broken by routing one of these through $(RUN):
# scripts/check_gateway_exclusive.py shells out to `docker ps`, the dev service
# has no Docker CLI and no socket, so the check exited 2 ("could not tell") and
# took the target down with it. Worse, its port probe did not fail -- inside a
# container 127.0.0.1:4002 is the container's own loopback, so it cheerfully
# reported the host's port clear without having looked at it.
#
# The rest of gateway-start was already host-native ($(COMPOSE), not $(RUN)),
# which is what made the one container call easy to miss.
HOST_RUN = $(UV) run

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
	@# HOST_RUN, not RUN: both of this script's checks read host state. See the
	@# HOST_RUN comment at the top. `|| true` stays because this target is a
	@# report -- it prints the conflict rather than failing on it.
	@$(HOST_RUN) python scripts/check_gateway_exclusive.py || true
	@nc -z -G 3 127.0.0.1 $${IB_HOST_PORT:-4012} 2>/dev/null \
		&& echo "ok   desk-ib-gateway reachable on 127.0.0.1:$${IB_HOST_PORT:-4012}" \
		|| echo "warn desk-ib-gateway not running. \`make gateway-start\`. Not needed until Stage 7."

# --------------------------------------------------------------------------- #
# IB Gateway -- OURS. Never run it alongside the ORB+GEX engine's.
# --------------------------------------------------------------------------- #

.PHONY: gateway-start
gateway-start:  ## Start this project's IB Gateway (refuses if the ORB one is up)
	@# HOST_RUN, and no `|| true`: this one is a GATE, not a report. One IB
	@# username supports one Gateway session, so a check that cannot see the
	@# other Gateway must stop the start rather than shrug.
	@$(HOST_RUN) python scripts/check_gateway_exclusive.py
	@grep -qE '^IB_USERNAME=.+' .env \
		|| { echo "IB_USERNAME is empty in .env -- the Gateway will start and sit"; \
		     echo "on the login screen forever. Fill it in first."; exit 1; }
	$(COMPOSE) --profile execute up -d ib-gateway
	@echo
	@echo "Starting. First login can take a minute, and if IB asks for 2FA you"
	@echo "will only see it over VNC:  open vnc://localhost:5912"
	@echo "Watch progress with: make gateway-logs"

.PHONY: execute
execute:  ## THE STAGE 7 GATE: place an approved proposal. Needs a human and the Gateway.
	@# `compose run` not `$(RUN)`: the confirmation gate reads stdin, so the
	@# execute service sets stdin_open/tty. A closed stdin DECLINES rather than
	@# proceeding, which is why this must not be run detached.
	@#
	@# --build IS NOT OPTIONAL HERE, and this is the one target where it is
	@# worth the seconds.
	@#
	@# The `dev` service bind-mounts the WHOLE REPO at /app, so source edits are
	@# live and nothing goes stale. `execute` deliberately mounts only data/,
	@# logs/, config/ and prompts/ -- NOT src/ -- so the process that places
	@# real orders runs built, reviewed code rather than whatever happens to be
	@# in the working tree. That is a feature.
	@#
	@# The cost is that its image goes stale silently. Stage 7 exposed the loud
	@# version: the `execute` console script did not exist in the image yet, so
	@# the container died with "executable file not found in $$PATH". The
	@# dangerous version is subtler -- change a sizing rule or a safety check in
	@# broker.py and a stale image runs YESTERDAY'S order logic against today's
	@# proposal, with no indication whatsoever. For the one process that moves
	@# money, always-current beats fast.
	@#
	@# Layer caching makes this nearly free when nothing changed.
	$(COMPOSE) --profile execute run --rm --build execute execute $(EXEC_ARGS)

.PHONY: execute-list
execute-list:  ## Which proposals are pending, and which already executed
	@# --build for the same reason as above: this reads receipts written by the
	@# code it is about to run.
	$(COMPOSE) --profile execute run --rm --build execute execute --list

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

# --------------------------------------------------------------------------- #
# Stage 6 -- intent, sizing, compliance. None of these touches IB or a model.
# --------------------------------------------------------------------------- #

.PHONY: review
review:  ## Read past runs out of the decision journal. No model, no network.
	$(RUN) desk review $(if $(SYMBOL),--symbol $(SYMBOL),) $(if $(LAST),--last $(LAST),)

.PHONY: portfolio
portfolio:  ## THE STAGE 6 GATE (part 1): the book and the drift table, no LLM, no broker
	$(RUN) desk portfolio

.PHONY: candidates
candidates:  ## Channel 1: today's tradeable set, computed before any model runs
	$(RUN) desk candidates --earnings

.PHONY: portfolio-seed
portfolio-seed:  ## Rebuild config/portfolio.yaml from the bars cache. Reads the cache; never fetches.
	$(RUN) python scripts/seed_portfolio.py --seed

.PHONY: portfolio-refresh
portfolio-refresh:  ## Re-mark the book against the newest cached bars (as_of tracks the OLDEST mark)
	$(RUN) python scripts/seed_portfolio.py --refresh

.PHONY: test-stage6
test-stage6:  ## THE STAGE 6 GATE (part 2): an OrderPlan with zero IB contact, and the veto
	$(RUN) pytest -q tests/intent tests/graph/test_compliance_node.py \
	    tests/test_intent_blindness.py

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
verify:  ## Run every exit gate in order, stopping at the first failure
	@echo "running in: $(IN)"
	@echo
	@echo "=== 1/8  config ============================================"
	@$(MAKE) --no-print-directory config-check
	@echo
	@echo "=== 2/8  tests ============================================="
	@# Via the target, not $(UV) directly: this step used to run on the host
	@# even in container mode, one line after announcing "running in:
	@# container". A gate that reports a context it did not use is the same
	@# defect as one that passes without running.
	@$(MAKE) --no-print-directory test
	@echo
	@echo "=== 3/8  environment ======================================="
	@$(MAKE) --no-print-directory doctor
	@echo
	@echo "=== 4/8  Stage 0 exit gate ================================="
	@$(MAKE) --no-print-directory toy-graph
	@echo
	@echo "=== 5/8  Stage 1 exit gate ================================="
	@$(MAKE) --no-print-directory smoke
	@echo
	@echo "=== 6/8  Stage 2 exit gate ================================="
	@$(MAKE) --no-print-directory snapshot
	@echo
	@echo "=== 7/8  Stage 6 exit gate ================================="
	@# Before the Stage 3 gate, not after: this one needs no model and no
	@# network, so it takes a second and tells you whether the book is stale
	@# BEFORE you spend three minutes of inference sizing against it.
	@$(MAKE) --no-print-directory portfolio
	@$(MAKE) --no-print-directory test-stage6
	@echo
	@echo "=== 8/8  Stage 3 exit gate ================================="
	@$(MAKE) --no-print-directory decide
