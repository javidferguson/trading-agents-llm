#!/usr/bin/env bash
#
# Research sweep: run `desk decide` once per symbol, independently.
#
# WHY A LOOP AND NOT A MULTI-SYMBOL RUN
# =====================================
# `decide` is per-symbol and stays that way. `DecisionState.symbol` is a plain
# string -- one state, one symbol -- which is architecture decision 3 and is
# what keeps a run replayable and small enough to journal. So a sweep is N
# INDEPENDENT runs, which is safe for a specific reason: each run READS the book
# and never writes it.
#
# The one thing successive runs share is the decision journal, and that is
# deliberate: `compliance.decisions_today()` reads it, which is how
# `max_decisions_per_day` self-limits a sweep. Past the third symbol with an
# actionable decision, the rest are vetoed to HOLD. That is the cadence limit
# working, not the sweep failing -- use `desk review --wanted` to see what the
# pipeline actually decided before Python rewrote it.
#
# WHY ONE CONTAINER
# =================
# `make decide SYMBOL=X` spins a fresh container per call. Thirty-one of those
# is pure overhead, so this runs the whole loop inside one.
#
# FAILURE IS NOT FATAL
# ====================
# One symbol with no cached bars, or a model that will not return valid JSON
# after its repair turn, must not kill a fifty-minute sweep. Each symbol is
# allowed to fail and the sweep reports which did at the end.
#
# PROPOSALS FROM THIS SWEEP EXPIRE IN ~18 HOURS
# =============================================
# `cadence.proposal_ttl_hours`. That is what makes "sweep today, trade
# tomorrow" safe: tomorrow's trades come from TOMORROW's sweep, and today's
# thinking cannot be executed by accident after it has gone stale.
#
# Usage:
#   scripts/sweep.sh                     # the whole tradeable universe
#   scripts/sweep.sh TSM AMD PLTR        # just these
#   DECIDE_ARGS=--slice scripts/sweep.sh TSM    # pass flags through to decide
set -uo pipefail

SYMBOLS=("$@")

if [ "${#SYMBOLS[@]}" -eq 0 ]; then
  # The full TRADEABLE universe, not candidates(): research wants to know about
  # a symbol even when the pre-filter would skip it for an earnings blackout or
  # a full book. The per-symbol compliance checks still record those reasons.
  mapfile -t SYMBOLS < <(python -c \
    "from research_desk.intent.engine import load_intent
for s in load_intent().universe.tradeable: print(s)")
fi

COUNT="${#SYMBOLS[@]}"
# ~100s per symbol measured across Stage 5-7 runs: 13 model calls on a local
# quantized model. Printed up front because a command that silently runs for an
# hour is a command people kill halfway through.
EST=$(( COUNT * 100 / 60 ))

echo "SWEEP: ${COUNT} symbol(s), ~100s each -- expect roughly ${EST} minute(s)."
echo "       ${SYMBOLS[*]}"
echo
echo "Proposals from this sweep expire in ~18h (cadence.proposal_ttl_hours),"
echo "and max_decisions_per_day will veto past the third actionable symbol."
echo "Neither is a failure. \`desk review --wanted\` shows the real verdicts."
echo

STARTED=$(date +%s)
FAILED=()
INDEX=0

for symbol in "${SYMBOLS[@]}"; do
  INDEX=$(( INDEX + 1 ))
  echo
  echo "=============================================================="
  echo "[${INDEX}/${COUNT}]  ${symbol}"
  echo "=============================================================="
  # shellcheck disable=SC2086  # DECIDE_ARGS is intentionally word-split
  if ! desk decide --symbol "${symbol}" ${DECIDE_ARGS:-}; then
    # `decide` exits non-zero on a DEGRADED run too, which is information
    # rather than breakage -- the decision is a HOLD carrying its reason.
    echo "  >> ${symbol}: non-zero exit (degraded or failed). Continuing."
    FAILED+=("${symbol}")
  fi
done

ELAPSED=$(( $(date +%s) - STARTED ))

echo
echo "=============================================================="
echo "SWEEP DONE: ${COUNT} symbol(s) in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "  non-zero exits (${#FAILED[@]}): ${FAILED[*]}"
  # Single-quoted: backticks inside a double-quoted string are command
  # substitution, and this one actually RAN `desk review` mid-summary.
  echo '  `desk review --symbol SYM` shows why -- a degraded HOLD states its reason.'
fi
echo "=============================================================="
echo
# `--last COUNT` unfiltered, deliberately. A `--wanted --last COUNT` here would
# filter FIRST and then take the last N, so a sweep where only some symbols
# were actionable would pad the list with older runs from outside the sweep --
# which it did, and read as if those were part of it.
#
# Unfiltered is correct because these COUNT runs are the most recent ones, and
# render_summary already shows pre-veto intent as `BUY->HOLD` wherever Python
# changed the answer. Nothing is hidden by not filtering.
echo "THIS SWEEP (intent->outcome where they differ):"
desk review --last "${COUNT}" || true
echo
echo "Across days, filter on what the pipeline wanted rather than what"
echo "survived compliance:   desk review --wanted --last 30"
