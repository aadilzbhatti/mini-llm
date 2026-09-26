#!/usr/bin/env bash
# Pull a Modal training run's outputs to ./runs/<run_id>.
#
#   scripts/fetch_modal_run.sh               list runs in the wiki-llm-runs volume
#   scripts/fetch_modal_run.sh <run_id>      download /<run_id> -> ./runs/<run_id>
#
# Safe to re-run on a live run (outputs are committed every few minutes):
# --force replaces the local copy with the current one.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then
  exec uv run --group modal modal volume ls wiki-llm-runs /
fi
mkdir -p runs  # must exist, or `volume get` treats it as a single-file destination
uv run --group modal modal volume get --force wiki-llm-runs "/$1" runs
echo "-> runs/$1"
