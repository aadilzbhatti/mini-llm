#!/usr/bin/env bash
# Pull a Modal training run's outputs to ./runs/<run_id>, then import it like a
# local run: checkpoint + sample report -> checkpoints/, plot -> plots/, row ->
# baselines.md (see mini_llm.import_run). Unfinished runs are fetched, not imported.
#
#   scripts/fetch_modal_run.sh                          list runs in the wiki-llm-runs volume
#   scripts/fetch_modal_run.sh <run_id>                 fetch + import into this repo
#   scripts/fetch_modal_run.sh <run_id> --repo DIR      fetch + import into another checkout
#   scripts/fetch_modal_run.sh <run_id> --no-import     fetch only
#
# Safe to re-run on a live run (outputs are committed every few minutes):
# --force replaces the local copy with the current one, and importing again
# replaces the run's baselines row rather than duplicating it.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then
  exec uv run --group modal modal volume ls wiki-llm-runs /
fi
run_id=$1; shift
repo=.; do_import=1
while [ $# -gt 0 ]; do
  case $1 in
    --repo) repo=$2; shift 2 ;;
    --no-import) do_import=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
mkdir -p runs  # must exist, or `volume get` treats it as a single-file destination
uv run --group modal modal volume get --force wiki-llm-runs "/$run_id" runs
echo "-> runs/$run_id"
if [ "$do_import" = 1 ]; then
  if grep -q '"returncode": 0' "runs/$run_id/run.json"; then
    uv run mini-llm-import-run "runs/$run_id" --repo "$repo"
  else
    echo "not imported: run hasn't finished successfully (no returncode 0 in run.json)"
  fi
fi
