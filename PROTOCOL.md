# Remote experiment protocol (v1)

A small, file-first protocol for making a training setup asynchronous and
steerable from anywhere: queue work, steer live runs, collect results. It
deliberately does **not** do metrics or plots; those go to TensorBoard (or
any logger you already use). What it adds is the part those tools lack: a job
queue with validation, and typed, logged, step-aligned intervention in
running jobs.

Everything is plain JSON files in a repo. The HTTP API (`mini_llm/server.py`)
is a thin, stateless view over them, so anything that can write a file or make
an HTTP request can take part: a phone, a script, cron, an LLM agent.

```
 clients (phone page, curl, agents)
        │  HTTP (behind Tailscale)
        ▼
 control API ── stateless; reads/writes the files below
        │
 queue/*.json ──► runner ──► trainer process ──► runs/tb/<id>/  (TensorBoard)
                               ▲   │
      runs/<id>.commands.jsonl ┘   └► runs/<id>.events.jsonl, .live.json, .status.json
```

## 1. Jobs: `queue/<file>.json`

```json
{ "name": "emb256-lr3e-4", "kind": "train", "args": { "steps": 160000, "lr": 0.0003 } }
```

- `kind` selects the executor; `args` are its CLI flags without `--`.
- The runner validates every job against a per-kind **allowlist** with types
  and ranges before anything executes. There is no shell. A job can only ever
  become one known command shape.
- The API validates with the same code at submit time, so bad jobs are
  rejected before they reach the queue.
- Jobs run serially, oldest first. Outcome: `queue/done/`, `queue/failed/`
  (rejected) or `queue/cancelled/`.
- Writers must write atomically (temp file + rename): the runner only picks up
  `*.json`.

## 2. Run records: `runs/<run_id>.status.json`

Written by the runner. `status` goes `running → completed | failed | interrupted`
(or `rejected`), and the record holds `args`, `cmd`, timing, parsed final
`metrics`, and the pre-run `forecast` and its `forecast_error`. Finished runs
are also appended to `runs/index.jsonl`.

The trainer receives the run id as `MINI_LLM_RUN_ID` and uses it for
everything below, so all of a run's files share one key.

## 3. Live control: inbox, outbox, heartbeat

**Inbox** `runs/<id>.commands.jsonl`: append one JSON object per line. The
line must be newline-terminated, written in a single `O_APPEND` write.

| type | fields | effect |
| --- | --- | --- |
| `set` | `knob`, `value` | change a declared knob (below) |
| `pause` / `resume` | | block / continue at a step boundary |
| `eval_now` | | sampled + full val eval at the next boundary |
| `checkpoint` | | save `checkpoints/<stem>.step<N>.pt` |
| `stop` | | end now; still runs final eval, plot, save, report |

Knobs, with type and range: `lr_scale` (float, 1e-4 to 10; multiplies the
scheduled LR), `log_interval`, `eval_interval` and `full_eval_interval` (int).
Anything not declared is immutable mid-run. Changing it would make it a
different experiment, and that belongs in a new job.

Every command may carry an `id`. The trainer applies each id at most once,
across retries and restarts, so clients can retry freely.

**Outbox** `runs/<id>.events.jsonl`: one line per command, applied or
rejected, with the exact `step` it took effect. A steered run is reproducible
from `args + events`. The same events are written to TensorBoard as text, and
`lr_scale` as the scalar `control/lr_scale`, so interventions line up with the
curves.

**Heartbeat** `runs/<id>.live.json`: rewritten atomically at most every 2s,
with `step`, `total_steps`, `steps_per_sec`, `eta_sec`, `lr`, latest losses,
current knob values and `paused`. A stale `updated` means the process is gone.

Commands are only read at step boundaries, every `--control-poll` steps, so
nothing lands mid-backward.

## 4. HTTP API

| method | path | |
| --- | --- | --- |
| GET | `/api/meta` | allowlists, knobs, command types, datasets: enough to build a UI |
| GET | `/api/runs`, `/api/runs/{id}` | records (+ `live` for running ones) |
| GET | `/api/runs/{id}/log?tail=N`, `/report`, `/events` | log tail, sample report, commands sent/applied |
| POST | `/api/runs/{id}/commands` | append a command (409 unless running) |
| GET | `/api/queue` | pending jobs |
| POST | `/api/jobs` | validate + enqueue (422 with the reason if rejected) |
| DELETE | `/api/queue/{file}` | cancel a pending job |
| GET | `/api/baselines` | leaderboard |

An OpenAPI schema is served at `/openapi.json`, with an interactive explorer
at `/docs`. The optional bearer token is set with `MINI_LLM_TOKEN`.

## Adopting it in another project

What's specific to this repo is small: the allowlists in `runner/run_queue.py`
and the handful of lines in `train.py` that construct a `RunControl`, call
`control.poll(step)` at the top of each step, read knobs from `control.state`,
and send scalars through `control.scalar(...)`. `mini_llm/control.py` itself is
dependency-free apart from an optional TensorBoard import, and can be copied
as-is.

Obvious next steps, none built yet:

- an MCP adapter over the HTTP API, so an agent can watch and steer runs
  under the same allowlists and audit log;
- a pull-based remote worker (lease + heartbeat) for machines other than
  this Mac;
- object storage for checkpoints.
