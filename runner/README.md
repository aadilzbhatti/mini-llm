# Queue runner

Drop a JSON job file into `queue/`, and a background watcher on this Mac runs
it as a `mini-llm-train` invocation, streaming output to `runs/<run_id>.log`
and writing `runs/<run_id>.status.json` when it finishes.

The point: Claude can write job files into `queue/` from anywhere (phone,
laptop, wherever), but cannot execute anything on this Mac. This watcher is
the one piece that executes, and it only ever executes one command shape.

## From your phone

Three services on this Mac, all bound to localhost, published to your devices
over [Tailscale](https://tailscale.com):

| service | what it's for | plist |
| --- | --- | --- |
| queue runner | executes jobs, one at a time | `com.aadil.mini-llm-runner` |
| control API + phone page | queue/cancel jobs, steer live runs, read results | `com.aadil.mini-llm-control` |
| TensorBoard | every plot: live, historical, run-vs-run, hparams | `com.aadil.mini-llm-tensorboard` |

Plots are entirely TensorBoard's job. The control page only shows numbers
(step, ETA, latest losses) and links out to TensorBoard for curves.

### One-time setup

1. **Tailscale** on the Mac and the phone, same account:
   `brew install --cask tailscale`, sign in, then install the iOS/Android app.
2. **Dependencies**: `uv sync` (adds tensorboard, fastapi, uvicorn).
3. **Services**:
   ```bash
   for s in runner control tensorboard; do
     cp runner/com.aadil.mini-llm-$s.plist ~/Library/LaunchAgents/
     launchctl unload ~/Library/LaunchAgents/com.aadil.mini-llm-$s.plist 2>/dev/null
     launchctl load -w ~/Library/LaunchAgents/com.aadil.mini-llm-$s.plist
   done
   ```
   (The runner must be reloaded to pick up the new code: dataset jobs, run ids,
   notifications.)
4. **Publish to your tailnet** (HTTPS, tailnet-only; nothing is exposed to the
   public internet):
   ```bash
   tailscale serve --bg --https=443  http://127.0.0.1:8765   # control page
   tailscale serve --bg --https=8443 http://127.0.0.1:6006   # TensorBoard
   tailscale serve status                                    # shows your https://<mac>.<tailnet>.ts.net URLs
   ```
   Put the TensorBoard URL into `MINI_LLM_TENSORBOARD_URL` in the control plist
   and reload it. On the phone, open the control URL and "Add to Home Screen".
5. **Notifications (optional)**: install the [ntfy](https://ntfy.sh) app,
   subscribe to a random topic name, set `NTFY_TOPIC` in the runner plist,
   reload. You get a push when a job starts, finishes (with its full val
   loss), fails, or is rejected.

`MINI_LLM_TOKEN` in the control plist adds a bearer token on top of Tailscale
if you ever share the tailnet.

### What you can do from the phone

- **Live**: progress, ETA, it/s, current LR, latest losses; pause/resume,
  eval now, checkpoint now, stop early, scale the LR (×0.5 … ×2), change
  eval/log cadence. Commands apply at the next step boundary (every
  `--control-poll` steps, default 25) and are logged with their step.
- **Queue**: see and cancel pending jobs; queue a training run or a dataset
  build (`kind: "prepare-data"`); "Clone" on any past run pre-fills its args.
- **History**: every run with status, duration and full val loss; its args,
  metrics, sample report and log tail.
- **Best**: the baselines table.

Everything the page does is a plain HTTP call, documented at `/docs` on the
same URL (FastAPI's built-in API explorer), and the on-disk protocol behind it
is in [`PROTOCOL.md`](../PROTOCOL.md).

## Install

```bash
cp runner/com.aadil.mini-llm-runner.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.aadil.mini-llm-runner.plist
```

Check it came up:

```bash
launchctl list | grep mini-llm-runner    # second column 0 = healthy
tail -f runner/runner.log
```

Stop it:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.aadil.mini-llm-runner.plist
```

The plist hardcodes `/Users/aadil/dev/wiki-llm` and runs the watcher with
`.venv/bin/python`. If you move the repo or delete `.venv`, edit the plist and
reload. `ProcessType` is `Standard` so macOS doesn't throttle training the way
it would a `Background` job.

## Job format

Training (the default kind):

```json
{
  "name": "emb192-seed7",
  "args": { "n-embd": 192, "steps": 20000, "seed": 7 }
}
```

Dataset build:

```json
{
  "name": "data20k",
  "kind": "prepare-data",
  "args": { "num-examples": 20000, "val-examples": 200, "out-dir": "data/data20k" }
}
```

`prepare-data` accepts `num-examples`, `val-examples`, `seed`,
`val-pool-fraction`, `dataset`, `config`, `split`, `text-field`, `tokenizer` and
a required `out-dir`, which must be a folder under `data/`.

`name` is optional (defaults to the filename) and only affects the run id.
`args` are `mini-llm-train` flags without the leading `--`. Anything you omit
falls back to the training script's own defaults, except these, which the
runner fills in:

- `tokens`: `data/train.pt`
- `val-tokens`: `data/val.pt`
- `plot-loss`: `true`

Run it by hand to test, without waiting for the watcher:

```bash
cp runner/example-job.json queue/
# or drain the queue synchronously in the foreground:
.venv/bin/python runner/run_queue.py --once
```

## What a job may contain

Only these flags, with these types. Anything else is rejected before a
process starts.

| kind | flags |
| --- | --- |
| integers | `block-size` `n-embd` `n-head` `n-layer` `batch-size` `steps` `warmup-steps` `seed` `log-interval` `eval-interval` `eval-batches` `eval-seed` `full-eval-interval` `sample-tokens` |
| floats | `dropout` `lr` `min-lr` `weight-decay` |
| repo-relative paths | `tokens` `val-tokens` `text` `resume` |
| filenames / name fragments | `plot-name` `save-name` `plot-suffix` |
| booleans | `fixed-batch` `plot-loss` `save` `sample-report` |

`sample-report` generates from a fixed prompt battery after training and writes
`checkpoints/<save-name stem>.md`. Pair it with `save` so the checkpoint and its
report share a name.

`plot-suffix` is appended to the generated plot filename (and ignored when
`plot-name` is set), so `"plot-suffix": "widevocab"` gives
`loss_blk64_emb128_..._seed42_widevocab.png`. Like the other name flags it's
restricted to `[A-Za-z0-9._-]`.

Numeric flags are also range-checked, to catch `"steps": 2000000` before it
costs you a day. Paths must stay inside the repo and must already exist.
There is no shell involved: the command is assembled as an argv list, so
nothing in a job file can become a shell token.

## Where things land

| path | what |
| --- | --- |
| `queue/*.json` | pending jobs, run oldest-first, one at a time |
| `queue/done/` | jobs that ran (prefixed with a timestamp) |
| `queue/failed/` | jobs rejected by validation, with the reason in `runs/` |
| `queue/cancelled/` | jobs cancelled from the control page before they started |
| `runs/tb/<id>/` | TensorBoard event files for the run |
| `runs/<id>.commands.jsonl` / `.events.jsonl` / `.live.json` | live-control inbox, applied-command log, heartbeat |
| `runs/<id>.log` | full training output, unbuffered so it streams live |
| `runs/<id>.status.json` | config, argv, timing, and parsed final metrics |
| `runs/index.jsonl` | one line per run, for quick history |
| `plots/` | plots, named by hyperparameters as before |
| `runner/runner.log` | the watcher's own log |

`status.json` carries `metrics` with `eval_train_loss`, `eval_val_loss`,
`full_val_loss`, the full-val curve as `[[step, loss], ...]`, the parameter
count and the plot path — so a finished run can be summarized without
re-parsing the training log.

## Notes

- Jobs run **serially**. Two MPS jobs at once would just fight over the GPU.
- The Mac has to be awake. Closing the lid sleeps it and the watcher stops
  with everything else. Power + "Prevent automatic sleeping when the display
  is off" in Battery settings, or Amphetamine's closed-display mode.
- While a job runs, the runner holds a `caffeinate -i` assertion so the Mac
  doesn't idle to sleep mid-run. It can't stop lid-close sleep.
- `KeepAlive` means launchd restarts the watcher if it dies, and starts it
  again at login. A job that was mid-run when the Mac slept does **not**
  resume; when the watcher next starts, it marks that status `interrupted`.
- `runs/` and `queue/done/` accumulate. Consider adding `runs/` and
  `queue/` to `.gitignore` if you don't want run history in git.
- If a job is rejected, look in `runs/*.status.json` for `"status":
  "rejected"` and the reason.
