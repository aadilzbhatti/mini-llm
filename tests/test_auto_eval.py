"""Auto-eval: which finished runs get evaluated, one at a time, recorded in their status."""

import json
import stat

from mini_llm.auto_eval import AutoEvaluator


def fake_eval(tmp_path, fail_for=""):
    """Stands in for `python -m mini_llm.evals <ckpt>`: records each call, fails on request."""
    script = tmp_path / "python"
    script.write_text(f"""#!/bin/sh
echo "$3" >> "{tmp_path}/calls.txt"
case "$3" in *{fail_for or "NEVER"}*) exit 3 ;; esac
exit 0
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def run(repo, run_id, finished, **extra):
    status = {
        "run_id": run_id,
        "kind": "train",
        "status": "completed",
        "finished": finished,
        "args": {"val-tokens": "data/v.pt"},
        "log": f"runs/{run_id}.log",
        **extra,
    }
    (repo / "runs" / f"{run_id}.status.json").write_text(json.dumps(status))


def test_evaluates_new_finished_runs_once(tmp_path):
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    (repo / "checkpoints").mkdir()
    (repo / "runs" / "auto_eval.json").write_text(json.dumps({"since": "2026-09-28T00:00:00Z"}))
    for name in ("local.pt", "modal_x.pt", "bad.pt"):
        (repo / "checkpoints" / name).write_bytes(b"")
    run(repo, "local", "2026-09-28T01:00:00Z")
    (repo / "runs" / "local.log").write_text("step 1 | loss 1\nSaved to checkpoints/local.pt\n")
    run(
        repo,
        "modal",
        "2026-09-28T01:00:00Z",
        remote={"provider": "modal"},
        args={"val-tokens": "v", "save-name": "modal_x.pt"},
    )
    run(
        repo,
        "bad",
        "2026-09-28T01:00:00Z",
        remote={"provider": "modal"},
        args={"val-tokens": "v", "save-name": "bad.pt"},
    )
    run(repo, "old", "2026-09-27T23:00:00Z")  # before auto-eval was on
    run(repo, "noval", "2026-09-28T01:00:00Z", args={})  # no val set
    run(
        repo,
        "pending",
        "2026-09-28T01:00:00Z",
        remote={"provider": "modal"},
        args={"val-tokens": "v", "save-name": "not_imported.pt"},
    )

    ev = AutoEvaluator(repo, python=fake_eval(tmp_path, fail_for="bad.pt"))
    assert sorted(ev.scan()) == ["bad", "local", "modal"]
    ev.jobs.join()
    assert ev.scan() == []  # nothing is evaluated twice, and a failure isn't retried

    st = lambda r: json.loads((repo / "runs" / f"{r}.status.json").read_text()).get("eval")
    assert st("local")["state"] == "done" and st("local")["report"] == "local"
    assert st("modal")["state"] == "done" and st("modal")["report"] == "modal_x"
    assert st("bad")["state"] == "failed" and "exited 3" in st("bad")["error"]
    assert st("old") is None and st("noval") is None and st("pending") is None
    calls = (tmp_path / "calls.txt").read_text().split()
    assert len(calls) == 3 and len(set(calls)) == 3


def test_one_eval_task_per_model(tmp_path):
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    (repo / "checkpoints").mkdir()
    (repo / "runs" / "auto_eval.json").write_text(json.dumps({"since": "2026-09-28T00:00:00Z"}))
    (repo / "checkpoints" / "modal_a.pt").write_bytes(b"")
    run(
        repo,
        "a",
        "2026-09-28T01:00:00Z",
        remote={"provider": "modal"},
        args={"val-tokens": "v", "save-name": "modal_a.pt", "sample-report": True},
    )
    script = tmp_path / "python"
    script.write_text(f'#!/bin/sh\necho "$*" >> "{tmp_path}/calls.txt"\nexit 0\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    ev = AutoEvaluator(repo, python=str(script))
    ev.scan()
    ev.jobs.join()
    # evals, samples, benchmark and summaries are all one `mini_llm.evals` run
    assert (tmp_path / "calls.txt").read_text().splitlines() == [
        f"-m mini_llm.evals {repo / 'checkpoints' / 'modal_a.pt'}"
    ]
