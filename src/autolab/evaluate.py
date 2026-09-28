"""The evaluation cascade: AlphaEvolve's `h`, as a resumable state machine.

Each program moves through the stages below. `advance(program)` does as much as
it can without waiting and returns. CPU stages run inline; GPU stages submit a
Modal trial and return, and a later call (from the daemon, every cycle) picks up
the collected result. The first failing stage rejects the program with a reason.
All state is in the program's JSON, so the daemon can restart at any point.

    static   diff already applied + scope-checked at creation; forbidden names
             in blocks; hparams within [evolve.hparams]; the program imports, and
             `mini_llm` resolves to the program's own code
    cpu      pytest over [evolve] cpu_tests (the full protected suite) against the
             program's code, with the causal-leak/shape test at the program's own size
    params   parameter count <= param_cap_mult x the initial program's
    screen   Modal, screen_tokens, seed 1. Also the smoke test: no NaN, no
             divergence, throughput >= throughput_floor x initial. Passes if within
             screen_margin of the incumbent's screen mean (screens overstate gains,
             so they can only reject)
    full     Modal, full_tokens, seed 1
    confirm  only if the full result beats the incumbent's mean by confirm_trigger_sigma x
             the full seed std: confirm_seeds more.
             Accepted as the new incumbent iff the mean over all seeds beats it by
             more than accept_sigma x the full-budget seed std; otherwise a contender.

A separate smoke stage was folded into the screen: on Modal a 640-step screen
costs ~$0.03, barely more than the container overhead a separate smoke run would
add (decision log, 2026-09-27).

    uv run autolab evolve init                              # session + initial program p0
    uv run autolab evolve propose --diff patch.txt --rationale "..." [--parent p0] [--hparams '{..}']
    uv run autolab evolve status
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from autolab.activity import set_activity
from autolab.config import REPO_ROOT
from autolab.program import (
    Program,
    ScopeError,
    apply_diffs,
    base_sources,
    extract_blocks,
    load,
    materialize,
    render,
    save,
    static_violations,
    validate_hparams,
)

STATE_ROOT = REPO_ROOT / "autolab" / "state" / "evolve"  # one subdirectory per session


def active_session_name(root: Path | None = None) -> str:
    root = root or STATE_ROOT  # resolved at call time, so tests can redirect STATE_ROOT
    try:
        return (root / "ACTIVE").read_text().strip()
    except OSError:
        return "default"


def set_active_session(name: str, root: Path | None = None) -> None:
    root = root or STATE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    (root / "ACTIVE").write_text(name + "\n")
MODEL_KEYS = ("n_embd", "n_head", "n_layer", "dropout")
OPTIM_KEYS = ("batch_size", "lr", "min_lr", "warmup_steps", "weight_decay")
VOCAB = 50257
DONE = {"rejected", "evaluated", "contender", "accepted"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evolve_cfg() -> dict:
    return tomllib.loads((REPO_ROOT / "autolab" / "config.toml").read_text())["evolve"]


@dataclass
class Paths:
    """Where one session lives. The default is the active session (STATE_ROOT/ACTIVE)."""

    root: Path = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.root is None:
            self.root = STATE_ROOT / active_session_name()

    @property
    def session(self) -> Path:
        return self.root / "session.json"

    @property
    def programs(self) -> Path:
        return self.root / "programs"

    def work(self, pid: str) -> Path:
        return self.root / "work" / pid


# --- session -----------------------------------------------------------------------------


def load_session(paths: Paths | None = None) -> dict | None:
    paths = paths or Paths()
    try:
        return json.loads(paths.session.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save_session(session: dict, paths: Paths | None = None) -> None:
    paths = paths or Paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    tmp = paths.session.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session, indent=2))
    tmp.replace(paths.session)


def _report(run_id: str, runs_dir: Path) -> dict | None:
    try:
        return json.loads((runs_dir / run_id / "report.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def init_session(base_commit: str, hparams: dict, runs: dict[str, list[str]], cfg: dict | None = None,
                 paths: Paths | None = None, runs_dir: Path | None = None, repo: Path = REPO_ROOT,
                 budgets: dict | None = None, name: str = "", blocks: dict | None = None,
                 dataset_id: str | None = None, rationale: str = "") -> dict:
    """Create the session and its initial program p0 from already-run baseline trials.

    `runs` maps "screen"/"full" to finished run ids of p0 on different seeds; they give
    the incumbent's scores, the seed noise, the wall-clock caps and the throughput floor.
    """
    cfg = cfg or evolve_cfg()
    paths = paths or Paths()
    runs_dir = runs_dir or repo / "autolab" / "runs"
    if paths.session.exists():
        raise FileExistsError(f"session already exists: {paths.session}")
    files = base_sources(repo, base_commit)
    reports = {k: [_report(r, runs_dir) for r in ids] for k, ids in runs.items()}
    if any(r is None for rs in reports.values() for r in rs):
        raise ValueError("every initial run needs a collected report.json")

    def losses(k):
        return [r["summary"]["final_full_val_loss"] for r in reports[k]]

    def walls(k):
        return [r["performance"]["wall_s"] for r in reports[k]]

    p0 = Program(id="p0", parent_id=None, base_commit=base_commit, blocks=blocks or extract_blocks(files), hparams=hparams,
                 rationale=rationale or "Initial program: the owner's emb256/blk128/bs64 regime at the base commit.",
                 stage="done", status="accepted", runs=runs)
    p0.scores = _scores(mean(losses("screen")), losses("full"), reports["full"][0])
    p0.scores["screen_losses"] = losses("screen")
    session = {
        "name": name or paths.root.name,
        "created": now_iso(),
        "dataset_id": dataset_id or cfg["dataset_id"],
        # Fixed for the session: what "better" means (token budgets, eval settings).
        "budgets": {**(budgets or {"screen_tokens": cfg["screen_tokens"], "full_tokens": cfg["full_tokens"],
                                   "eval": dict(cfg["eval"])}),
                    # M8: fixed tokens per optimizer step (batch x context), from the starting program
                    "tokens_per_step": (budgets or {}).get("tokens_per_step")
                    or hparams["batch_size"] * int(hparams.get("block_size", cfg["block_size"]))},
        "base_commit": base_commit,
        "incumbent": "p0",
        "initial_params": reports["full"][0]["scale"]["params"],
        "initial_tokens_per_sec": mean(r["performance"]["tokens_per_sec"] for r in reports["full"]),
        "noise": {k: {"n": len(losses(k)), "mean": mean(losses(k)), "std": stdev(losses(k))} for k in ("screen", "full")},
        "wall_caps": {k: round(cfg["wall_cap_mult"] * mean(walls(k)), 1) for k in ("screen", "full")},
        "incumbent_history": [{"at": now_iso(), "program": "p0", "full_mean": mean(losses("full"))}],
        "next_id": 1,
    }
    save(p0, paths.programs)
    save_session(session, paths)
    return session


def _scores(screen: float | None, fulls: list[float], report: dict | None) -> dict:
    s = {"screen_loss": screen, "full_losses": fulls}
    if fulls:
        s.update(full_mean=mean(fulls), n_seeds=len(fulls), full_std=stdev(fulls) if len(fulls) > 1 else None,
                 neg_full_val_loss=-mean(fulls))
    if report:
        s.update(tokens_per_sec=report["performance"]["tokens_per_sec"], params=report["scale"]["params"],
                 neg_params=-report["scale"]["params"])
    return s


# --- creating children ---------------------------------------------------------------------


def programs(paths: Paths | None = None) -> dict[str, Program]:
    paths = paths or Paths()
    return {p.stem: load(p) for p in sorted(paths.programs.glob("*.json"), key=lambda f: (len(f.stem), f.stem))}


def propose(parent_id: str, diffs: list[dict], hparams_patch: dict | None = None, rationale: str = "",
            created_by: str = "human", paths: Paths | None = None, repo: Path = REPO_ROOT) -> Program:
    """Create (and save) a child of `parent_id`. A scope failure is saved as a rejected program."""
    paths = paths or Paths()
    session = load_session(paths)
    if session is None:
        raise RuntimeError("no evolve session; run `autolab evolve init` first")
    parent = load(paths.programs / f"{parent_id}.json")
    pid = f"p{session['next_id']}"
    session["next_id"] += 1
    save_session(session, paths)
    child = Program(id=pid, parent_id=parent_id, base_commit=parent.base_commit, blocks=dict(parent.blocks),
                    hparams={**parent.hparams, **(hparams_patch or {})}, rationale=rationale,
                    created_by=created_by, diffs=diffs)
    try:
        if diffs:
            child.blocks = apply_diffs(base_sources(repo, parent.base_commit), parent.blocks, diffs)
    except ScopeError as exc:
        _reject(child, "static", f"scope: {exc}")
    save(child, paths.programs)
    return child


def port(program: Program, source_session: str, paths: Paths | None = None) -> Program:
    """Propose another session's program (e.g. accepted after a data switch) as a child of this
    session's incumbent, so it is re-evaluated from scratch on this session's data and budget."""
    paths = paths or Paths()
    session = load_session(paths)
    pid = f"p{session['next_id']}"
    session["next_id"] += 1
    save_session(session, paths)
    child = Program(id=pid, parent_id=session["incumbent"], base_commit=program.base_commit, blocks=dict(program.blocks),
                    hparams=dict(program.hparams), created_by=f"port:{source_session}/{program.id}",
                    rationale=f"Ported from {source_session}/{program.id} "
                              f"({program.scores.get('full_mean', float('nan')):.4f} there): {program.rationale}")
    child.meta = {"ported_from": f"{source_session}/{program.id}"}
    save(child, paths.programs)
    return child


# --- the state machine -----------------------------------------------------------------------


def _record(p: Program, stage: str, ok: bool, detail: str = "", **extra) -> None:
    p.stages.append({"stage": stage, "ok": ok, "at": now_iso(), "detail": detail[:4000], **extra})


def _reject(p: Program, stage: str, reason: str) -> None:
    p.stage, p.status, p.reason = stage, "rejected", reason[:2000]
    _record(p, stage, False, reason)


def block_size(p: Program, cfg: dict) -> int:
    """Context length: an evolvable hparam since M8 (older programs: the config's fixed value)."""
    return int(p.hparams.get("block_size", cfg["block_size"]))


def tokens_per_step(session: dict, cfg: dict) -> int:
    return int(session.get("budgets", {}).get("tokens_per_step") or 64 * cfg["block_size"])


def _model_cfg(p: Program, cfg: dict) -> dict:
    return {"vocab_size": VOCAB, "block_size": block_size(p, cfg), **{k: p.hparams[k] for k in MODEL_KEYS}}


def _env(src: Path, extra: dict | None = None) -> dict:
    return {**os.environ, "PYTHONPATH": str(src), "AUTOLAB_FORCE_CPU": "1", "HF_HUB_OFFLINE": "1",
            "AUTOLAB_IN_CASCADE": "1", **(extra or {})}


def stage_static(p: Program, cfg: dict, paths: Paths, repo: Path) -> Path | None:
    set_activity("static", f"{p.id}: static checks (scope, forbidden code, hparams, import)", program=p.id)
    problems = static_violations(p.blocks) + validate_hparams(
        {k: v for k, v in p.hparams.items()}, cfg["hparams"])
    session = load_session(paths) or {}
    tps = tokens_per_step(session, cfg)
    if p.hparams["batch_size"] * block_size(p, cfg) != tps:
        problems.append(f"batch_size x block_size = {p.hparams['batch_size']} x {block_size(p, cfg)} must equal "
                        f"{tps} tokens per update (keep tokens/update fixed when changing context)")
    if problems:
        _reject(p, "static", "; ".join(problems))
        return None
    work = paths.work(p.id)
    shutil.rmtree(work, ignore_errors=True)
    src = materialize(render(base_sources(repo, p.base_commit), p.blocks), work)
    r = subprocess.run([sys.executable, "-c", "import mini_llm, mini_llm.model, mini_llm.train; print(mini_llm.__file__)"],
                       env=_env(src), capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        _reject(p, "static", "import failed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
        return None
    if not Path(r.stdout.strip()).resolve().is_relative_to(src.resolve()):
        _reject(p, "static", f"mini_llm resolved to {r.stdout.strip()}, not the program's code")
        return None
    _record(p, "static", True)
    return src


FAILED_LINE = re.compile(r"^FAILED (\S+)", re.M)


CAUSAL_TEST = "tests/autolab/test_causal_leak.py"


def stage_cpu(p: Program, src: Path, cfg: dict, repo: Path) -> bool:
    """Shape test, then causal/label-leak tests, then the full protected suite.

    Separate runs so the reject reason is specific: a shape bug crashes every forward
    pass (the causal tests too), and a leak must be reported as a leak.
    """
    env = _env(src, {"AUTOLAB_TEST_MODEL": json.dumps(_model_cfg(p, cfg))})
    steps = [("shape", [f"{repo / CAUSAL_TEST}::test_shapes_and_backward"]),
             ("causal-leak", [str(repo / CAUSAL_TEST), "-k", "leak or targets"]),
             ("tests", [str(repo / t) for t in cfg["cpu_tests"]]
              + [f"--deselect={d}" for d in cfg.get("cpu_test_deselect", [])])]  # rootdir-relative node ids
    summary = []
    for i, (kind, args) in enumerate(steps, 1):
        set_activity("cpu", f"{p.id}: CPU checks {i}/3 — {'full protected test suite' if kind == 'tests' else kind + ' test'}",
                     program=p.id)
        cmd = [sys.executable, "-m", "pytest", "-q", "-rf", "--no-header", "-p", "no:cacheprovider",
               "-p", "autolab.pytest_seed", *args]
        try:
            r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=cfg["cpu_test_timeout_s"])
        except subprocess.TimeoutExpired:
            _reject(p, "cpu", f"{kind}: timed out after {cfg['cpu_test_timeout_s']}s")
            return False
        tail = (r.stdout + r.stderr)[-3000:]
        if r.returncode != 0:
            failed = FAILED_LINE.findall(r.stdout)
            broken = _suite_broken(failed, cmd, env, repo, cfg) if kind == "tests" else []
            if broken:
                # The same tests fail on the incumbent's code: the test suite (or the checkout) is broken,
                # not the candidate. Don't reject: requeue and let the controller pause with an alert.
                p.stage, p.status, p.reason = "static", "queued", f"infra: protected tests fail on the incumbent too: {', '.join(broken[:4])}"
                _record(p, "cpu", False, p.reason, infra=True)
                _write_infra_alert(p.reason)
                return False
            _reject(p, "cpu", f"{kind}: {len(failed)} failed: {', '.join(failed[:6]) or tail[-500:]}")
            p.stages[-1]["detail"] = tail
            return False
        summary.append(f"{kind}: {tail.strip().splitlines()[-1] if tail.strip() else 'ok'}")
    _record(p, "cpu", True, "; ".join(summary))
    return True


INFRA_ALERT = REPO_ROOT / "autolab" / "state" / "infra_alert.json"


def _suite_broken(failed: list[str], cmd: list[str], env: dict, repo: Path, cfg: dict) -> list[str]:
    """Canary: rerun a candidate's failing protected tests against the active incumbent's code.
    Returns the ones that fail there too (so the failure isn't the candidate's)."""
    if not failed:
        return []
    try:
        paths = Paths()
        session = load_session(paths)
        inc = load(paths.programs / f"{session['incumbent']}.json")
        src = materialize(render(base_sources(repo, inc.base_commit), inc.blocks), paths.work(f"canary-{inc.id}"))
        base_env = {**env, "PYTHONPATH": str(src), "AUTOLAB_TEST_MODEL": json.dumps(_model_cfg(inc, cfg))}
        ids = [f.split(" ")[0] for f in failed]
        head = [c for c in cmd if not (c.startswith(str(repo / "tests")) or c.startswith("--deselect"))]
        r = subprocess.run(head + ids, cwd=repo, env=base_env, capture_output=True, text=True,
                           timeout=cfg["cpu_test_timeout_s"])
        return FAILED_LINE.findall(r.stdout)
    except Exception:  # noqa: BLE001 - a broken canary must not hide a real rejection
        return []


def _write_infra_alert(reason: str) -> None:
    INFRA_ALERT.parent.mkdir(parents=True, exist_ok=True)
    INFRA_ALERT.write_text(json.dumps({"at": now_iso(), "reason": reason}, indent=2))


def stage_params(p: Program, src: Path, cfg: dict, session: dict) -> bool:
    set_activity("params", f"{p.id}: counting parameters", program=p.id)
    code = ("import json,sys; from mini_llm.config import ModelConfig, build_model; "
            "m = build_model(ModelConfig(**json.loads(sys.argv[1]))); print(sum(x.numel() for x in m.parameters()))")
    r = subprocess.run([sys.executable, "-c", code, json.dumps(_model_cfg(p, cfg))], env=_env(src),
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        _reject(p, "params", "could not build the model: " + (r.stderr.strip().splitlines() or ["?"])[-1])
        return False
    n = int(r.stdout.strip())
    cap = cfg["param_cap_mult"] * session["initial_params"]
    p.scores["params"] = n
    if n > cap:
        _reject(p, "params", f"{n:,} parameters > cap {cap:,.0f} ({cfg['param_cap_mult']}x initial)")
        return False
    _record(p, "params", True, f"{n:,} parameters", params=n)
    return True


def _request(p: Program, stage: str, seed: int, cfg: dict, session: dict):
    from autolab.config import load_config
    from autolab.trainer import Budget, TrainRequest

    budgets = session.get("budgets") or {"screen_tokens": cfg["screen_tokens"], "full_tokens": cfg["full_tokens"],
                                         "eval": cfg["eval"]}
    tokens = budgets["screen_tokens"] if stage == "screen" else budgets["full_tokens"]
    cap = session["wall_caps"]["screen" if stage == "screen" else "full"]
    return TrainRequest(
        run_id=f"ev-{session.get('name', 's')}-{p.id}-{stage}-s{seed}", dataset_id=session.get("dataset_id", cfg["dataset_id"]),
        train_tokens=str(load_config().datasets_dir / session.get("dataset_id", cfg["dataset_id"]) / "train.pt"),
        budget=Budget(tokens=tokens, wall_clock_s=cap), seed=seed,
        model={"block_size": block_size(p, cfg), **{k: p.hparams[k] for k in MODEL_KEYS}},
        optim={k: p.hparams[k] for k in OPTIM_KEYS}, eval=dict(budgets["eval"]),
        suite=stage in ("full", "confirm", "measure"))


def _submit(p: Program, stage: str, seeds: list[int], cfg: dict, session: dict, paths: Paths, submit) -> bool:
    src = paths.work(p.id) / "src"
    if not src.exists():  # e.g. after a restart that cleaned work/: rebuild from the stored blocks
        src = materialize(render(base_sources(REPO_ROOT, p.base_commit), p.blocks), paths.work(p.id))
    ids = []
    for seed in seeds:
        req = _request(p, stage, seed, cfg, session)
        try:
            submit(req, cfg["gpu"], src)
        except FileExistsError:
            pass  # already submitted (restart between submit and save)
        except RuntimeError as exc:  # cost cap: wait, retry next cycle
            p.status, p.reason = "blocked", str(exc)
            return False
        ids.append(req.run_id)
    p.runs.setdefault(stage, []).extend(ids)
    p.stage, p.status, p.reason = stage, "running", ""
    set_activity("gpu", f"{p.id}: {stage} submitted to Modal ({len(ids)} run{'s' if len(ids) > 1 else ''})", program=p.id)
    return True


METRIC_KEYS = {  # report path -> metric name (report["eval"] comes from autolab.evalsuite over mini_llm.evals)
    ("eval", "context_capability", "long_range_score"): "long_range_score",
    ("eval", "context_capability", "effective_context"): "effective_context",
    ("eval", "context_capability", "context_benefit_nats"): "context_benefit_nats",
    ("eval", "quality", "short_context_loss"): "short_context_loss",
    ("eval", "inference", "decode_tokens_per_s"): "decode_tokens_per_s",
    ("eval", "inference", "decode_ms_per_token"): "decode_ms_per_token",
    ("eval", "inference", "prefill_ms"): "prefill_ms",
    ("eval", "inference", "peak_inference_mem_bytes"): "peak_inference_mem_bytes",
    ("eval", "inference", "fwd_flops_per_token"): "fwd_flops_per_token",
    ("eval", "inference", "params"): "params",
    ("eval", "context"): "context",
    ("eval", "training", "train_tokens_per_sec"): "train_tokens_per_sec",
    ("eval", "training", "peak_train_mem_bytes"): "peak_train_mem_bytes",
    ("performance", "train_wall_s"): "train_wall_s",
}


def run_metrics(rep: dict | None) -> dict | None:
    """The multi-objective metrics of one run (None if the eval suite didn't run)."""
    if not rep or not rep.get("eval"):
        return None
    out = {}
    for path, name in METRIC_KEYS.items():
        v = rep
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)):
            out[name] = v
    return out


def collect_metrics(p: Program, runs_dir: Path) -> dict | None:
    """Mean of each metric over the program's full-budget runs (full, confirm, measure) that have them."""
    rows = [m for rid in p.runs.get("full", []) + p.runs.get("confirm", []) + p.runs.get("measure", [])
            if (m := run_metrics(_report(rid, runs_dir)))]
    if not rows:
        return None
    keys = set().union(*rows)
    agg = {k: mean(r[k] for r in rows if k in r) for k in keys}
    agg["n_runs"] = len(rows)
    return agg


def _results(p: Program, stage: str, calls: dict, runs_dir: Path) -> list[tuple[str, dict | None, str | None]] | None:
    """[(run_id, report, error)] once every run of the stage is done, else None."""
    out = []
    for rid in p.runs.get(stage, []):
        c = calls.get(rid)
        if c is None or c["state"] == "pending":
            return None
        out.append((rid, _report(rid, runs_dir), c.get("error") if c["state"] == "failed" else None))
    return out


def _health(report: dict | None, error: str | None, session: dict, cfg: dict) -> str | None:
    if error or report is None:
        return f"run failed: {(error or 'no report')[:300]}"
    if report["health"]["nan_or_inf"]:
        return "NaN/inf in training" + (" (trainer traceback)" if report["health"].get("trainer_traceback") else "")
    if report["summary"].get("final_full_val_loss") is None:
        return "no final full val loss"
    tps = report["performance"].get("tokens_per_sec") or 0
    floor = cfg["throughput_floor"] * session["initial_tokens_per_sec"]
    if tps < floor:
        return f"throughput {tps:,.0f} tok/s < floor {floor:,.0f} ({cfg['throughput_floor']}x initial)"
    return None


def sigma(session: dict, cfg: dict | None = None) -> float:
    """Seed noise used for decisions: the session's measured full-budget std, floored at [evolve]
    noise_floor. A 3-seed std is a very noisy estimate: s2+data40k@122M measured 0.0069 where every
    other session measured 0.031-0.043, which would have made the accept bar only 0.014 deep."""
    cfg = cfg or evolve_cfg()
    return max(session["noise"]["full"]["std"], float(cfg.get("noise_floor", 0.0)))


def incumbent(session: dict, progs: dict[str, Program]) -> Program:
    return progs[session["incumbent"]]


def advance(p: Program, session: dict, cfg: dict, calls: dict, paths: Paths | None = None, repo: Path = REPO_ROOT,
            runs_dir: Path | None = None, submit=None, log=print) -> Program:
    paths = paths or Paths()
    runs_dir = runs_dir or repo / "autolab" / "runs"
    if submit is None:
        from autolab.modal_backend import submit as modal_submit

        def submit(req, gpu, src):
            return modal_submit(req, gpu, src_root=src)

    if p.status in DONE:
        return p
    inc = load(paths.programs / f"{session['incumbent']}.json")

    if p.stage in ("static", "cpu", "params") and p.status in ("queued", "running"):
        p.status = "running"
        save(p, paths.programs)  # CPU gates take ~2 min; let the dashboard show it
        src = stage_static(p, cfg, paths, repo)
        if src and stage_cpu(p, src, cfg, repo) and stage_params(p, src, cfg, session):
            p.stage, p.status = "screen", "queued"
        save(p, paths.programs)
        if p.status == "queued" and p.reason.startswith("infra:"):
            log(f"{p.id}: {p.reason}")
            return p
        if p.status == "rejected":
            log(f"{p.id}: rejected at {p.stage}: {p.reason[:200]}")
            _cleanup(p, paths)
            return p

    if p.stage == "screen":
        if p.status in ("queued", "blocked"):
            _submit(p, "screen", [1], cfg, session, paths, submit)
        else:
            res = _results(p, "screen", calls, runs_dir)
            if res is not None:
                rid, rep, err = res[0]
                bad = _health(rep, err, session, cfg)
                if bad:
                    _reject(p, "screen", f"smoke: {bad}")
                else:
                    loss = rep["summary"]["final_full_val_loss"]
                    p.scores = {**p.scores, **_scores(loss, [], rep)}
                    inc_screen = mean(inc.scores.get("screen_losses") or [inc.scores["screen_loss"]])
                    limit = inc_screen + cfg["screen_margin"]
                    if loss > limit:
                        _reject(p, "screen", f"screen {loss:.4f} > incumbent {inc_screen:.4f} + margin {cfg['screen_margin']}")
                    else:
                        _record(p, "screen", True, f"screen {loss:.4f} vs incumbent {inc_screen:.4f}", loss=loss)
                        p.stage, p.status = "full", "queued"

    if p.stage == "full":
        if p.status in ("queued", "blocked"):
            _submit(p, "full", [1], cfg, session, paths, submit)
        else:
            res = _results(p, "full", calls, runs_dir)
            if res is not None:
                rid, rep, err = res[0]
                bad = _health(rep, err, session, cfg)
                if bad:
                    _reject(p, "full", bad)
                else:
                    loss = rep["summary"]["final_full_val_loss"]
                    p.scores = {**p.scores, **_scores(p.scores.get("screen_loss"), [loss], rep)}
                    if (mt := collect_metrics(p, runs_dir)):
                        p.scores["metrics"] = mt
                    inc_mean = inc.scores["full_mean"]
                    trigger = inc_mean - cfg.get("confirm_trigger_sigma", 0.0) * sigma(session, cfg)
                    _record(p, "full", True, f"full {loss:.4f} vs incumbent mean {inc_mean:.4f} (confirm below {trigger:.4f})",
                            loss=loss)
                    if loss < trigger:
                        p.stage, p.status = "confirm", "queued"
                    else:
                        p.stage, p.status, p.reason = "done", "evaluated", \
                            f"full {loss:.4f} not below {trigger:.4f} (incumbent mean - {cfg.get('confirm_trigger_sigma', 0.0)}σ)"

    if p.stage == "confirm":
        if p.status in ("queued", "blocked"):
            _submit(p, "confirm", list(cfg["confirm_seeds"]), cfg, session, paths, submit)
        else:
            res = _results(p, "confirm", calls, runs_dir)
            if res is not None:
                good = [(rid, rep) for rid, rep, err in res if _health(rep, err, session, cfg) is None]
                fulls = p.scores["full_losses"] + [rep["summary"]["final_full_val_loss"] for _, rep in good]
                p.scores = {**p.scores, **_scores(p.scores.get("screen_loss"), fulls, None)}
                if (mt := collect_metrics(p, runs_dir)):
                    p.scores["metrics"] = mt
                sig = sigma(session, cfg)
                bar = inc.scores["full_mean"] - cfg["accept_sigma"] * sig
                m = p.scores["full_mean"]
                detail = f"mean {m:.4f} over {len(fulls)} seeds; bar {bar:.4f} = incumbent {inc.scores['full_mean']:.4f} - {cfg['accept_sigma']}x{sig:.4f}"
                if len(fulls) >= 2 and m < bar:
                    _record(p, "confirm", True, detail)
                    p.stage, p.status, p.reason = "done", "accepted", detail
                    session["incumbent"] = p.id
                    session.setdefault("incumbent_history", []).append({"at": now_iso(), "program": p.id, "full_mean": m})
                    save_session(session, paths)
                    log(f"{p.id}: ACCEPTED as new incumbent ({detail})")
                    set_activity("accepted", f"{p.id} ACCEPTED: {detail}", program=p.id)
                else:
                    _record(p, "confirm", False, detail)
                    p.stage, p.status, p.reason = "done", "contender", f"not significant: {detail}"

    save(p, paths.programs)
    if p.status in DONE:
        log(f"{p.id}: {p.status} ({p.reason[:200]})")
        _cleanup(p, paths)
    return p


def _cleanup(p: Program, paths: Paths) -> None:
    shutil.rmtree(paths.work(p.id), ignore_errors=True)  # blocks live in the JSON; re-materializable


def all_session_paths() -> list[Paths]:
    """Every session on disk (the active one last, so its acceptances are the freshest)."""
    active = active_session_name()
    names = sorted(d.name for d in STATE_ROOT.iterdir() if (d / "session.json").exists()) if STATE_ROOT.exists() else []
    return [Paths(STATE_ROOT / n) for n in names if n != active] + ([Paths(STATE_ROOT / active)] if active in names else [])


def advance_everything(log=print) -> list[str]:
    """Advance unfinished programs and data checks in every session. A session switch can happen
    while programs of the old session are mid-evaluation; they must still be judged."""
    from autolab.modal_backend import load_calls

    touched = []
    for paths in all_session_paths():
        touched += [f"{paths.root.name}/{pid}" for pid in advance_all(log=log, paths=paths)]
        touched += [f"{paths.root.name}/{c}" for c in advance_data_checks(load_calls(), paths=paths, log=log)]
        touched += [f"{paths.root.name}/{pid}" for pid in advance_measures(load_calls(), paths=paths, log=log)]
    return touched


def advance_all(log=print, paths: Paths | None = None) -> list[str]:
    """One pass over every unfinished program of one session (default: the active one)."""
    paths = paths or Paths()
    session = load_session(paths)
    if session is None:
        return []
    from autolab.modal_backend import load_calls

    cfg, calls, touched = evolve_cfg(), load_calls(), []
    for pid, p in programs(paths).items():
        if p.status in DONE:
            continue
        before = (p.stage, p.status)
        session = load_session(paths)  # an acceptance earlier in this pass may have moved the incumbent
        p = advance(p, session, cfg, calls, paths=paths, log=log)
        if (p.stage, p.status) != before:
            touched.append(pid)
    return touched


# --- data checks: does more training data beat the current program? --------------------------


def start_data_check(program_id: str, dataset_id: str, seeds: list[int] | None = None, paths: Paths | None = None,
                     submit=None, repo: Path = REPO_ROOT, tokens: int | None = None) -> dict:
    """Train `program_id` on a bigger dataset at the session's full budget, on several seeds.

    The daemon (advance_data_checks) judges it when the runs finish: more data "helped" iff the mean
    beats the program's own mean on the session's data by more than accept_sigma x the full seed std.
    """
    from autolab.config import load_config

    paths = paths or Paths()
    cfg, session = evolve_cfg(), load_session(paths)
    p = load(paths.programs / f"{program_id}.json")
    if p.scores.get("full_mean") is None:
        raise ValueError(f"{program_id} has no full-budget score to compare against")
    if not (load_config().datasets_dir / dataset_id / "train.pt").exists():
        raise FileNotFoundError(f"dataset {dataset_id} not built")
    if submit is None:
        from autolab.modal_backend import submit as modal_submit

        def submit(req, gpu, src):
            return modal_submit(req, gpu, src_root=src)

    work = paths.work(f"data-{dataset_id}-{program_id}")
    shutil.rmtree(work, ignore_errors=True)
    src = materialize(render(base_sources(repo, p.base_commit), p.blocks), work)
    seeds = seeds or [1, 2, 3]
    run_ids = []
    kind = "budget" if tokens else "data"
    tag = f"{dataset_id}@{tokens // 1_000_000}M" if tokens else dataset_id
    for seed in seeds:
        req = _request(p, "full", seed, cfg, session)
        req.run_id = f"ev-{session.get('name', 's')}-data-{tag}-{p.id}-s{seed}"
        req.dataset_id = dataset_id
        req.train_tokens = str(load_config().datasets_dir / dataset_id / "train.pt")
        if tokens:  # compute ladder: longer run, wall cap scaled with it
            scale = tokens / session["budgets"]["full_tokens"]
            req.budget.tokens, req.budget.wall_clock_s = tokens, round(req.budget.wall_clock_s * scale)
        submit(req, cfg["gpu"], src)
        run_ids.append(req.run_id)
    check = {"id": f"{tag}-{p.id}", "kind": kind, "program": p.id, "dataset": dataset_id, "tokens": tokens,
             "runs": run_ids, "baseline_mean": p.scores["full_mean"], "status": "running", "started": now_iso()}
    session.setdefault("data_checks", []).append(check)
    save_session(session, paths)
    return check


def advance_data_checks(calls: dict, paths: Paths | None = None, runs_dir: Path | None = None, log=print) -> list[str]:
    paths = paths or Paths()
    runs_dir = runs_dir or REPO_ROOT / "autolab" / "runs"
    session = load_session(paths)
    if not session or not session.get("data_checks"):
        return []
    cfg, done = evolve_cfg(), []
    for chk in session["data_checks"]:
        if chk["status"] != "running":
            continue
        if any(calls.get(r, {}).get("state", "pending") == "pending" for r in chk["runs"]):
            continue
        losses = [rep["summary"]["final_full_val_loss"] for r in chk["runs"]
                  if (rep := _report(r, runs_dir)) and rep["summary"].get("final_full_val_loss") is not None]
        sig = sigma(session, cfg)
        chk.update(losses=losses, finished=now_iso())
        if len(losses) < 2:
            chk.update(status="failed", verdict="too few finished runs")
        else:
            m = mean(losses)
            margin = cfg["accept_sigma"] * sig
            helped = m < chk["baseline_mean"] - margin
            chk.update(status="done", mean=m, std=stdev(losses), delta=m - chk["baseline_mean"],
                       delta_sigma=(m - chk["baseline_mean"]) / sig, helped=helped,
                       verdict=(f"{'helped' if helped else 'did not help'}: {m:.4f} vs {chk['baseline_mean']:.4f} "
                                f"({(m - chk['baseline_mean']) / sig:+.1f}σ; bar −{margin:.4f})"))
            # the notebook-style entry diagnose() reads for capacity_limited / data_limited evidence
            session.setdefault("notebook", []).append({"at": now_iso(),
                                                      "action": "raise_budget" if chk.get("kind") == "budget" else "build_dataset",
                                                      "dataset": chk["dataset"], "program": chk["program"],
                                                      "improved": helped, "accepted": helped, "detail": chk["verdict"]})
        log(f"data check {chk['id']}: {chk.get('verdict')}")
        done.append(chk["id"])
    if done:
        save_session(session, paths)
    for chk_id in done:
        for w in (paths.root / "work").glob(f"data-{chk_id.rsplit('-', 1)[0]}-*"):
            shutil.rmtree(w, ignore_errors=True)
    return done


# --- transfer checks: a program at a different data/budget than its session's --------------------


def start_transfer(label: str, program_ref: str, dataset_id: str, steps: int, seeds: list[int],
                   hparams: dict | None = None, variant: str | None = None, submit=None, repo: Path = REPO_ROOT) -> dict:
    """Run `session/program` (optionally with an hparam patch) on `dataset_id` for `steps` optimizer steps.

    For questions the fixed-budget sessions can't ask, e.g. "does what autolab found still help at
    the owner's current regime?". Runs are named tr-<label>-<variant>-s<seed> and listed in
    autolab/experiments/<label>.json so the dashboard's Experiments tab analyzes them.
    """
    import json as _json

    from autolab.config import load_config

    if submit is None:
        from autolab.modal_backend import submit as modal_submit

        def submit(req, gpu, src):
            return modal_submit(req, gpu, src_root=src)

    sname, pid = program_ref.split("/")
    paths = Paths(STATE_ROOT / sname)
    session, cfg = load_session(paths), evolve_cfg()
    p = load(paths.programs / f"{pid}.json")
    p.hparams = {**p.hparams, **(hparams or {})}
    variant = variant or f"{sname}-{pid}"
    work = STATE_ROOT / "transfer" / f"{label}-{variant}"
    shutil.rmtree(work, ignore_errors=True)
    src = materialize(render(base_sources(repo, p.base_commit), p.blocks), work)
    tokens = steps * p.hparams["batch_size"] * cfg["block_size"]
    speed = session.get("initial_tokens_per_sec") or 40_000
    cap = round(1.5 * tokens / speed * (p.hparams["n_layer"] / 4) + 300)  # generous: deeper = slower
    exp_path = repo / "autolab" / "experiments" / f"{label}.json"
    exp = _json.loads(exp_path.read_text()) if exp_path.exists() else {
        "id": label, "title": f"Transfer check: {label}", "gpu": cfg["gpu"], "dataset_id": dataset_id,
        "purpose": "", "model": {}, "optim": {}, "eval": session["budgets"]["eval"],
        "analysis": {"kind": "noise_and_ranking", "baseline_variant": "", "screen": "", "full": "full"}, "jobs": []}
    for seed in seeds:
        req = _request(p, "full", seed, cfg, session)
        req.run_id = f"tr-{label}-{variant}-s{seed}"
        req.dataset_id = dataset_id
        req.train_tokens = str(load_config().datasets_dir / dataset_id / "train.pt")
        req.budget.tokens, req.budget.wall_clock_s = tokens, cap
        submit(req, cfg["gpu"], src)
        exp["jobs"].append({"run_id": req.run_id, "variant": variant, "budget": "full", "seed": seed,
                            "tokens": tokens, "wall_clock_s": cap, "program": program_ref, "hparams": p.hparams})
    exp_path.write_text(_json.dumps(exp, indent=2))
    return exp


# --- measure: backfill multi-objective metrics for programs scored before the eval suite existed -------


def start_measure(program_id: str, paths: Paths | None = None, submit=None, repo: Path = REPO_ROOT, seed: int = 1) -> str:
    """One full-budget run of the program with the eval suite; its metrics join the program when done."""
    paths = paths or Paths()
    cfg, session = evolve_cfg(), load_session(paths)
    p = load(paths.programs / f"{program_id}.json")
    if submit is None:
        from autolab.modal_backend import submit as modal_submit

        def submit(req, gpu, src):
            return modal_submit(req, gpu, src_root=src)

    src = materialize(render(base_sources(repo, p.base_commit), p.blocks), paths.work(f"measure-{p.id}"))
    req = _request(p, "measure", seed, cfg, session)
    submit(req, cfg["gpu"], src)
    p.runs.setdefault("measure", []).append(req.run_id)
    save(p, paths.programs)
    session.setdefault("measures", {})[p.id] = req.run_id
    save_session(session, paths)
    set_activity("gpu", f"{p.id}: measuring quality/context/inference metrics ({req.run_id})", program=p.id)
    return req.run_id


def advance_measures(calls: dict, paths: Paths | None = None, runs_dir: Path | None = None, log=print) -> list[str]:
    paths = paths or Paths()
    runs_dir = runs_dir or REPO_ROOT / "autolab" / "runs"
    session = load_session(paths)
    if not session or not session.get("measures"):
        return []
    done = []
    for pid, rid in list(session["measures"].items()):
        c = calls.get(rid)
        if c is None or c["state"] == "pending":
            continue
        p = load(paths.programs / f"{pid}.json")
        mt = collect_metrics(p, runs_dir)
        if mt:
            p.scores["metrics"] = mt
            save(p, paths.programs)
            log(f"{pid}: metrics measured ({rid})")
        else:
            log(f"{pid}: measure run {rid} gave no metrics ({c.get('state')})")
        del session["measures"][pid]
        done.append(pid)
        shutil.rmtree(paths.work(f"measure-{pid}"), ignore_errors=True)
    if done:
        save_session(session, paths)
    return done
