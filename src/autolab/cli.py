"""`autolab` command line.

    autolab daemon [--interval 60] [--once]   collector/orchestrator loop (launchd runs this)
    autolab dashboard [--port 8766]           the web dashboard (launchd runs this)
    autolab modal <deploy|upload-data|submit|submit-batch|collect|status> ...
    autolab report <run_dir>...               rebuild report.json + diagnosis.json from local TB events
    autolab evolve init                       evolve session + initial program p0 (from the M3 prep runs)
    autolab evolve propose --diff FILE [--parent P] [--hparams JSON] [--rationale TEXT]
    autolab evolve generate [-n N] [--model M] new children from Claude (fallback: hparam mutation)
    autolab evolve advance                    one cascade pass now (the daemon does this every cycle)
    autolab evolve status
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["modal"]:
        from autolab.modal_backend import main as modal_main

        return modal_main(argv[1:])
    p = argparse.ArgumentParser(prog="autolab")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("daemon")
    d.add_argument("--interval", type=float, default=60.0)
    d.add_argument("--once", action="store_true")
    w = sub.add_parser("dashboard")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8766)
    sub.add_parser("modal", help="Modal backend commands (see autolab.modal_backend)")
    sub.add_parser("start", help="Enable the controller: the daemon proposes and evaluates on its own")
    sub.add_parser("resume", help="Clear a pause (usage limit / early rejects) and keep going")
    stp = sub.add_parser("stop", help="Stop proposing (in-flight programs finish)")
    stp.add_argument("--cancel-running", action="store_true", help="Also cancel running Modal trials")
    sub.add_parser("status", help="Controller, budget and session summary")
    sub.add_parser("session-report", help="Regenerate autolab/SESSION_1.md")
    rs = sub.add_parser("research", help="Research agent: audit the incumbent, write technique cards")
    rsub = rs.add_subparsers(dest="research_cmd", required=True)
    rr = rsub.add_parser("run")
    rr.add_argument("--trigger", default="Manual run: audit the current best program for outdated or missing techniques.")
    rr.add_argument("--result", default=None, help="Write the run's result JSON here (the controller uses this)")
    rsub.add_parser("cards")
    tr = sub.add_parser("transfer", help="Run a program at another dataset/budget (e.g. the owner's regime)")
    tr.add_argument("label")
    tr.add_argument("program", help="session/program, e.g. s2+data40k/p1")
    tr.add_argument("--dataset", required=True)
    tr.add_argument("--steps", type=int, required=True)
    tr.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    tr.add_argument("--hparams", default="{}")
    tr.add_argument("--variant", default=None)
    bud = sub.add_parser("budget", help="One-time daily budget override that expires by itself")
    bud.add_argument("--daily", type=float, required=True, help="Rolling-24h budget in $ while the override lasts")
    bud.add_argument("--hours", type=float, default=24.0)
    e = sub.add_parser("evolve", help="Program database + evaluation cascade (autolab.evaluate)")
    esub = e.add_subparsers(dest="evolve_cmd", required=True)
    ini = esub.add_parser("init")
    ini.add_argument("--name", required=True, help="Session name, e.g. s2-82m")
    ini.add_argument("--from-batch", required=True,
                     help="Experiment batch whose 'base' variant runs define p0, its budgets and noise")
    act = esub.add_parser("activate")
    act.add_argument("name")
    esub.add_parser("advance")
    esub.add_parser("status")
    gen = esub.add_parser("generate", help="New children from Claude (falls back to mutation)")
    gen.add_argument("-n", type=int, default=1)
    gen.add_argument("--model", default=None, help="Override the [llm] model mix, e.g. sonnet")
    pr = esub.add_parser("propose")
    pr.add_argument("--diff", type=str, default=None, help="File with SEARCH/REPLACE blocks")
    pr.add_argument("--parent", default=None, help="Parent program id (default: the incumbent)")
    pr.add_argument("--hparams", default="{}", help="JSON hyperparameter patch")
    pr.add_argument("--rationale", default="")
    d = sub.add_parser("data", help="Grow the training set (autolab.datasets)")
    dsub = d.add_subparsers(dest="data_cmd", required=True)
    b = dsub.add_parser("build")
    b.add_argument("--num-examples", type=int, required=True)
    s = dsub.add_parser("slice")
    s.add_argument("source")
    s.add_argument("--docs", type=int, required=True)
    ab = dsub.add_parser("ablate", help="Train a program on a bigger dataset (3 seeds) and judge whether it helped")
    ab.add_argument("dataset")
    ab.add_argument("--program", default=None, help="Default: the incumbent")
    ch = dsub.add_parser("check", help="Build a tiny set in scratch and check it is a prefix of the reference")
    ch.add_argument("--num-examples", type=int, default=200)
    r = sub.add_parser("report", help="Rebuild report.json + diagnosis.json (needs runs/tb/ locally)")
    r.add_argument("run_dirs", nargs="+")
    args = p.parse_args(argv)
    if args.cmd == "daemon":
        from autolab.daemon import run

        run(args.interval, args.once)
    elif args.cmd == "research":
        import json
        from pathlib import Path

        from autolab import research

        if args.research_cmd == "run":
            try:
                out = research.run(args.trigger)
            except Exception as exc:  # noqa: BLE001 - the controller reads the result file
                out = {"error": f"{type(exc).__name__}: {exc}", "added": []}
            if args.result:
                Path(args.result).write_text(json.dumps(out, indent=2))
            print(json.dumps(out, indent=2))
        else:
            cards = research.load_cards()
            stats = research.card_stats(cards)
            for c in cards:
                s = stats[c["id"]]
                print(f"{c['id']:4} {c['category']:14} used {s['used']} {s['outcomes']}  {c['name']}")
    elif args.cmd == "transfer":
        import json

        from autolab import evaluate as ev

        exp = ev.start_transfer(args.label, args.program, args.dataset, args.steps, args.seeds,
                                json.loads(args.hparams), args.variant)
        print("\n".join(f"{j['run_id']}: {j['tokens']:,} tokens, cap {j['wall_clock_s']}s" for j in exp["jobs"][-len(args.seeds):]))
    elif args.cmd == "budget":
        from datetime import timedelta

        from autolab import controller as c

        ctl = c.load_control()
        until = c.now() + timedelta(hours=args.hours)
        ctl["daily_usd_override"] = {"usd": args.daily, "until": c.iso(until), "set": c.iso(c.now())}
        c.save_control(ctl)
        c.note("budget_override", daily_usd=args.daily, until=c.iso(until))
        print(f"daily budget ${args.daily} until {c.iso(until)} (then back to [controller] daily_usd)")
    elif args.cmd == "session-report":
        from autolab.session_report import build

        print(build())
    elif args.cmd in ("start", "resume", "stop", "status"):
        control_main(args)
    elif args.cmd == "data":
        data_main(args)
    elif args.cmd == "evolve":
        evolve_main(args)
    elif args.cmd == "report":
        import json
        from pathlib import Path

        from autolab.diagnose import History, diagnose
        from autolab.report import build_report, write_report

        for d in map(Path, args.run_dirs):
            report = build_report(d)
            write_report(report, d)
            diag = diagnose(report, History())
            (d / "diagnosis.json").write_text(json.dumps(diag.to_dict(), indent=2))
            print(f"{d.name}: {diag.summary()}" + (f"  [{'; '.join(diag.notes)}]" if diag.notes else ""))
    elif args.cmd == "dashboard":
        import uvicorn

        from autolab.dashboard import app

        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


MARKERS_COMMIT = "7a7e0e7"  # the EVOLVE-BLOCK refactor; behavior-identical to the prep runs' code


def control_main(args) -> None:
    import json

    from autolab import controller as c
    from autolab import evaluate as ev

    ctl = c.load_control()
    if args.cmd == "start":
        ctl.update(enabled=True, started=c.iso(c.now()))
        c.note("started")
    elif args.cmd == "resume":
        ctl.update(enabled=True, paused_until=None, pause_reason=None)
        c.note("resumed")
    elif args.cmd == "stop":
        ctl["enabled"] = False
        c.note("stopped", cancel_running=args.cancel_running)
        if args.cancel_running:
            import modal

            from autolab import modal_backend as mb

            for rid, call in mb.load_calls().items():
                if call["state"] == "pending":
                    modal.FunctionCall.from_id(call["call_id"]).cancel()
                    print(f"cancelled {rid}")
    if args.cmd != "status":
        c.save_control(ctl)
    session = ev.load_session()
    print(json.dumps({"controller": {k: v for k, v in ctl.items() if k != "recorded"},
                      "session": session and {"name": session.get("name"), "incumbent": session["incumbent"],
                                              "dataset": session.get("dataset_id")}}, indent=2, default=str))


def data_main(args) -> None:
    import json
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from autolab import datasets
    from autolab.config import load_config

    if args.data_cmd == "build":
        print(json.dumps(datasets.build(args.num_examples), indent=2))
    elif args.data_cmd == "slice":
        print(json.dumps(datasets.slice_prefix(args.source, args.docs), indent=2))
    elif args.data_cmd == "ablate":
        from autolab import evaluate as ev

        session = ev.load_session()
        print(json.dumps(ev.start_data_check(args.program or session["incumbent"], args.dataset), indent=2))
    elif args.data_cmd == "check":
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "-m", "mini_llm.prepare_dataset", "--num-examples", str(args.num_examples),
                            "--val-examples", "0", "--seed", str(datasets.data_cfg()["seed"]), "--out-dir", tmp],
                           check=True)
            tiny = datasets.doc_hashes(Path(tmp) / "train.pt")
        ref = datasets.doc_hashes(cfg.datasets_dir / datasets.data_cfg()["reference"] / "train.pt")
        prefix = ref[: len(tiny)] == tiny
        inside = sum(h in set(ref) for h in tiny)
        print(json.dumps({"tiny_docs": len(tiny), "in_reference": inside, "is_prefix_of_reference": prefix}))


def evolve_main(args) -> None:
    import json
    import subprocess
    from pathlib import Path

    from autolab import evaluate as ev
    from autolab.program import parse_diff

    if args.evolve_cmd == "init":
        commit = subprocess.run(["git", "rev-parse", MARKERS_COMMIT], capture_output=True, text=True,
                                check=True).stdout.strip()
        batch = json.loads((ev.REPO_ROOT / "autolab" / "experiments" / f"{args.from_batch}.json").read_text())
        base = [j for j in batch["jobs"] if j.get("variant") == "base"]
        runs = {k: [j["run_id"] for j in base if j["budget"] == k] for k in ("screen", "full")}
        budgets = {"screen_tokens": next(j["tokens"] for j in base if j["budget"] == "screen"),
                   "full_tokens": next(j["tokens"] for j in base if j["budget"] == "full"),
                   "eval": batch["eval"]}
        hparams = {**{k: batch["model"][k] for k in ev.MODEL_KEYS}, **{k: batch["optim"][k] for k in ev.OPTIM_KEYS}}
        paths = ev.Paths(ev.STATE_ROOT / args.name)
        s = ev.init_session(commit, hparams, runs, paths=paths, budgets=budgets, name=args.name)
        ev.set_active_session(args.name)
        print(json.dumps(s, indent=2))
    elif args.evolve_cmd == "activate":
        if not (ev.STATE_ROOT / args.name / "session.json").exists():
            raise SystemExit(f"no session {args.name}")
        ev.set_active_session(args.name)
    elif args.evolve_cmd == "propose":
        session = ev.load_session()
        diffs = parse_diff(Path(args.diff).read_text()) if args.diff else []
        p = ev.propose(args.parent or session["incumbent"], diffs, json.loads(args.hparams), args.rationale)
        print(f"{p.id}: {p.status} at {p.stage}" + (f" ({p.reason})" if p.reason else ""))
    elif args.evolve_cmd == "generate":
        from autolab.generate import generate_one

        for _ in range(args.n):
            generate_one(model=args.model)
    elif args.evolve_cmd == "advance":
        ev.advance_all()
    elif args.evolve_cmd == "status":
        session = ev.load_session()
        if session is None:
            raise SystemExit("no evolve session")
        print(f"incumbent {session['incumbent']}  noise σ screen {session['noise']['screen']['std']:.4f} "
              f"full {session['noise']['full']['std']:.4f}  wall caps {session['wall_caps']}")
        for pid, p in ev.programs().items():
            sc = p.scores
            print(f"{pid:5} {p.parent_id or '-':5} {p.stage:8} {p.status:10} "
                  f"screen {sc.get('screen_loss') or float('nan'):.4f} full {sc.get('full_mean') or float('nan'):.4f} "
                  f"n={sc.get('n_seeds', 0)}  {p.reason[:90]}")


if __name__ == "__main__":
    main()
