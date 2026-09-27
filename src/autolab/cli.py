"""`autolab` command line.

    autolab daemon [--interval 60] [--once]   collector/orchestrator loop (launchd runs this)
    autolab dashboard [--port 8766]           the web dashboard (launchd runs this)
    autolab modal <deploy|upload-data|submit|submit-batch|collect|status> ...
    autolab report <run_dir>...               rebuild report.json + diagnosis.json from local TB events
    autolab evolve init                       evolve session + initial program p0 (from the M3 prep runs)
    autolab evolve propose --diff FILE [--parent P] [--hparams JSON] [--rationale TEXT]
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
    e = sub.add_parser("evolve", help="Program database + evaluation cascade (autolab.evaluate)")
    esub = e.add_subparsers(dest="evolve_cmd", required=True)
    esub.add_parser("init")
    esub.add_parser("advance")
    esub.add_parser("status")
    pr = esub.add_parser("propose")
    pr.add_argument("--diff", type=str, default=None, help="File with SEARCH/REPLACE blocks")
    pr.add_argument("--parent", default=None, help="Parent program id (default: the incumbent)")
    pr.add_argument("--hparams", default="{}", help="JSON hyperparameter patch")
    pr.add_argument("--rationale", default="")
    r = sub.add_parser("report", help="Rebuild report.json + diagnosis.json (needs runs/tb/ locally)")
    r.add_argument("run_dirs", nargs="+")
    args = p.parse_args(argv)
    if args.cmd == "daemon":
        from autolab.daemon import run

        run(args.interval, args.once)
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


INITIAL_HPARAMS = {"n_embd": 256, "n_head": 4, "n_layer": 4, "dropout": 0.0, "batch_size": 64, "lr": 1.2e-3,
                   "min_lr": 2e-6, "warmup_steps": 100, "weight_decay": 0.0}
MARKERS_COMMIT = "7a7e0e7"  # the EVOLVE-BLOCK refactor; behavior-identical to the M3 prep runs' code


def evolve_main(args) -> None:
    import json
    import subprocess
    from pathlib import Path

    from autolab import evaluate as ev
    from autolab.program import parse_diff

    if args.evolve_cmd == "init":
        commit = subprocess.run(["git", "rev-parse", MARKERS_COMMIT], capture_output=True, text=True,
                                check=True).stdout.strip()
        s = ev.init_session(commit, INITIAL_HPARAMS, {
            "screen": [f"m3p-base-screen-s{i}" for i in (1, 2, 3)],
            "full": [f"m3p-base-full-s{i}" for i in (1, 2, 3)]})
        print(json.dumps(s, indent=2))
    elif args.evolve_cmd == "propose":
        session = ev.load_session()
        diffs = parse_diff(Path(args.diff).read_text()) if args.diff else []
        p = ev.propose(args.parent or session["incumbent"], diffs, json.loads(args.hparams), args.rationale)
        print(f"{p.id}: {p.status} at {p.stage}" + (f" ({p.reason})" if p.reason else ""))
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
