"""`autolab` command line.

    autolab daemon [--interval 60] [--once]   collector/orchestrator loop (launchd runs this)
    autolab dashboard [--port 8766]           the web dashboard (launchd runs this)
    autolab modal <deploy|upload-data|submit|submit-batch|collect|status> ...
    autolab report <run_dir>...               rebuild report.json + diagnosis.json from local TB events
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
    r = sub.add_parser("report", help="Rebuild report.json + diagnosis.json (needs runs/tb/ locally)")
    r.add_argument("run_dirs", nargs="+")
    args = p.parse_args(argv)
    if args.cmd == "daemon":
        from autolab.daemon import run

        run(args.interval, args.once)
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


if __name__ == "__main__":
    main()
