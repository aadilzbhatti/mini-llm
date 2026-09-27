"""`autolab` command line.

    autolab daemon [--interval 60] [--once]   collector/orchestrator loop (launchd runs this)
    autolab dashboard [--port 8766]           the web dashboard (launchd runs this)
    autolab modal <deploy|upload-data|submit|submit-batch|collect|status> ...
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
    args = p.parse_args(argv)
    if args.cmd == "daemon":
        from autolab.daemon import run

        run(args.interval, args.once)
    elif args.cmd == "dashboard":
        import uvicorn

        from autolab.dashboard import app

        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
