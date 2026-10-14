"""uv run labs <lab> <command>: the labs that reuse autolab's loop on other problems.

    labs infer daemon | cycle | status | pause | unpause
"""

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="labs")
    p.add_argument("lab", choices=["infer"])
    p.add_argument("cmd", choices=["daemon", "cycle", "status", "pause", "unpause"])
    a = p.parse_args(argv)
    from labs.infer import lab

    if a.cmd == "daemon":
        lab.run()
    elif a.cmd == "cycle":
        print(json.dumps(lab.cycle(), indent=2))
    elif a.cmd in ("pause", "unpause"):
        lab.set_hold(a.cmd == "pause", "cli")
        lab.note("held" if a.cmd == "pause" else "released", by="cli")
        print(json.dumps(lab.control(), indent=2))
    else:
        progs = lab.programs()
        st = lab.load_lab()
        print(json.dumps({"incumbent": st.get("incumbent"), "history": st.get("history"), "blocked": lab.blocked(),
                          "spend_24h": round(lab.spend_24h(), 2),
                          "programs": {k: {"status": v.status, "stage": v.stage, "x_p0": v.scores.get("x_p0"),
                                           "reason": v.reason[:120]} for k, v in progs.items()}}, indent=2))


if __name__ == "__main__":
    main()
