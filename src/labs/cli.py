"""uv run labs ...: autolab's loop on other problems (src/labs/).

    labs daemon                     every lab, one process (what com.aadil.autolab-labs runs)
    labs pause | unpause            all labs
    labs <lab> cycle | status       one lab: infer, decode
"""

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    from labs import core, daemon

    names = [lab.name for lab in daemon.all_labs()]
    p = argparse.ArgumentParser(prog="labs")
    p.add_argument("what", choices=["daemon", "pause", "unpause", *names])
    p.add_argument("cmd", nargs="?", choices=["cycle", "status"], default="status")
    a = p.parse_args(argv)
    if a.what == "daemon":
        return daemon.run()
    if a.what in ("pause", "unpause"):
        core.set_hold(a.what == "pause", "cli")
        print(json.dumps(core.control(), indent=2))
        return
    lab = next(x for x in daemon.all_labs() if x.name == a.what)
    if a.cmd == "cycle":
        print(json.dumps(lab.cycle(), indent=2))
    else:
        st = lab.load_lab()
        print(json.dumps({"incumbent": st.get("incumbent"), "blocked": lab.blocked(), "spend_24h": round(core.spend_24h(), 2),
                          "programs": {k: {"status": v.status, "stage": v.stage, "fitness": lab.fitness(v),
                                           "reason": v.reason[:160]} for k, v in lab.programs().items()}}, indent=2))


if __name__ == "__main__":
    main()
