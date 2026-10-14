"""The protected measuring side of labs.pack. Runs in a subprocess (the lab's timeout bounds it).

    python -m labs.pack.harness run --program P --n 26 --seed S --deadline D --out o.json

Runs Packer().run(n, random.Random(seed), deadline) once, then checks the packing independently of the program:
n circles of finite, positive radius, each inside the unit square, no two overlapping, all within TOL. Score = sum of
the radii. The circles are returned too: the packing, not the program's claim, is the result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

TOL = 1e-9


def check(circles, n: int) -> tuple[float | None, str | None]:
    try:
        cs = [(float(x), float(y), float(r)) for x, y, r in circles]
    except (TypeError, ValueError) as exc:
        return None, f"not a list of (x, y, r): {exc}"
    if len(cs) != n:
        return None, f"{len(cs)} circles, expected {n}"
    for i, (x, y, r) in enumerate(cs):
        if not all(math.isfinite(v) for v in (x, y, r)) or r <= 0:
            return None, f"circle {i} is not finite with r > 0: {(x, y, r)}"
        if x - r < -TOL or x + r > 1 + TOL or y - r < -TOL or y + r > 1 + TOL:
            return None, f"circle {i} leaves the unit square: {(x, y, r)}"
    for i in range(n):
        for j in range(i + 1, n):
            (xi, yi, ri), (xj, yj, rj) = cs[i], cs[j]
            if math.hypot(xi - xj, yi - yj) < ri + rj - TOL:
                return None, f"circles {i} and {j} overlap by {ri + rj - math.hypot(xi - xj, yi - yj):.3g}"
    return sum(r for _, _, r in cs), None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run"])
    p.add_argument("--program", required=True)
    p.add_argument("--n", type=int, default=26)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deadline", type=float, default=240)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    t = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("packer", a.program)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        circles = mod.Packer().run(a.n, random.Random(a.seed), a.deadline)
        score, err = check(circles, a.n)
        res = {"ok": err is None, "score": score, "error": err, "wall_s": round(time.perf_counter() - t, 2),
               "circles": [list(map(float, c)) for c in circles] if err is None else None}
    except Exception as exc:  # noqa: BLE001 - the candidate's failure is a result
        import traceback

        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "trace": traceback.format_exc()[-3000:]}
    Path(a.out).write_text(json.dumps(res))
    sys.exit(0)


if __name__ == "__main__":
    main()
