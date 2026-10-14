"""p0 of labs.pack: a plain grid of 26 centers, each radius as large as the walls and its neighbours allow.

The evolved program is this whole file. The harness (protected) calls, once, in a fresh process:

    circles = Packer().run(n, rng, deadline_s)   # n = 26; rng: a seeded random.Random; deadline_s: seconds left
                                                 # -> list of n (x, y, r) floats

and checks the result itself: every circle inside the unit square, no two overlapping (tolerance 1e-9). The score is
the sum of the radii. numpy, scipy, math and random are available; no file, OS or network access.
"""

import math


class Packer:
    def run(self, n: int, rng, deadline_s: float) -> list[tuple[float, float, float]]:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        centers = [((c + 0.5) / cols, (r + 0.5) / rows) for r in range(rows) for c in range(cols)][:n]
        radii = [min(x, y, 1 - x, 1 - y) for x, y in centers]
        for _ in range(200):  # shrink overlapping pairs in proportion until none overlap
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    d = math.dist(centers[i], centers[j])
                    if radii[i] + radii[j] > d:
                        s = d / (radii[i] + radii[j])
                        radii[i] *= s
                        radii[j] *= s
                        changed = True
            if not changed:
                break
        return [(x, y, r) for (x, y), r in zip(centers, radii)]
