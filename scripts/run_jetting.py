"""
Longer runs of the jetting cases (Q = 1/2, Ca >= 0.032): in the main sweep the
break-up point was still moving downstream after three droplets. Here ten
droplets are formed to see whether the jet settles or reaches the outlet.
Writes results/jetting/<name>.json and data/jetting/<name>.npz
"""
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "jetting")
DATA = os.path.join(ROOT, "data", "jetting")


def run(Ca):
    from lbm.tjunction import TJunctionCase
    c = TJunctionCase(Ca=Ca, Q=0.5, n_drops=10)
    c.run()
    nm = f"Ca{Ca:.3f}_Q0.500_long"
    res = c.save(os.path.join(OUT, nm), os.path.join(DATA, nm))
    xs = [round(d["x_rear_pinch"] / 20, 2) for d in c.drops]
    return Ca, res["status"], xs


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    with Pool(2) as p:
        for Ca, st, xs in p.imap_unordered(run, [0.032, 0.056]):
            print(Ca, st, "break-up x/w per droplet:", xs, flush=True)
