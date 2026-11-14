"""
Parameter sweeps (Liu & Zhang 2009, Sec. III). Every case is written to
results/runs/<name>.json (+ data/runs/<name>.npz with time series and phi snapshots), and
finished cases are skipped on re-run, so the script can be restarted.

usage: python scripts/run_sweeps.py [n_workers] [group ...]
groups: A (Ca x Q), B (extra Q at Ca = 0.005), C (viscosity ratio),
        D (contact angle), E (mobility / grid checks)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "runs")
DATA = os.path.join(ROOT, "data", "runs")

CA_MAIN = [0.005, 0.008, 0.012, 0.016, 0.02, 0.024, 0.032, 0.04, 0.048, 0.056]


def name(c):
    s = f"Ca{c['Ca']:.3f}_Q{c['Q']:.3f}_lam{c['lam']:.3f}_th{c['theta']:.0f}"
    if c.get("gamma", 0.5) != 0.5:
        s += f"_G{c['gamma']:.2f}"
    if c.get("w", 20) != 20:
        s += f"_w{c['w']}"
    return s


def case(Ca, Q, lam=0.125, theta=180.0, **kw):
    c = dict(Ca=Ca, Q=Q, lam=lam, theta=theta)
    c.update(kw)
    return c


def groups():
    G = {}
    G["A"] = [case(Ca, Q) for Q in (0.125, 0.25, 0.5) for Ca in CA_MAIN]
    G["B"] = [case(0.005, Q) for Q in (0.375, 0.75, 1.0)]
    G["C"] = [case(Ca, 0.25, lam=1.0) for Ca in CA_MAIN if Ca != 0.048]
    G["D"] = ([case(Ca, 0.25, theta=th) for th in (110, 130, 150)
               for Ca in (0.008, 0.016, 0.032, 0.056)] +
              [case(Ca, 0.25, lam=1.0, theta=th) for th in (110, 150)
               for Ca in (0.008, 0.032, 0.056)])
    G["E"] = [case(Ca, 0.25, w=40, L=600, Ld=80, xj=100, xi=2.0)
              for Ca in (0.012, 0.032)]
    return G


def cost(c):
    """rough relative cost (steps x nodes) for longest-first scheduling"""
    A = 1 + 1.8 * c["Q"]
    return A / (c["Q"] * c["Ca"]) * (c.get("w", 20) / 20) ** 4


def run_one(c):
    from lbm.tjunction import TJunctionCase
    nm = name(c)
    path = os.path.join(OUT, nm)
    if os.path.exists(path + ".json"):
        return nm, "skipped"
    t0 = time.time()
    c = dict(c)
    # successive droplets agree to <0.1% at low Ca, so two suffice there
    c.setdefault("n_drops", 2 if c["Ca"] < 0.015 else 3)
    tc = TJunctionCase(**c)
    tc.run()
    res = tc.save(path, os.path.join(DATA, nm))
    return nm, f"{res['status']} D={res.get('D_um', float('nan')):.1f}um " \
               f"steps={res['steps']} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    sel = sys.argv[2:] or ["A", "B", "C", "D"]
    G = groups()
    todo = []
    for g in sel:
        todo += sorted(G[g], key=cost, reverse=True)
    todo = [c for c in todo if not os.path.exists(os.path.join(OUT, name(c) + ".json"))]
    print(f"{len(todo)} cases to run", flush=True)
    with Pool(nw) as pool:
        for nm, msg in pool.imap_unordered(run_one, todo):
            print(time.strftime("%H:%M:%S"), nm, msg, flush=True)
