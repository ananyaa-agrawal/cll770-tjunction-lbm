"""
Verification tests (all lattice units, sigma = 0.005, xi = 2, Gamma = 0.5):
  1. Laplace law: static 2D droplets, dp = sigma / R
  2. Poiseuille flow: single-phase channel with the T-junction inlet/outlet BCs
  3. Static contact angle: droplet on a wall with the wetting condition
Writes results/validation.json and results/validation.npz
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from lbm.solver import (Geometry, Simulation, poiseuille_profile, FLUID, WALL,
                        INLET, OUTLET)
import contourpy

SIGMA, XI, GAMMA = 0.005, 2.0, 0.5


def laplace_test(n=128, radii=(12, 16, 20, 24, 28, 32), steps=20000):
    out = []
    for R in radii:
        geo = Geometry(np.zeros((n, n), np.int64), True, True)
        sim = Simulation(geo, sigma=SIGMA, xi=XI, eta_c=0.08, eta_d=0.01,
                         gamma=GAMMA)
        r = np.hypot(geo.X - n / 2, geo.Y - n / 2)
        sim.set_phi(np.tanh((R - r) / XI))
        sim.init_equilibrium()
        sim.run(steps)
        P = sim.field(sim.p)
        ph = sim.field(sim.phi)
        dp = P[n // 2, n // 2] - P[0, 0]
        # measured radius from the area of phi > 0 (sub-grid via (1+phi)/2)
        Rm = np.sqrt(((1 + ph) / 2).sum() / np.pi)
        umax = np.hypot(sim.ux, sim.uy).max()
        out.append(dict(R0=R, R=float(Rm), dp=float(dp), umax=float(umax)))
        print("Laplace", out[-1], flush=True)
    invR = np.array([o["R"] for o in out]) ** -1
    dp = np.array([o["dp"] for o in out])
    slope, icpt = np.polyfit(invR, dp, 1)
    return dict(cases=out, sigma_fit=float(slope), intercept=float(icpt),
                sigma_theory=SIGMA,
                rel_err=float(abs(slope - SIGMA) / SIGMA))


def straight_channel(w=20, L=200):
    nx, ny = L + 2, w + 2
    t = np.full((nx, ny), WALL, np.int64)
    t[1:L + 1, 1:w + 1] = FLUID
    t[0, 1:w + 1] = INLET
    t[L + 1, 1:w + 1] = OUTLET
    geo = Geometry(t)
    geo.inl_idx = np.array([geo.idx(0, y + 1) for y in range(w)], np.int64)
    geo.inl_nb = np.array([geo.idx(1, y + 1) for y in range(w)], np.int64)
    geo.inl_u = np.zeros(w)
    geo.inl_v = np.zeros(w)
    geo.inl_phi = -np.ones(w)
    geo.out_idx = np.array([geo.idx(L + 1, y + 1) for y in range(w)], np.int64)
    geo.out_nb = np.array([geo.idx(L, y + 1) for y in range(w)], np.int64)
    geo.n_cin = w
    return geo


def poiseuille_test(w=20, L=200, umean=0.002, eta=0.08, steps=60000,
                    uniform_inlet=False):
    """Parabolic inlet (as in the T-junction runs) -> check the profile is kept
    downstream, the flux is conserved and dp/dx matches 12 eta U / w^2."""
    geo = straight_channel(w, L)
    geo.inl_u[:] = umean if uniform_inlet else poiseuille_profile(w, umean)
    sim = Simulation(geo, sigma=SIGMA, xi=XI, eta_c=eta, eta_d=eta,
                     gamma=GAMMA, theta=180.0)
    sim.set_phi(-np.ones(geo.N))
    sim.init_equilibrium()
    sim.run(steps)
    U = sim.field(sim.ux)[1:L + 1, 1:w + 1]
    P = sim.field(sim.p)[1:L + 1, 1:w + 1]
    y = np.arange(w) + 0.5
    ua = 6 * umean * y * (w - y) / w ** 2
    xm = L // 2
    err = np.sqrt(((U[xm] - ua) ** 2).sum() / (ua ** 2).sum())
    dpdx = np.polyfit(np.arange(60, 140), P[60:140, w // 2], 1)[0]
    dpdx_th = -12 * eta * umean / w ** 2
    return dict(umean=umean, w=w, L2_err=float(err),
                umax_num=float(U[xm].max()), umax_th=float(1.5 * umean),
                dpdx=float(dpdx), dpdx_th=float(dpdx_th),
                dpdx_rel_err=float(abs(dpdx - dpdx_th) / abs(dpdx_th)),
                flux_num=float(U[xm].sum()), flux_th=float(umean * w)), \
        dict(y=y, u_num=U[xm], u_th=ua, u_inlet=U[0], u_x10=U[10], u_x40=U[40])


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (xc, yc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return xc, yc, np.sqrt(c + xc ** 2 + yc ** 2)


def contact_angle_test(thetas=(90, 110, 130, 150), nx=200, h=80, R0=25,
                       steps=80000):
    out = []
    shapes = {}
    for th in thetas:
        t = np.full((nx, h + 2), WALL, np.int64)
        t[:, 1:h + 1] = FLUID
        geo = Geometry(t, periodic_x=True)
        sim = Simulation(geo, sigma=SIGMA, xi=XI, eta_c=0.08, eta_d=0.08,
                         gamma=GAMMA, theta=th)
        X, Y = geo.X, geo.Y - 1  # fluid y; wall surface at y = -0.5
        r = np.hypot(X - nx / 2, Y + 0.5)
        sim.set_phi(np.tanh((R0 - r) / XI))
        sim.init_equilibrium()
        hist = []
        for k in range(steps // 10000):
            sim.run(10000)
            ph = sim.field(sim.phi)
            gen = contourpy.contour_generator(np.arange(nx), np.arange(h + 2) - 1.0,
                                              ph.T, line_type="Separate")
            lines = gen.lines(0.0)
            pts = max(lines, key=len)
            sel = pts[:, 1] > 4.0
            xc, yc, R = fit_circle(pts[sel, 0], pts[sel, 1])
            theta_m = np.degrees(np.arccos(np.clip(-(yc + 0.5) / R, -1, 1)))
            hist.append(theta_m)
        out.append(dict(theta_in=th, theta_meas=float(hist[-1]),
                        theta_hist=[float(v) for v in hist], R=float(R),
                        phi_w=float(sim.phiw)))
        shapes[f"phi_{th}"] = sim.field(sim.phi).astype(np.float32)
        print("Contact angle", out[-1], flush=True)
    return out, shapes


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    res = {}
    res["poiseuille"], prof = poiseuille_test()
    print("Poiseuille", res["poiseuille"], flush=True)
    res["laplace"] = laplace_test()
    print("Laplace fit", {k: v for k, v in res["laplace"].items() if k != "cases"})
    res["contact_angle"], shapes = contact_angle_test()
    with open(os.path.join(ROOT, "results", "validation.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(os.path.join(ROOT, "results", "validation.npz"),
                        **{"pois_" + k: v for k, v in prof.items()}, **shapes)
    print("done")
