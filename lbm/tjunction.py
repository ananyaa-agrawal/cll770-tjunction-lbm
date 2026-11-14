"""
T-junction droplet-formation case: set-up, run loop, droplet tracking.

Lattice parameters follow Liu & Zhang (2009) Sec. III.B:
    sigma = 0.005, eta_c = 0.08, lambda = eta_d/eta_c = 1/8, w_c = w_d = 20
    (lattice spacing 5 um, so w = 100 um), domain 300 x 60, rho = 1.
Ca = eta_c u_c / sigma  with u_c the mean continuous-phase inlet velocity.
Q  = Q_d / Q_c.
"""
import json
import time
import numpy as np
from scipy import ndimage
import contourpy

from .solver import (Simulation, build_tjunction, poiseuille_profile, FLUID,
                     WALL)

DEFAULTS = dict(Ca=0.02, Q=0.25, lam=0.125, theta=180.0, sigma=0.005,
                eta_c=0.08, xi=2.0, gamma=0.5, w=20, L=300, Ld=40, xj=50,
                width_um=100.0, n_drops=3, max_steps=None, frames=80,
                tag=None)


def droplet_polygons(phi2d, xs, ys, min_area=4.0):
    """Closed phi = 0 contours -> list of droplets (area, centroid, extents)."""
    gen = contourpy.contour_generator(xs, ys, phi2d.T, line_type="Separate")
    out = []
    for line in gen.lines(0.0):
        if len(line) < 4:
            continue
        if not np.allclose(line[0], line[-1]):
            continue  # open line: thread attached to the inlet
        x, y = line[:, 0], line[:, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        A = 0.5 * cross.sum()
        if abs(A) < min_area:
            continue
        cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * A)
        cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * A)
        out.append(dict(area=abs(A), cx=cx, cy=cy, xmin=x.min(), xmax=x.max(),
                        ymin=y.min(), ymax=y.max()))
    out.sort(key=lambda d: d["cx"])
    return out


class TJunctionCase:
    def __init__(self, **kw):
        P = dict(DEFAULTS)
        P.update(kw)
        self.P = P
        w, L, Ld, xj = P["w"], P["L"], P["Ld"], P["xj"]
        self.geo = geo = build_tjunction(w=w, wd=w, L=L, Ld=Ld, xj=xj)
        self.uc = P["Ca"] * P["sigma"] / P["eta_c"]
        self.ud = P["Q"] * self.uc
        sim = Simulation(geo, sigma=P["sigma"], xi=P["xi"], eta_c=P["eta_c"],
                         eta_d=P["lam"] * P["eta_c"], gamma=P["gamma"],
                         theta=P["theta"])
        self.sim = sim
        nc = geo.n_cin
        geo.inl_u[:nc] = poiseuille_profile(w, self.uc)
        geo.inl_v[nc:] = -poiseuille_profile(w, self.ud)
        # initial condition: main channel continuous, lateral channel dispersed
        X, Y = geo.X - 1, geo.Y - 1  # fluid coordinates
        y0 = w + 1.0
        phi0 = np.where(Y >= w, np.tanh((Y - y0) / P["xi"]), -1.0)
        sim.set_phi(phi0)
        # fully developed velocity in the main channel to start
        u0 = np.zeros(geo.N)
        inmain = (geo.ntype == FLUID) & (Y < w)
        prof = poiseuille_profile(w, self.uc)
        u0[inmain] = prof[np.clip(Y[inmain], 0, w - 1)]
        v0 = np.zeros(geo.N)
        inlat = (geo.ntype == FLUID) & (Y >= w)
        profd = poiseuille_profile(w, self.ud)
        v0[inlat] = -profd[np.clip(X[inlat] - xj, 0, w - 1)]
        sim.init_equilibrium(u0, v0)
        # analysis helpers
        self.w, self.L, self.Ld, self.xj = w, L, Ld, xj
        self.x_exit = xj + w - 0.5          # downstream corner of junction
        self.xs = np.arange(geo.nx) - 1.0   # fluid coordinates of array index
        self.ys = np.arange(geo.ny) - 1.0
        self.dx_um = P["width_um"] / w
        # time scales
        A_est = 0.9 * w * w * (1 + 1.8 * P["Q"])
        self.T_est = A_est / (self.ud * w)
        self.check_every = int(max(50, round(0.25 / self.uc)))
        if P["max_steps"] is None:
            # break-up periods at low Ca run up to ~2x T_est (dispersed phase
            # partly dissolves into the continuous phase, see README)
            P["max_steps"] = int((P["n_drops"] + 1.5) * 2.2 * self.T_est + L / self.uc)
        # probes (fluid coords): upstream continuous, lateral dispersed
        self.i_pc = geo.idx(20 + 1, w // 2 + 1)
        self.i_pd = geo.idx(xj + w // 2 + 1, w + Ld // 2 + 1)

    # ------------------------------------------------------------------
    def phi2d(self):
        return self.sim.field(self.sim.phi)

    def thread_metrics(self, phi2d):
        """Gap between emerging thread and the opposite wall; neck width."""
        w, xj = self.w, self.xj
        fl = phi2d[1:self.L + 1, 1:w + self.Ld + 1]
        mask = fl > 0
        lab, _ = ndimage.label(mask)
        top = lab[xj:xj + w, w + self.Ld - 1]
        ids = set(top[top > 0].tolist())
        if not ids:
            return np.nan, np.nan, np.nan
        th = np.isin(lab, list(ids))
        # gap: lowest point of the thread in the main channel
        cols = th[:, :w]
        if cols.any():
            ylow = np.where(cols.any(axis=0))[0].min()
            gap = ylow + 0.5     # wall surface at y = -0.5
        else:
            gap = w
        # neck width: narrowest horizontal extent near the junction mouth
        widths = []
        for yy in range(max(0, w - 6), w + 3):
            row = th[max(0, xj - 3):xj + w + 3, yy]
            widths.append(row.sum())
        neck = float(min(widths)) if widths else np.nan
        tip = np.where(th.any(axis=1))[0].max() + 0.5 if th.any() else np.nan
        return gap, neck, tip

    # ------------------------------------------------------------------
    def run(self, verbose=False):
        P, sim = self.P, self.sim
        t_start = time.time()
        ts = dict(t=[], p_c=[], p_d=[], gap=[], neck=[], tip=[], ndrop=[], phi_out=[])
        drops = []          # finished/active droplet records
        active = []         # indices into drops, ordered upstream->downstream
        frames, frame_t = [], []
        frame_every = max(self.check_every, int(self.T_est / 14))
        n_measured = 0
        status = "ok"
        next_frame = 0
        while sim.t < P["max_steps"]:
            sim.run(self.check_every)
            if not np.isfinite(sim.phi[self.i_pc]):
                status = "diverged"
                break
            ph = self.phi2d()
            gap, neck, tip = self.thread_metrics(ph)
            polys = droplet_polygons(ph, self.xs, self.ys)
            polys = [d for d in polys if d["xmax"] < self.L - 2]
            # --- match droplets: a new one appears upstream of all others,
            #     droplets leave through the outlet at the downstream end
            prev = sorted(active, key=lambda j: drops[j]["cx"])
            new_active = []
            rest = polys
            if polys and (not prev or polys[0]["cx"] < drops[prev[0]]["cx"] - 0.5):
                d = polys[0]
                drops.append(dict(id=len(drops), t_pinch=sim.t,
                                  x_rear_pinch=d["xmin"] - self.x_exit,
                                  x_front_pinch=d["xmax"] - self.x_exit,
                                  y_rear_pinch=d["ymin"], measured=False,
                                  cx=d["cx"]))
                new_active.append(len(drops) - 1)
                rest = polys[1:]
            new_active += prev[:len(rest)]
            for d, j in zip(polys, new_active):
                drops[j]["cx"] = d["cx"]
                if not drops[j]["measured"] and d["xmin"] > self.x_exit + 0.5 * self.w:
                    drops[j].update(measured=True, t_meas=sim.t, area=d["area"],
                                    length=d["xmax"] - d["xmin"],
                                    height=d["ymax"] - d["ymin"], cy=d["cy"])
                    n_measured += 1
            active = new_active
            ts["t"].append(sim.t)
            ts["p_c"].append(sim.p[self.i_pc])
            ts["p_d"].append(sim.p[self.i_pd])
            ts["gap"].append(gap)
            ts["neck"].append(neck)
            ts["tip"].append(tip)
            ts["ndrop"].append(len(drops))
            ts["phi_out"].append(float(np.mean(sim.phi[self.geo.out_nb])))
            if sim.t >= next_frame:
                frames.append(ph[1:self.L + 1, 1:self.w + self.Ld + 1].astype(np.float16))
                frame_t.append(sim.t)
                next_frame = sim.t + frame_every
            if verbose and len(ts["t"]) % 200 == 0:
                print(sim.t, len(drops), n_measured, f"{time.time() - t_start:.0f}s", flush=True)
            # jetting: thread reaches the end of the channel
            if np.isfinite(tip) and tip > self.L - 15:
                status = "jet"
                break
            if n_measured >= P["n_drops"]:
                break
        else:
            status = "max_steps" if n_measured == 0 else "ok_partial"
        self.ts = {k: np.array(v) for k, v in ts.items()}
        self.drops = drops
        self.frames = np.array(frames) if frames else np.zeros((0,))
        self.frame_t = np.array(frame_t)
        self.status = status
        self.wall_time = time.time() - t_start
        return self.summary()

    # ------------------------------------------------------------------
    def summary(self):
        P = self.P
        meas = [d for d in self.drops if d.get("measured")]
        use = meas[1:] if len(meas) >= 2 else meas
        dx = self.dx_um
        res = dict(params=P, status=self.status, steps=int(self.sim.t),
                   wall_time=self.wall_time, uc=self.uc, ud=self.ud,
                   Re=self.uc * self.w / self.P["eta_c"],
                   n_drops=len(self.drops), n_measured=len(meas))
        if use:
            A = np.array([d["area"] for d in use])
            res.update(
                area_lu=float(A.mean()),
                D_um=float(np.sqrt(4 * A.mean() / np.pi) * dx),
                D_over_w=float(np.sqrt(4 * A.mean() / np.pi) / self.w),
                L_over_w=float(np.mean([d["length"] for d in use]) / self.w),
                H_over_w=float(np.mean([d["height"] for d in use]) / self.w),
                area_spread=float(A.std() / A.mean()) if len(A) > 1 else 0.0,
                x_rear_pinch_over_w=float(np.mean([d["x_rear_pinch"] for d in use]) / self.w),
            )
            tp = [d["t_pinch"] for d in self.drops]
            if len(tp) >= 2:
                res["period_steps"] = float(np.diff(tp[1:] if len(tp) > 2 else tp).mean())
                # dimensionless period  T u_c / w
                res["period_nd"] = res["period_steps"] * self.uc / self.w
                # fraction of the injected dispersed phase that ends up in droplets
                res["mass_ratio"] = res["area_lu"] / (self.ud * self.w * res["period_steps"])
            gaps = self.ts["gap"]
            # minimum gap reached in the last formation cycle(s)
            t = self.ts["t"]
            t0 = self.drops[0]["t_pinch"] if self.drops else 0
            sel = t > t0
            if sel.any():
                res["min_gap_over_w"] = float(np.nanmin(gaps[sel]) / self.w)
        return res

    def save(self, path_prefix, data_prefix=None):
        """Summary -> <path_prefix>.json; time series and phi snapshots ->
        <data_prefix>.npz (defaults to path_prefix)."""
        res = self.summary()
        with open(path_prefix + ".json", "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        data_prefix = data_prefix or path_prefix
        np.savez_compressed(data_prefix + ".npz", frames=self.frames,
                            frame_t=self.frame_t, drops=json.dumps(self.drops, default=float),
                            **{"ts_" + k: v for k, v in self.ts.items()})
        return res
