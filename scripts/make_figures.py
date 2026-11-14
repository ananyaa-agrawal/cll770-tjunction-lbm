"""
Figures + comparison table from results/validation.json and results/runs/*.
usage: python scripts/make_figures.py
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "results", "runs")   # per-case summaries (.json)
DATA = os.path.join(ROOT, "data", "runs")      # per-case time series (.npz)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- style (reference categorical palette, fixed order) -------------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF, WALLC = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c9c8c2"
MK = ["o", "s", "^", "D"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 7, "figure.dpi": 150,
})
PHI_CMAP = LinearSegmentedColormap.from_list("phi", ["#e9eef4", "#2a78d6"])

PAPER = dict(Ca_c=0.018, alpha=1.82, m=-0.78)
JET_X = 2.0     # break-up more than 2 w downstream of the junction -> jetting


def load_runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(f))
        if "params" not in r:
            continue
        p = r["params"]
        r.update(Ca=p["Ca"], Q=p["Q"], lam=p["lam"], theta=p["theta"],
                 gamma=p.get("gamma", 0.5), w=p.get("w", 20),
                 name=os.path.basename(f)[:-5])
        if "period_steps" in r and "area_lu" in r:
            # fraction of injected dispersed phase that ends up in the droplets
            r["mass_ratio"] = r["area_lu"] / (r["ud"] * r["w"] * r["period_steps"])
        r["regime"] = "jetting" if r["status"] == "jet" else "none"
        npz = os.path.join(DATA, os.path.basename(f)[:-5] + ".npz")
        if os.path.exists(npz):
            d = json.loads(str(np.load(npz, allow_pickle=True)["drops"]))
            r["x_rear_seq"] = [x["x_rear_pinch"] / r["w"] for x in d]
        rows.append(r)
    return rows


def sel(rows, **kw):
    out = [r for r in rows if all(abs(r[k] - v) < 1e-9 for k, v in kw.items())
           and "D_um" in r]
    return sorted(out, key=lambda r: r["Ca"])


def frac(q):
    return {0.125: "1/8", 0.25: "1/4", 0.375: "3/8", 0.5: "1/2", 0.75: "3/4",
            1.0: "1"}.get(q, f"{q:g}")


def plain_ticks(axis, ticks):
    axis.set_ticks(ticks)
    axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_validation():
    path = os.path.join(ROOT, "results", "validation.json")
    if not os.path.exists(path):
        return {}
    V = json.load(open(path))
    Z = np.load(os.path.join(ROOT, "results", "validation.npz"))
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.6))
    # (a) Laplace
    a = ax[0, 0]
    L = V["laplace"]
    invR = np.array([c["R"] for c in L["cases"]]) ** -1
    dp = np.array([c["dp"] for c in L["cases"]])
    xx = np.linspace(0, invR.max() * 1.1, 50)
    a.plot(xx, 0.005 * xx * 1e4, color=INK2, lw=1.5, ls="--", label=r"theory $\Delta p=\sigma/R$")
    a.plot(invR, dp * 1e4, MK[0], color=C[0], label="LBM")
    a.set_xlabel("1 / R  (lattice units$^{-1}$)")
    a.set_ylabel(r"$\Delta p\;\times10^{4}$  (lattice units)")
    a.set_title("(a) Laplace test")
    s0 = float((invR * dp).sum() / (invR * invR).sum())   # fit through origin
    dev = np.abs(dp / invR / 0.005 - 1).max()
    V["laplace"]["sigma_fit0"] = s0
    V["laplace"]["max_dev"] = float(dev)
    a.text(0.04, 0.78, f"fitted σ = {s0:.5f}  (imposed 0.00500)\n"
           f"every radius within {100 * dev:.1f}% of σ/R", transform=a.transAxes,
           color=INK2)
    a.legend(loc="lower right")
    a.set_xlim(0, None)
    a.set_ylim(0, None)
    # (b) Poiseuille
    a = ax[0, 1]
    P = V["poiseuille"]
    y, un, ut = Z["pois_y"], Z["pois_u_num"], Z["pois_u_th"]
    w = P["w"]
    yy = np.linspace(0, w, 100)
    a.plot(6 * yy * (w - yy) / w ** 2, yy / w, color=INK2, lw=1.5, ls="--",
           label="analytical")
    a.plot(un / P["umean"], y / w, MK[0], color=C[0], label="LBM (x = L/2)")
    a.set_xlabel(r"$u/\bar u$")
    a.set_ylabel("y / w")
    a.set_title("(b) Poiseuille channel flow")
    a.text(0.97, 0.94, f"L2 error {100 * P['L2_err']:.2f}%\n"
           f"dp/dx error {100 * P['dpdx_rel_err']:.2f}%\n"
           f"flux error {100 * abs(P['flux_num'] / P['flux_th'] - 1):.2f}%",
           transform=a.transAxes, color=INK2, ha="right", va="top")
    a.legend(loc="lower right")
    # (c) contact angle
    a = ax[1, 0]
    CA = V["contact_angle"]
    ti = np.array([c["theta_in"] for c in CA])
    tm = np.array([c["theta_meas"] for c in CA])
    a.plot([80, 160], [80, 160], color=INK2, lw=1.5, ls="--", label="imposed = measured")
    a.plot(ti, tm, MK[0], color=C[0], label="LBM")
    for x_, y_ in zip(ti, tm):
        a.annotate(f"{y_:.1f}°", (x_, y_), xytext=(6, -12), textcoords="offset points",
                   color=INK2, fontsize=9)
    a.set_xlabel("imposed contact angle (deg)")
    a.set_ylabel("measured contact angle (deg)")
    a.set_title("(c) Static contact angle")
    a.legend(loc="upper left")
    # (d) droplet shapes
    a = ax[1, 1]
    for k, c in enumerate(CA):
        ph = Z[f"phi_{c['theta_in']}"]
        nx, ny = ph.shape
        xs = np.arange(nx) - nx / 2
        ys = np.arange(ny - 1) * 1.0          # fluid rows only (y >= 0)
        cs = a.contour(xs, ys, ph[:, 1:].T, levels=[0], colors=[C[k % 4]], linewidths=2)
        a.plot([], [], color=C[k % 4], label=f"θ = {c['theta_in']}°")
    a.axhspan(-3, -0.5, color=WALLC)
    a.text(-43, -2.2, "wall", color=INK2, fontsize=8)
    a.set_xlim(-45, 45)
    a.set_ylim(-3, 45)
    a.set_aspect("equal")
    a.set_xlabel("x (lattice units)")
    a.set_ylabel("y (lattice units)")
    a.set_title("(d) Equilibrium droplet shapes")
    a.legend(loc="upper right", fontsize=9)
    a.grid(False)
    fig.tight_layout()
    save(fig, "fig1_validation.png")
    return V


# ---------------------------------------------------------------------------
def load_npz(name):
    Z = np.load(os.path.join(DATA, name + ".npz"), allow_pickle=True)
    drops = json.loads(str(Z["drops"]))
    return Z, drops


def cycle_frames(name, n=5):
    Z, drops = load_npz(name)
    F, ft = Z["frames"].astype(float), Z["frame_t"]
    tp = [d["t_pinch"] for d in drops]
    if len(tp) >= 3:
        t0, t1 = tp[-2], tp[-1]
    elif len(tp) == 2:
        t0, t1 = tp[0], tp[1]
    else:
        t0, t1 = ft[0], ft[-1]
    T = t1 - t0
    targets = t0 + np.linspace(0.15, 1.02, n) * T
    idx = [int(np.argmin(abs(ft - t))) for t in targets]
    return [(F[i], (ft[i] - t0) / T) for i in idx]


def fig_snapshots(rows):
    cases = [("Ca0.005_Q0.250_lam0.125_th180", "Squeezing\nCa 0.005, Q 1/4"),
             ("Ca0.016_Q0.250_lam0.125_th180", "Near Ca$_c$\nCa 0.016, Q 1/4"),
             ("Ca0.056_Q0.250_lam0.125_th180", "Dripping\nCa 0.056, Q 1/4"),
             ("Ca0.056_Q0.500_lam0.125_th180", "Jetting\nCa 0.056, Q 1/2")]
    cases = [c for c in cases if os.path.exists(os.path.join(DATA, c[0] + ".npz"))]
    if not cases:
        return
    n = 5
    fig, ax = plt.subplots(len(cases), n, figsize=(13, 1.2 * len(cases) + 0.4),
                           gridspec_kw=dict(hspace=0.35, wspace=0.04))
    ax = np.atleast_2d(ax)
    geo_mask = None
    for r, (nm, title) in enumerate(cases):
        frames = cycle_frames(nm, n)
        for k, (ph, tt) in enumerate(frames):
            a = ax[r, k]
            w = 20
            solid = np.zeros_like(ph, bool)
            solid[:, w:] = True
            solid[50:70, w:] = False
            img = np.ma.masked_where(solid, ph)
            a.imshow(np.where(solid, 1, np.nan).T, origin="lower", cmap=LinearSegmentedColormap.from_list("w", [WALLC, WALLC]),
                     extent=(-0.5, ph.shape[0] - 0.5, -0.5, ph.shape[1] - 0.5))
            a.imshow(img.T, origin="lower", cmap=PHI_CMAP, vmin=-1, vmax=1,
                     extent=(-0.5, ph.shape[0] - 0.5, -0.5, ph.shape[1] - 0.5),
                     interpolation="bilinear")
            a.axhspan(-4, -0.5, color=WALLC, lw=0)
            a.set_xlim(35, 205)
            a.set_ylim(-4, 36)
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
            a.set_title(f"t/T = {tt:.2f}", fontsize=9, fontweight="normal", color=INK2)
        ax[r, 0].text(-0.05, 0.45, title, transform=ax[r, 0].transAxes,
                      ha="right", va="center", fontsize=9.5, fontweight="bold")
    fig.suptitle("Droplet formation over one break-up cycle (λ = 1/8, θ = 180°; T = break-up period)",
                 fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.02)
    save(fig, "fig2_snapshots.png")


# ---------------------------------------------------------------------------
def hinge_fit(series, lo=0.007, hi=0.045, n=600):
    """Common break point b for several D(Ca) series, each fitted with two
    power laws joined at b (log D linear in log Ca on either side)."""
    best = None
    for bl in np.linspace(np.log(lo), np.log(hi), n):
        sse, coefs = 0.0, []
        for Ca, D in series:
            x, y = np.log(Ca), np.log(D)
            X = np.c_[np.ones_like(x), np.minimum(x - bl, 0), np.maximum(x - bl, 0)]
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse += ((X @ c - y) ** 2).sum()
            coefs.append(c)
        if best is None or sse < best[0]:
            best = (sse, bl, coefs)
    return float(np.exp(best[1])), best[2]


def critical_ca(series, nboot=300):
    """Ca_c with a bootstrap (drop ~20% of points) 16-84% interval."""
    Cac, coefs = hinge_fit(series)
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(nboot):
        sub = []
        for Ca, D in series:
            keep = rng.random(len(Ca)) > 0.2
            if keep.sum() >= 5:
                sub.append((Ca[keep], D[keep]))
        if sub:
            bs.append(hinge_fit(sub, n=200)[0])
    lo, hi = np.percentile(bs, [16, 84])
    return dict(Ca_c=Cac, lo=float(lo), hi=float(hi), coefs=coefs,
                slopes=[(float(c[1]), float(c[2])) for c in coefs])


def series_D(rows, **kw):
    R = sel(rows, **kw)
    return np.array([r["Ca"] for r in R]), np.array([r["D_um"] for r in R])


def assign_regimes(rows, Cac):
    for r in rows:
        if "D_um" not in r:
            continue
        xs = r.get("x_rear_seq", [0])
        growing = len(xs) >= 2 and xs[-1] > JET_X and xs[-1] > xs[-2]
        if r["status"] == "jet" or growing:
            # break-up point beyond 2 w and still moving downstream
            r["regime"] = "jetting"
        else:
            r["regime"] = "squeezing" if r["Ca"] < Cac else "dripping"


def draw_hinge(a, Ca, coef, Cac, color):
    cc = np.geomspace(Ca.min(), Ca.max(), 80)
    x, b = np.log(cc), np.log(Cac)
    y = coef[0] + coef[1] * np.minimum(x - b, 0) + coef[2] * np.maximum(x - b, 0)
    a.plot(cc, np.exp(y), color=color, lw=1.4, alpha=0.9)


def mark_cac(a, fit, label="this work"):
    a.axvspan(fit["lo"], fit["hi"], color=C[0], alpha=0.10, lw=0)
    a.axvline(fit["Ca_c"], color=C[0], lw=1.2)
    a.axvline(PAPER["Ca_c"], color=INK2, ls="--", lw=1.2)
    ylim = a.get_ylim()
    a.text(fit["Ca_c"] - 0.0008, ylim[0] + 0.03 * (ylim[1] - ylim[0]),
           f"{label}\nCa$_c$ = {fit['Ca_c']:.4f}", color=C[0], fontsize=8.5, ha="right")
    a.text(PAPER["Ca_c"] + 0.0008, ylim[0] + 0.03 * (ylim[1] - ylim[0]),
           "Liu & Zhang\nCa$_c$ = 0.018", color=INK2, fontsize=8.5, ha="left")


def fig_D_vs_Ca(rows, fit):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a = ax[0]
    Qs = (0.125, 0.25, 0.5)
    for k, Q in enumerate(Qs):
        Ca, D = series_D(rows, Q=Q, lam=0.125, theta=180, gamma=0.5, w=20)
        if len(Ca):
            a.plot(Ca, D, MK[k], color=C[k], label=f"Q = {frac(Q)}")
            draw_hinge(a, Ca, fit["coefs"][k], fit["Ca_c"], C[k])
    a.set_xlabel("Ca")
    a.set_ylabel("droplet diameter D (µm)")
    a.set_title("(a) Droplet size vs Ca  (λ = 1/8, θ = 180°)")
    a.legend(loc="upper right", fontsize=9)
    mark_cac(a, fit)
    # (b) log-log view: power-law exponents on either side of Ca_c
    a = ax[1]
    for k, Q in enumerate(Qs):
        Ca, D = series_D(rows, Q=Q, lam=0.125, theta=180, gamma=0.5, w=20)
        if not len(Ca):
            continue
        s1, s2 = fit["slopes"][k]
        a.loglog(Ca, D / 100, MK[k], color=C[k],
                 label=f"Q = {frac(Q)}:  D $\\propto$ Ca$^{{{s1:.2f}}}$ → Ca$^{{{s2:.2f}}}$")
        draw_hinge(a, Ca, fit["coefs"][k] - np.r_[np.log(100), 0, 0], fit["Ca_c"], C[k])
    a.axvline(fit["Ca_c"], color=C[0], lw=1.2)
    a.axvline(PAPER["Ca_c"], color=INK2, ls="--", lw=1.2)
    a.set_xlabel("Ca")
    a.set_ylabel("D / w")
    a.set_title("(b) Same data, log-log: two power laws meeting at Ca$_c$")
    a.legend(loc="lower left", fontsize=8.5)
    a.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    a.set_xticks([0.005, 0.01, 0.02, 0.03, 0.05])
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    save(fig, "fig3_size_vs_Ca.png")
    return fit_eq27(rows)


def eq27(X, Vc, tn, m):
    Ca, Qd = X
    return Vc * Ca ** m + tn * Ca ** m * Qd


def fit_eq27(rows):
    R = [r for r in sel(rows, lam=0.125, theta=180, gamma=0.5, w=20)
         if r["Q"] in (0.125, 0.25, 0.5)]
    if len(R) < 5:
        return None
    Ca = np.array([r["Ca"] for r in R])
    Qd = np.array([r["Q"] * r["uc"] * 20 for r in R])
    A = np.array([r["area_lu"] for r in R])
    p, cov = curve_fit(eq27, np.vstack([Ca, Qd]), A, p0=[10, 1e4, -0.7],
                       sigma=A, maxfev=20000)
    pred = eq27(np.vstack([Ca, Qd]), *p)
    return dict(p=p, err=np.sqrt(np.diag(cov)),
                rms_rel=float(np.sqrt(np.mean((pred / A - 1) ** 2))))


# ---------------------------------------------------------------------------
def fig_garstecki(rows):
    R = [r for r in sel(rows, Ca=0.005, lam=0.125, theta=180, gamma=0.5, w=20)]
    if len(R) < 3:
        return None
    R = sorted(R, key=lambda r: r["Q"])
    Q = np.array([r["Q"] for r in R])
    l = np.array([r["L_over_w"] for r in R])
    alpha = float(np.sum(Q * (l - 1)) / np.sum(Q * Q))       # intercept fixed at 1
    mq = Q <= 0.5                                            # Q range used in the paper
    alpha_p = float(np.sum(Q[mq] * (l[mq] - 1)) / np.sum(Q[mq] ** 2))
    b1, b0 = np.polyfit(Q, l, 1)                             # free intercept
    fig, a = plt.subplots(figsize=(5.6, 4.2))
    qq = np.linspace(0, 1.05, 50)
    a.plot(qq, 1 + PAPER["alpha"] * qq, ls="--", color=INK2, lw=1.5,
           label="Liu & Zhang: l/w = 1 + 1.82 Q")
    a.plot(qq, 1 + alpha * qq, color=C[0], lw=1.5, label=f"this work (Q = 1/8–1): l/w = 1 + {alpha:.2f} Q")
    a.plot(Q, l, MK[0], color=C[0], label="LBM, Ca = 0.005")
    a.set_xlabel(r"flow-rate ratio  Q = Q$_d$/Q$_c$")
    a.set_ylabel("droplet length  l / w")
    a.set_title("Squeezing-regime scaling (Garstecki law)")
    a.set_xlim(0, 1.05)
    a.text(0.97, 0.05, f"fit over the paper's range Q = 1/8–1/2:\nα = {alpha_p:.2f}  (paper 1.82)",
           transform=a.transAxes, ha="right", color=INK2, fontsize=9)
    a.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_garstecki.png")
    return dict(alpha=alpha, alpha_Q_le_half=alpha_p, free_slope=float(b1), free_intercept=float(b0),
                Q=Q.tolist(), l=l.tolist())


# ---------------------------------------------------------------------------
def fig_viscosity(rows, fit):
    """lambda = 1/8 uses the joint three-Q fit (Q = 1/4 branch); lambda = 1 is
    fitted on its own (single series)."""
    fig, a = plt.subplots(figsize=(6.4, 4.4))
    out = {}
    for k, lam in enumerate((0.125, 1.0)):
        Ca, D = series_D(rows, Q=0.25, lam=lam, theta=180, gamma=0.5, w=20)
        if len(Ca) < 6:
            continue
        if lam == 0.125:
            f = dict(Ca_c=fit["Ca_c"], lo=fit["lo"], hi=fit["hi"], coefs=[fit["coefs"][1]])
        else:
            f = critical_ca([(Ca, D)], nboot=200)
        a.plot(Ca, D, MK[k], color=C[k],
               label=f"λ = {frac(lam)}   Ca$_c$ = {f['Ca_c']:.4f} ({f['lo']:.4f}–{f['hi']:.4f})")
        draw_hinge(a, Ca, f["coefs"][0], f["Ca_c"], C[k])
        a.axvline(f["Ca_c"], color=C[k], lw=1.0)
        out[lam] = dict(Ca_c=f["Ca_c"], lo=f["lo"], hi=f["hi"],
                        D={float(c): float(d) for c, d in zip(Ca, D)})
    a.axvline(PAPER["Ca_c"], color=INK2, ls="--", lw=1.2)
    ylim = a.get_ylim()
    a.text(PAPER["Ca_c"] - 0.0008, ylim[0] + 0.03 * (ylim[1] - ylim[0]),
           "Liu & Zhang\nCa$_c$ = 0.018", color=INK2, fontsize=8.5, ha="right")
    a.set_xlabel("Ca")
    a.set_ylabel("droplet diameter D (µm)")
    a.set_title("Effect of viscosity ratio (Q = 1/4, θ = 180°)")
    a.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig5_viscosity_ratio.png")
    return out


def fig_contact_angle(rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    out = {}
    for j, lam in enumerate((0.125, 1.0)):
        a = ax[j]
        for k, th in enumerate((110, 130, 150, 180)):
            R = sel(rows, Q=0.25, lam=lam, theta=th, gamma=0.5, w=20)
            if th == 180 and lam == 1.0:
                R = [r for r in R if r["Ca"] in (0.008, 0.032, 0.056)]
            if th == 180 and lam == 0.125:
                R = [r for r in R if r["Ca"] in (0.008, 0.016, 0.032, 0.056)]
            if not R:
                continue
            ok = [r for r in R if r["regime"] != "jetting"]
            jet = [r for r in R if r["regime"] == "jetting"]
            a.plot([r["Ca"] for r in ok], [r["D_um"] for r in ok], "-" + MK[k], color=C[k],
                   label=f"θ = {th}°")
            for r in jet:
                a.plot(r["Ca"], r["D_um"], MK[k], mfc=SURF, mec=C[k], mew=2)
            out[(lam, th)] = {r["Ca"]: r["D_um"] for r in R}
        a.set_xlabel("Ca")
        a.set_title(f"({'ab'[j]}) λ = {frac(lam)}, Q = 1/4")
        a.plot([], [], "o", mfc=SURF, mec=INK2, mew=2, label="open: jetting (thread\nstill lengthening)")
        a.legend(loc="upper right", fontsize=8.5)
    ax[0].set_ylabel("droplet diameter D (µm)")
    fig.tight_layout()
    save(fig, "fig6_contact_angle.png")
    return out


# ---------------------------------------------------------------------------
def fig_pressure(rows):
    cases = [("Ca0.005_Q0.250_lam0.125_th180", "Squeezing, Ca = 0.005"),
             ("Ca0.056_Q0.250_lam0.125_th180", "Dripping, Ca = 0.056")]
    cases = [c for c in cases if os.path.exists(os.path.join(DATA, c[0] + ".npz"))]
    if not cases:
        return
    fig, ax = plt.subplots(2, len(cases), figsize=(5.5 * len(cases), 5.6),
                           sharex="col", squeeze=False)
    sigma, w = 0.005, 20
    for j, (nm, title) in enumerate(cases):
        Z, drops = load_npz(nm)
        t = Z["ts_t"]
        tp = np.array([d["t_pinch"] for d in drops])
        if len(tp) < 2:
            continue
        T = np.diff(tp).mean()
        t0 = tp[0]
        m = t >= t0 - 0.2 * T
        tt = (t[m] - t0) / T
        pc = Z["ts_p_c"][m]
        pd = Z["ts_p_d"][m]
        a = ax[0, j]
        a.plot(tt, (pc - pc.min()) * w / sigma, color=C[0], label="continuous phase, upstream")
        a.plot(tt, (pd - pd.min()) * w / sigma, color=C[1], label="dispersed phase, side channel")
        for tk in (tp - t0) / T:
            a.axvline(tk, color=INK2, lw=0.8, ls=":")
        a.set_ylabel(r"pressure rise  $(p - p_{min})\,w/\sigma$")
        a.set_title(title)
        a.legend(loc="upper left", fontsize=8)
        a = ax[1, j]
        a.plot(tt, Z["ts_gap"][m] / w, color=C[2], label="gap to opposite wall")
        a.plot(tt, Z["ts_neck"][m] / w, color=C[3], label="neck width")
        for tk in (tp - t0) / T:
            a.axvline(tk, color=INK2, lw=0.8, ls=":")
        a.set_ylabel("length / w  (node resolution)")
        a.set_xlabel("t / T  (T = break-up period; dotted = pinch-off)")
        a.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig7_pressure_traces.png")


# ---------------------------------------------------------------------------
def fig_regime_and_breakup(rows, fit):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    a = ax[0]
    R = [r for r in rows if r["lam"] == 0.125 and r["theta"] == 180 and r["gamma"] == 0.5
         and r["w"] == 20]
    sty = {"squeezing": dict(marker="o", color=C[0]),
           "dripping": dict(marker="s", color=C[1]),
           "jetting": dict(marker="^", color=C[2])}
    for reg, s in sty.items():
        rr = [r for r in R if r["regime"] == reg]
        if rr:
            a.plot([r["Ca"] for r in rr], [r["Q"] for r in rr], ls="none",
                   label=reg, **s)
    a.axvspan(fit["lo"], fit["hi"], color=C[0], alpha=0.10, lw=0)
    a.axvline(fit["Ca_c"], color=C[0], lw=1.2, label="Ca$_c$ this work")
    a.axvline(PAPER["Ca_c"], color=INK2, ls="--", lw=1.2, label="Ca$_c$ Liu & Zhang")
    a.set_xscale("log")
    a.set_yscale("log", base=2)
    a.set_xlabel("Ca")
    a.set_ylabel(r"Q = Q$_d$/Q$_c$")
    a.set_yticks([0.125, 0.25, 0.375, 0.5, 0.75, 1])
    a.set_yticklabels(["1/8", "1/4", "3/8", "1/2", "3/4", "1"])
    a.set_title("(a) Regime map (λ = 1/8, θ = 180°)")
    plain_ticks(a.xaxis, [0.005, 0.01, 0.02, 0.03, 0.05])
    a.set_ylim(0.1, 1.25)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.5)
    # (b) break-up position
    a = ax[1]
    for k, Q in enumerate((0.125, 0.25, 0.5)):
        rr = sel(rows, Q=Q, lam=0.125, theta=180, gamma=0.5, w=20)
        rr = [r for r in rr if "x_rear_pinch_over_w" in r]
        if rr:
            a.plot([r["Ca"] for r in rr], [r["x_rear_pinch_over_w"] for r in rr],
                   "-" + MK[k], color=C[k], label=f"Q = {frac(Q)}")
    a.set_xlabel("Ca")
    a.set_ylabel(r"rear of new droplet at pinch-off, $x/w$")
    a.axhline(JET_X, color=INK2, ls=":", lw=1)
    a.text(0.03, 0.55, "Q = 1/2, Ca ≥ 0.032: the jet keeps lengthening\n(5.3 w after 10 droplets at Ca 0.032;\nreaches the outlet at Ca 0.056)",
           transform=a.transAxes, fontsize=8, color=INK2)
    a.set_title("(b) Break-up position (from downstream corner)")
    a.legend(loc="upper left", fontsize=9)
    # (c) period
    a = ax[2]
    for k, Q in enumerate((0.125, 0.25, 0.5)):
        rr = sel(rows, Q=Q, lam=0.125, theta=180, gamma=0.5, w=20)
        rr = [r for r in rr if "period_nd" in r]
        if rr:
            a.loglog([r["Ca"] for r in rr], [r["period_nd"] for r in rr],
                     "-" + MK[k], color=C[k], label=f"Q = {frac(Q)}")
    a.set_xlabel("Ca")
    a.set_ylabel(r"break-up period  $T\,\bar u_c / w$")
    plain_ticks(a.xaxis, [0.005, 0.01, 0.02, 0.03, 0.05])
    plain_ticks(a.yaxis, [2, 3, 5, 10, 20, 30])
    a.set_title("(c) Break-up period")
    a.text(0.03, 0.04, "low-Ca periods are stretched by\nphase-field dissolution (see Fig. 9)",
           transform=a.transAxes, color=INK2, fontsize=8)
    a.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, "fig8_regime_breakup.png")


# ---------------------------------------------------------------------------
def fig_diagnostics(rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    for k, Q in enumerate((0.125, 0.25, 0.5)):
        R = [r for r in sel(rows, Q=Q, lam=0.125, theta=180, gamma=0.5, w=20) if "mass_ratio" in r]
        ok = [r for r in R if r["regime"] != "jetting"]
        jet = [r for r in R if r["regime"] == "jetting"]
        if ok:
            a.plot([r["Ca"] for r in ok], [r["mass_ratio"] for r in ok], "-" + MK[k],
                   color=C[k], label=f"Q = {frac(Q)}")
        if jet:
            a.plot([r["Ca"] for r in jet], [r["mass_ratio"] for r in jet], MK[k],
                   mfc=SURF, mec=C[k], mew=2)
    a.plot([], [], "o", mfc=SURF, mec=INK2, mew=2, label="open: jetting (not periodic)")
    a.axhline(1, color=INK2, ls="--", lw=1.2)
    a.set_ylim(0, 1.05)
    a.set_xlabel("Ca")
    a.set_ylabel("droplet volume / dispersed volume injected per period")
    a.set_title("(a) Dispersed-phase mass balance (w = 20)")
    a.legend(loc="lower right", fontsize=9)
    out = {}
    a = ax[1]
    labels, x = [], 0
    for Ca in (0.012, 0.032):
        for k, w in enumerate((20, 40)):
            R = sel(rows, Ca=Ca, Q=0.25, lam=0.125, theta=180, gamma=0.5, w=w)
            if not R:
                continue
            r = R[0]
            a.bar(x, r["D_um"], width=0.7, color=C[k], label=f"w = {w} nodes" if Ca == 0.012 else None)
            a.text(x, r["D_um"] + 1.5, f"{r['D_um']:.1f} µm\nl/w {r['L_over_w']:.2f}\n"
                   f"mass {r.get('mass_ratio', float('nan')):.2f}", ha="center", fontsize=8,
                   color=INK2)
            out[f"Ca{Ca}_w{w}"] = dict(D=r["D_um"], l=r["L_over_w"], mass=r.get("mass_ratio"))
            x += 1
        x += 0.6
    a.set_xticks([0.5, 3.1])
    a.set_xticklabels(["Ca = 0.012 (squeezing)", "Ca = 0.032 (dripping)"])
    a.set_ylabel("droplet diameter D (µm)")
    a.set_ylim(0, 155)
    a.set_title("(b) Grid check: w = 20 vs 40 nodes (Q = 1/4)")
    a.legend(loc="upper right", fontsize=9)
    a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig9_diagnostics.png")
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows = load_runs()
    print(f"{len(rows)} runs loaded")
    summary = {}
    V = fig_validation()
    summary["validation"] = V
    Qs = (0.125, 0.25, 0.5)
    fit = critical_ca([series_D(rows, Q=Q, lam=0.125, theta=180, gamma=0.5, w=20) for Q in Qs])
    assign_regimes(rows, fit["Ca_c"])
    summary["Ca_c"] = {k: v for k, v in fit.items() if k != "coefs"}
    fig_snapshots(rows)
    summary["eq27_fit"] = fig_D_vs_Ca(rows, fit)
    summary["garstecki"] = fig_garstecki(rows)
    summary["viscosity"] = fig_viscosity(rows, fit)
    summary["contact_angle"] = {f"lam{k[0]}_th{k[1]}": v for k, v in fig_contact_angle(rows).items()}
    fig_pressure(rows)
    fig_regime_and_breakup(rows, fit)
    summary["grid_check"] = fig_diagnostics(rows)
    # table of all runs
    keys = ["name", "status", "regime", "Ca", "Q", "lam", "theta", "w", "gamma", "D_um",
            "L_over_w", "H_over_w", "x_rear_pinch_over_w", "period_nd", "mass_ratio",
            "area_spread", "n_measured", "steps", "Re"]
    with open(os.path.join(ROOT, "results", "all_runs.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in sorted(rows, key=lambda r: (r["w"], r["gamma"], r["lam"], r["theta"], r["Q"], r["Ca"])):
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    js = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=js)
    print(json.dumps({k: v for k, v in summary.items() if k != "validation"}, indent=1, default=js))
