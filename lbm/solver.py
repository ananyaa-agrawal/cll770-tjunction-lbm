"""
Phase-field lattice Boltzmann solver for two-phase flow (D2Q9).

Model: H. Liu & Y. Zhang, J. Appl. Phys. 106, 034906 (2009), Sec. II.
  * f_i : flow field, pressure-based equilibrium (incompressible, rho = 1)
          f_i^eq = w_i [ p/cs^2 + (e.u)/cs^2 + (e.u)^2/(2cs^4) - u^2/(2cs^2) ]
          forcing (1 - 1/(2 tau_f)) (e_i - u).F / cs^2 * Gamma_i(u),  F = mu grad(phi)
  * g_i : order parameter, Cahn-Hilliard via
          g_i^eq = w_i [ A_i + phi ((e.u)/cs^2 + ...) ],
          A_i = Gamma*mu/cs^2 (i>0),  A_0 = [phi - (1-w0) Gamma mu / cs^2] / w0
          mobility M = Gamma (tau_g - 1/2), tau_g = 1/(3 - sqrt 3)
  * free energy  psi = a/4 (phi^2-1)^2 + kappa/2 |grad phi|^2
          mu = a phi (phi^2 - 1) - kappa lap(phi)
          xi = sqrt(2 kappa / a),   sigma = 4 kappa / (3 xi)
  * wetting:  wall order parameter phi_w from cos(theta) = phi_w (3 - phi_w^2) / 2
          theta is measured through the phi = +1 phase (dispersed phase here).

Phase convention: continuous phase phi = -1, dispersed phase phi = +1.

Node types: FLUID, WALL (half-way bounce-back, phi = phi_w), INLET ghost
(velocity profile, non-equilibrium extrapolation; pure phase), OUTLET ghost
(fixed pressure, non-equilibrium extrapolation; zero-gradient for phi).
Everything lives on flattened arrays with a neighbour table, so the same
kernels handle T-junctions, periodic boxes and wall-bounded boxes.
"""
import numpy as np
from numba import njit

EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0

FLUID, WALL, INLET, OUTLET = 0, 1, 2, 3
TAU_G = 1.0 / (3.0 - np.sqrt(3.0))


def phi_wall(theta_deg):
    """Wall order parameter for a contact angle (Liu & Zhang Eq. 23)."""
    return _solve_phiw(np.cos(np.deg2rad(theta_deg)))


def _solve_phiw(c):
    lo, hi = -1.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * mid * (3 - mid * mid) < c:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Geometry:
    """Node-type map (nx, ny) -> flattened arrays + neighbour table."""

    def __init__(self, ntype2d, periodic_x=False, periodic_y=False):
        self.ntype2d = np.asarray(ntype2d, dtype=np.int64)
        self.nx, self.ny = self.ntype2d.shape
        nx, ny = self.nx, self.ny
        self.N = nx * ny
        self.ntype = self.ntype2d.ravel().copy()
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        X, Y = X.ravel(), Y.ravel()
        nb = np.empty((self.N, 9), dtype=np.int64)
        for i in range(9):
            xn, yn = X + EX[i], Y + EY[i]
            if periodic_x:
                xn %= nx
            if periodic_y:
                yn %= ny
            bad = (xn < 0) | (xn >= nx) | (yn < 0) | (yn >= ny)
            idx = np.where(bad, np.arange(self.N), xn * ny + yn)
            nb[:, i] = idx
        self.nb = nb
        self.fluid = np.where(self.ntype == FLUID)[0].astype(np.int64)
        self.X, self.Y = X, Y
        # boundary-condition data (filled by builders)
        self.inl_idx = np.zeros(0, np.int64)
        self.inl_nb = np.zeros(0, np.int64)
        self.inl_u = np.zeros(0)
        self.inl_v = np.zeros(0)
        self.inl_phi = np.zeros(0)
        self.out_idx = np.zeros(0, np.int64)
        self.out_nb = np.zeros(0, np.int64)

    def idx(self, x, y):
        return x * self.ny + y


def build_tjunction(w=20, wd=20, L=300, Ld=40, xj=50):
    """
    Main channel (width w, length L) along x at the bottom; lateral channel
    (width wd, length Ld) entering from the top at fluid columns xj..xj+wd-1.
    Fluid coordinates: x in [0, L), y in [0, w) for the main channel.
    Array index = fluid coordinate + 1 (one padding layer).
    """
    nx, ny = L + 2, w + Ld + 2
    t = np.full((nx, ny), WALL, dtype=np.int64)
    t[1:L + 1, 1:w + 1] = FLUID
    t[xj + 1:xj + wd + 1, w + 1:w + Ld + 1] = FLUID
    t[0, 1:w + 1] = INLET                     # continuous inlet (left)
    t[xj + 1:xj + wd + 1, w + Ld + 1] = INLET  # dispersed inlet (top)
    t[L + 1, 1:w + 1] = OUTLET                # outlet (right)
    geo = Geometry(t)
    geo.w, geo.wd, geo.L, geo.Ld, geo.xj = w, wd, L, Ld, xj
    # inlet data: (ghost, interior neighbour, u, v, phi); profiles set later
    ci = [geo.idx(0, y + 1) for y in range(w)]
    cn = [geo.idx(1, y + 1) for y in range(w)]
    di = [geo.idx(xj + 1 + x, w + Ld + 1) for x in range(wd)]
    dn = [geo.idx(xj + 1 + x, w + Ld) for x in range(wd)]
    geo.inl_idx = np.array(ci + di, dtype=np.int64)
    geo.inl_nb = np.array(cn + dn, dtype=np.int64)
    geo.n_cin = w
    geo.inl_u = np.zeros(w + wd)
    geo.inl_v = np.zeros(w + wd)
    geo.inl_phi = np.concatenate([-np.ones(w), np.ones(wd)])
    geo.out_idx = np.array([geo.idx(L + 1, y + 1) for y in range(w)], dtype=np.int64)
    geo.out_nb = np.array([geo.idx(L, y + 1) for y in range(w)], dtype=np.int64)
    return geo


def poiseuille_profile(n, umean):
    """Discrete parabolic profile for n nodes between half-way walls."""
    s = np.arange(n) + 0.5
    return 6.0 * umean * s * (n - s) / n ** 2


# ----------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------
@njit(cache=True)
def _feq(i, s, u, v):
    eu = EX[i] * u + EY[i] * v
    return W[i] * (s + 3.0 * eu + 4.5 * eu * eu - 1.5 * (u * u + v * v))


@njit(cache=True)
def _geq(i, ph, gm, u, v):
    eu = EX[i] * u + EY[i] * v
    vel = W[i] * (3.0 * eu + 4.5 * eu * eu - 1.5 * (u * u + v * v))
    if i == 0:
        return ph - (1.0 - W[0]) * 3.0 * gm + ph * vel
    return W[i] * 3.0 * gm + ph * vel


@njit(cache=True)
def compute_mu(fl, nb, phi, mu, a, kappa):
    for k in range(fl.size):
        n = fl[k]
        lap = 0.0
        pn = phi[n]
        for i in range(1, 9):
            lap += W[i] * (phi[nb[n, i]] - pn)
        lap *= 6.0
        mu[n] = a * pn * (pn * pn - 1.0) - kappa * lap


@njit(cache=True, fastmath=True)
def advance(nsteps, fl, nb, ntype, f, g, fpost, gpost, phi, mu, p, ux, uy,
            inl_idx, inl_nb, inl_u, inl_v, inl_phi, out_idx, out_nb, p_out,
            a, kappa, eta_c, eta_d, gamma, tau_g, body_fx):
    om_g = 1.0 / tau_g
    for step in range(nsteps):
        # zero-gradient phi on outlet ghosts
        for k in range(out_idx.size):
            phi[out_idx[k]] = phi[out_nb[k]]
        # ---- collision at fluid nodes (mu, grad phi and force fused)
        for k in range(fl.size):
            n = fl[k]
            gx = 0.0
            gy = 0.0
            lap = 0.0
            pn = phi[n]
            for i in range(1, 9):
                ph = phi[nb[n, i]]
                gx += W[i] * EX[i] * ph
                gy += W[i] * EY[i] * ph
                lap += W[i] * (ph - pn)
            gx *= 3.0
            gy *= 3.0
            mun = a * pn * (pn * pn - 1.0) - kappa * 6.0 * lap
            mu[n] = mun
            Fx = mun * gx + body_fx
            Fy = mun * gy
            s = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                fi = f[n, i]
                s += fi
                mx += fi * EX[i]
                my += fi * EY[i]
            u = mx + 0.5 * Fx
            v = my + 0.5 * Fy
            p[n] = CS2 * s
            ux[n] = u
            uy[n] = v
            ph = phi[n]
            c = min(1.0, max(-1.0, ph))
            eta = 0.5 * (1.0 - c) * eta_c + 0.5 * (1.0 + c) * eta_d
            om = 1.0 / (3.0 * eta + 0.5)
            fac = 1.0 - 0.5 * om
            usq = u * u + v * v
            gm = gamma * mun
            for i in range(9):
                eu = EX[i] * u + EY[i] * v
                vel = W[i] * (3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
                Gi = W[i] + vel
                feq = W[i] * s + vel
                Fi = fac * 3.0 * ((EX[i] - u) * Fx + (EY[i] - v) * Fy) * Gi
                fpost[n, i] = f[n, i] - om * (f[n, i] - feq) + Fi
                if i == 0:
                    geq = ph - (1.0 - W[0]) * 3.0 * gm + ph * vel
                else:
                    geq = W[i] * 3.0 * gm + ph * vel
                gpost[n, i] = g[n, i] - om_g * (g[n, i] - geq)
        # ---- inlet ghosts: velocity, non-equilibrium extrapolation; pure phase
        for k in range(inl_idx.size):
            n = inl_idx[k]
            m = inl_nb[k]
            s = 3.0 * p[m]
            c = min(1.0, max(-1.0, phi[m]))
            eta = 0.5 * (1.0 - c) * eta_c + 0.5 * (1.0 + c) * eta_d
            om = 1.0 / (3.0 * eta + 0.5)
            for i in range(9):
                fpost[n, i] = _feq(i, s, inl_u[k], inl_v[k]) + (1.0 - om) * (
                    f[m, i] - _feq(i, s, ux[m], uy[m]))
                gpost[n, i] = _geq(i, inl_phi[k], 0.0, inl_u[k], inl_v[k])
        # ---- outlet ghosts: fixed pressure, velocity extrapolated
        for k in range(out_idx.size):
            n = out_idx[k]
            m = out_nb[k]
            s = 3.0 * p[m]
            c = min(1.0, max(-1.0, phi[m]))
            eta = 0.5 * (1.0 - c) * eta_c + 0.5 * (1.0 + c) * eta_d
            om = 1.0 / (3.0 * eta + 0.5)
            so = 3.0 * p_out
            for i in range(9):
                fpost[n, i] = _feq(i, so, ux[m], uy[m]) + (1.0 - om) * (
                    f[m, i] - _feq(i, s, ux[m], uy[m]))
                gpost[n, i] = gpost[m, i]
        # ---- streaming (pull) with half-way bounce-back
        for k in range(fl.size):
            n = fl[k]
            sphi = 0.0
            for i in range(9):
                src = nb[n, OPP[i]]
                if ntype[src] == 1:
                    f[n, i] = fpost[n, OPP[i]]
                    g[n, i] = gpost[n, OPP[i]]
                else:
                    f[n, i] = fpost[src, i]
                    g[n, i] = gpost[src, i]
                sphi += g[n, i]
            phi[n] = sphi


@njit(cache=True)
def macros(fl, nb, f, phi, mu, p, ux, uy, body_fx):
    """Recompute p, u at fluid nodes from current f (for output)."""
    for k in range(fl.size):
        n = fl[k]
        gx = 0.0
        gy = 0.0
        for i in range(1, 9):
            ph = phi[nb[n, i]]
            gx += W[i] * EX[i] * ph
            gy += W[i] * EY[i] * ph
        Fx = mu[n] * 3.0 * gx + body_fx
        Fy = mu[n] * 3.0 * gy
        s = 0.0
        mx = 0.0
        my = 0.0
        for i in range(9):
            s += f[n, i]
            mx += f[n, i] * EX[i]
            my += f[n, i] * EY[i]
        p[n] = CS2 * s
        ux[n] = mx + 0.5 * Fx
        uy[n] = my + 0.5 * Fy


@njit(cache=True)
def _init_eq(f, g, p, ux, uy, phi, mu, gamma):
    for n in range(f.shape[0]):
        s = 3.0 * p[n]
        for i in range(9):
            f[n, i] = _feq(i, s, ux[n], uy[n])
            g[n, i] = _geq(i, phi[n], gamma * mu[n], ux[n], uy[n])


class Simulation:
    """
    Parameters (lattice units): sigma, xi (interface width), eta_c, eta_d,
    gamma (mobility coefficient), theta (contact angle through phi=+1 phase).
    """

    def __init__(self, geo, sigma=0.005, xi=2.0, eta_c=0.08, eta_d=0.01,
                 gamma=0.5, theta=180.0, p_out=CS2, body_fx=0.0):
        self.geo = geo
        self.kappa = 0.75 * sigma * xi
        self.a = 2.0 * self.kappa / xi ** 2
        self.sigma, self.xi = sigma, xi
        self.eta_c, self.eta_d = eta_c, eta_d
        self.gamma, self.tau_g = gamma, TAU_G
        self.M = gamma * (TAU_G - 0.5)
        self.theta = theta
        self.phiw = _solve_phiw(np.cos(np.deg2rad(theta)))
        self.p_out = p_out
        self.body_fx = body_fx
        N = geo.N
        self.phi = np.zeros(N)
        self.mu = np.zeros(N)
        self.p = np.full(N, CS2)
        self.ux = np.zeros(N)
        self.uy = np.zeros(N)
        self.f = np.zeros((N, 9))
        self.g = np.zeros((N, 9))
        self.fpost = np.zeros((N, 9))
        self.gpost = np.zeros((N, 9))
        self.t = 0

    def set_phi(self, phi_fluid):
        """phi_fluid: array (N,) values used at fluid nodes."""
        geo = self.geo
        self.phi[:] = phi_fluid
        self.phi[geo.ntype == WALL] = self.phiw
        self.phi[geo.inl_idx] = geo.inl_phi
        self.phi[geo.out_idx] = self.phi[geo.out_nb]

    def init_equilibrium(self, u0=None, v0=None):
        geo = self.geo
        compute_mu(geo.fluid, geo.nb, self.phi, self.mu, self.a, self.kappa)
        if u0 is not None:
            self.ux[:] = u0
        if v0 is not None:
            self.uy[:] = v0
        _init_eq(self.f, self.g, self.p, self.ux, self.uy, self.phi, self.mu,
                 self.gamma)

    def run(self, nsteps):
        geo = self.geo
        advance(nsteps, geo.fluid, geo.nb, geo.ntype, self.f, self.g, self.fpost,
                self.gpost, self.phi, self.mu, self.p, self.ux, self.uy,
                geo.inl_idx, geo.inl_nb, geo.inl_u, geo.inl_v, geo.inl_phi,
                geo.out_idx, geo.out_nb, self.p_out, self.a, self.kappa,
                self.eta_c, self.eta_d, self.gamma, self.tau_g, self.body_fx)
        self.t += nsteps
        compute_mu(geo.fluid, geo.nb, self.phi, self.mu, self.a, self.kappa)
        macros(geo.fluid, geo.nb, self.f, self.phi, self.mu, self.p, self.ux,
               self.uy, self.body_fx)

    def field(self, arr):
        return arr.reshape(self.geo.nx, self.geo.ny)
