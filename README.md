# CLL770 term project — LBM simulation of droplet formation in a T-junction

Phase-field lattice Boltzmann solver (Python + Numba) that reproduces
H. Liu & Y. Zhang, *J. Appl. Phys.* **106**, 034906 (2009).

CLL770 (Introduction to Microfluidics and Fabrication), IIT Delhi — term project by
Ananyaa Agrawal, Yash Balram Sakhare and Rishita Anand.

## Layout

```
lbm/solver.py          D2Q9 phase-field LBM (Liu & Zhang model), geometry, BCs
lbm/tjunction.py       T-junction case: set-up, run loop, droplet tracking
scripts/validate.py    Laplace, Poiseuille and static contact-angle tests
scripts/run_sweeps.py  all parameter sweeps (restartable; finished cases skipped)
scripts/run_jetting.py longer runs of the two jetting cases (results/jetting/, data/jetting/)
scripts/make_figures.py  figures/*.png, results/all_runs.csv, results/summary.json
results/runs/          one .json summary per case
data/runs/             one .npz per case (time series, phi snapshots) used by Figs. 2, 7-9
figures/               all plots
```

## Reproduce everything

```bash
pip install -r requirements.txt     # Python 3.10+
python scripts/validate.py          # ~5 min
python scripts/run_sweeps.py 2      # 2 worker processes; ~2.5 h on 2 cores
python scripts/run_jetting.py       # optional, ~5 min
python scripts/make_figures.py
```

Run a single case:

```python
from lbm.tjunction import TJunctionCase
case = TJunctionCase(Ca=0.02, Q=0.25, lam=0.125, theta=180)
res = case.run(verbose=True)
print(res["D_um"], res["L_over_w"])
```

## Model and set-up (lattice units)

| Item | Value | Source |
|---|---|---|
| Lattice | D2Q9, BGK, rho = 1 | paper |
| Flow LBE | pressure-based equilibrium, forcing (1 - 1/2tau)(e - u).F/cs² Γ_i, F = µ∇φ | paper Eqs. |
| Interface LBE | Cahn-Hilliard, tau_g = 1/(3 - √3), M = Γ(tau_g - 1/2) | paper |
| Free energy | ψ = a/4 (φ² - 1)² + κ/2 |∇φ|², ξ = √(2κ/a), σ = 4κ/(3ξ) | paper |
| σ, η_c, λ | 0.005, 0.08, 1/8 (λ = 1 for viscosity study) | paper Sec. III.B |
| ξ | 2 lattice units (interface ≈ 5-6 nodes) | paper |
| Γ (mobility) | 0.5 → M = 0.144 | **not given in paper — our choice** (Pe = O(10-100)) |
| Channels | w_c = w_d = 20 nodes = 100 µm (5 µm/node), domain 300 × 60 | paper (5 µm spacing inferred from its unit conversion) |
| Junction position | side channel at x = 50-69 (2.5 w downstream of inlet) | **not given — our choice** |
| Walls | half-way bounce-back; wetting via wall order parameter cos θ = φ_w(3 - φ_w²)/2 | paper Eq. 23 |
| Inlets | fully developed Poiseuille profiles (non-equilibrium extrapolation) | proposal |
| Outlet | fixed pressure | paper |
| Ca | η_c ū_c / σ = 0.005 – 0.056 | paper |
| Q | Q_d / Q_c = 1/8 – 1 | paper / proposal |

Phase convention: continuous phase φ = -1, dispersed phase φ = +1; θ is measured through
the dispersed phase (θ = 180° means the dispersed phase never wets the wall).

**Droplet size:** D = √(4A/π) from the area A enclosed by the φ = 0 contour (2D equivalent
diameter, converted at 5 µm/node). The first droplet (affected by the start-up) is
discarded; the next two are averaged (one for Ca < 0.015, where successive droplets agree
to < 0.5%). **Length l:** streamwise extent of the φ = 0 contour.
**Critical Ca:** D(Ca) for Q = 1/8, 1/4, 1/2 is fitted with two power laws per Q that
meet at one common break point Ca_c (the kink Liu & Zhang read off their Fig. 5);
16–84% interval from a bootstrap that drops 20% of the points.
**Regimes:** squeezing for Ca < Ca_c, dripping above; jetting when the break-up point is
more than 2 w downstream of the junction and still moving downstream droplet after droplet.

## Scope compared with the proposal

| Item | Proposal | Runs here |
|---|---|---|
| Ca | ≈ 0.005–0.05 | 0.005–0.056 (10 values) |
| Q = Q_d/Q_c | 0.125–1 | 1/8, 1/4, 1/2 at every Ca; 3/8, 3/4, 1 added at Ca = 0.005 |
| Viscosity ratio λ | 0.1–1 | 1/8 (the paper's value) and 1 |
| Contact angle θ | 120°–180° | 110°, 130°, 150°, 180° |
| Extra | – | mobility check (Γ = 0.1), grid check (w = 40 nodes), long jetting runs |

## Results compared with Liu & Zhang (2009)

All runs: `results/all_runs.csv` (63 cases). Fitted numbers: `results/summary.json`.

| Item | Liu & Zhang | This work | Fig. |
|---|---|---|---|
| Laplace law | – | fitted σ = 0.00503 (imposed 0.005); every radius within 2.4% of σ/R | 1a |
| Poiseuille flow | – | profile error 0.22%, dp/dx error 0.02%, flux error 0.11% | 1b |
| Static contact angle | – | 91.7°, 111.5°, 132.4°, 157.7° for imposed 90°, 110°, 130°, 150° | 1c |
| Squeezing → dripping, Ca_c | 0.018, same for all Q | **0.0179** (0.0173–0.0189), one common break point for Q = 1/8, 1/4, 1/2 | 3 |
| Garstecki law at Ca = 0.005 | l/w = 1 + 1.82 Q | **1 + 1.79 Q** over the paper's range Q = 1/8–1/2; 1 + 1.73 Q over Q = 1/8–1 | 4 |
| Viscosity ratio | squeezing: D nearly independent of λ; dripping: larger λ → smaller D; Ca_c unchanged | squeezing: λ = 1 and 1/8 within 0.6%; dripping: λ = 1 droplets 8–11% smaller; Ca_c(λ = 1) = 0.020 (0.0199–0.0225) | 5 |
| Wettability | D decreases as θ increases, most at small Ca; larger λ gives slightly larger D in squeezing, smaller in dripping | θ 110° → 180° (λ = 1/8): D −8.6% at Ca = 0.008, −7.3% at Ca = 0.032; at Ca = 0.056 the curves nearly merge (130° → 180°: −1.3%). λ trend reproduced at θ = 110° and 150°: λ = 1 droplets 0.2–0.5% larger at Ca = 0.008, 8–10% smaller at Ca = 0.032 | 6 |
| Break-up position | Q = 1/8: downstream corner at all Ca; Q = 1/4, 1/2: moves downstream with Ca; jet at high Ca and Q; higher λ moves it further downstream | Q = 1/8: within 0.2 w of the corner at all Ca; Q = 1/4: moves to 0.9 w; Q = 1/2: jetting for Ca ≥ 0.032; λ = 1: jetting at Ca = 0.056 | 8 |
| Break-up mechanism | upstream pressure build-up drives squeezing pinch-off | upstream continuous-phase pressure climbs ≈ 2 σ/w while the droplet blocks the channel and drops at pinch-off; dispersed pressure stays nearly flat | 7 |
| Power law V ∝ Ca^m (Eq. 27) | m = −0.78 | not compared like-for-like: the paper's Sec. III.A sweep (fixed lattice Q_d, λ = 1/4, Ca down to 5.6 × 10⁻⁴) was outside this project's scope. At fixed Q we get D ∝ Ca^−0.06…−0.08 below Ca_c and Ca^−0.12…−0.23 above | 3b |

For Q ≤ 1/2 the droplet diameters span 82–130 µm (Fig. 3), inside the roughly 80–140 µm
axis range of the paper's Fig. 5.

## Limitations and checks (read before writing these up)

1. **Dispersed phase partly dissolves at this resolution.** With ξ/w = 0.1 (the paper's
   values), the squeezed dispersed phase has a raised chemical potential, so the bulk order
   parameter shifts (continuous phase drifts from −1 to about −0.9) and part of the injected
   dispersed fluid leaves dissolved in the continuous stream. Only 20–92% of the injected
   volume ends up in droplets (Fig. 9a; worst at low Ca and low Q). This stretches the
   low-Ca break-up periods (Fig. 8c) and shrinks squeezing droplets. Changing the mobility
   (Γ = 0.1 instead of 0.5) does not remove it (D = 99.1 vs 98.1 µm at Ca = 0.012, Q = 1/8).
   It comes from the diffuse interface, so the paper's model at the same resolution should
   share it.
2. **Grid check (w = 40 nodes, ξ/w = 0.05, Fig. 9b).** Dripping (Ca = 0.032) changes by
   1.3% in D. Squeezing (Ca = 0.012) changes by +14% in D (l/w 1.46 → 2.02) because the
   mass balance improves from 0.77 to 0.95. So the squeezing-regime sizes at the paper's
   resolution are under-predicted; the paper's "≤ 5% mesh dependence" is not confirmed here.
   The match to the paper's numbers is a match at the paper's resolution.
3. **Wall condition at high angles:** an imposed 150° reads 157.7° on a static droplet.
4. **2D, equal densities, Re = 0.08–0.9**, as in the paper; 2D sizes give trends and
   scaling, not 3D experimental droplet sizes.
5. **Choices not specified in the paper:** mobility Γ, junction position, inlet profile
   shape, and the droplet-size definition (2D equivalent diameter). If the paper used a
   different size definition, absolute D values shift but l/w, Ca_c and trends do not.
6. **Jetting cases are not periodic** in a 300-node channel: at Q = 1/2 the jet lengthens to
   5.3 w after 10 droplets at Ca = 0.032 and reaches the outlet at Ca = 0.056.

## Figures

| File | Content |
|---|---|
| fig1_validation.png | Laplace, Poiseuille, contact-angle tests |
| fig2_snapshots.png | one break-up cycle: squeezing, near Ca_c, dripping, jetting |
| fig3_size_vs_Ca.png | D vs Ca for Q = 1/8, 1/4, 1/2 with Ca_c fit (linear and log-log) |
| fig4_garstecki.png | l/w vs Q at Ca = 0.005 against 1 + 1.82 Q |
| fig5_viscosity_ratio.png | λ = 1/8 vs 1 |
| fig6_contact_angle.png | θ = 110°, 130°, 150°, 180° for λ = 1/8 and 1 |
| fig7_pressure_traces.png | upstream/side-channel pressure, gap and neck width vs time |
| fig8_regime_breakup.png | regime map, break-up position, break-up period |
| fig9_diagnostics.png | dispersed-phase mass balance, w = 20 vs 40 grid check |
