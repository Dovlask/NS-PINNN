"""Single source of truth for all physical / geometric constants of the
prolate-spheroid (a/b = 4) potential-flow case.

Pure NumPy / Python only -- NO JAX, NO network. Imported by data_gen.py,
geometry.py, models.py and evaluate.py so that every module agrees on the same
numbers. The neural network never sees the *dimensional* values (U_inf, nu, T);
those exist only for metadata and for the H1/H3 hypothesis checks.

Non-dimensionalization (canonical, .tex Sec. 3 & 6):  L = 2b, x* = x/L,
phi* = phi/(U_inf L)  =>  Cp = 1 - ||grad* phi*||^2.

Equation map (see README.md):
    e, alpha0, k1   -> Hipoteses_loss_Cp.tex eq.(e),(alpha0),(k1); Lamb Art.103
    cp_surface(eta) -> .tex eq.(Cp), Lamb 1932 Art.114 eq.(13)
"""
import math

# ----------------------------------------------------------------------------
# Dimensional nominal scenario (METADATA ONLY -- the network never sees these).
# Air at 20 C, 1 atm.  Campaign v2 (CLAUDE.md Sec. 4 / Sec. 11).
# ----------------------------------------------------------------------------
T_AIR = 293.15          # K  (20 C)
RHO_AIR = 1.204         # kg/m^3
MU_AIR = 1.825e-5       # Pa.s
NU_AIR = MU_AIR / RHO_AIR        # ~1.516e-5 m^2/s
GAMMA = 1.4
R_GAS = 287.0           # J/(kg K)

U_INF = 30.0            # m/s   free-stream speed
B_DIM = 0.075           # m     minor semi-axis
A_DIM = 0.30            # m     major semi-axis (a = 4 b)
ASPECT = A_DIM / B_DIM  # = 4

# Hypothesis-check numbers (printed by data_gen.py, stored in metadata.json).
RE_B = U_INF * (2.0 * B_DIM) / NU_AIR          # ~2.97e5  (> 6e4  => H1 ok)
RE_A = U_INF * (2.0 * A_DIM) / NU_AIR          # = 4 Re_b
MA = U_INF / math.sqrt(GAMMA * R_GAS * T_AIR)  # ~0.087   (< 0.3 => H3 ok)
THETA_SEP_DEG = 128.2   # laminar separation (Thwaites, .tex App. F);
#                         invariant in Re_b/U_inf (theta^2 ~ nu, Ue ~ U_inf).

# ----------------------------------------------------------------------------
# Non-dimensional geometry  (L = 2b).   a* = 2,  b* = 0.5,  r*_inf = 10.
# NOTE: b* = 0.5 (CLAUDE.md v2 correction of the "b* = 0.25" typo).
# ----------------------------------------------------------------------------
L_REF = 2.0 * B_DIM     # characteristic length
A_STAR = A_DIM / L_REF  # = 2.0
B_STAR = B_DIM / L_REF  # = 0.5
R_INF = 5.0 * A_DIM / L_REF   # = 10.0  (physically r_inf = 5a = 1.5 m)

# Shape-only Lamb parameters (depend ONLY on a/b = 4, scale-invariant).
E = math.sqrt(1.0 - (B_STAR / A_STAR) ** 2)        # eccentricity  ~0.96825
C_STAR = A_STAR * E                                # focal semi-distance ~1.93649
XI0 = 1.0 / E                                      # surface coordinate  ~1.03280
ALPHA0 = (2.0 * (1.0 - E**2) / E**3) * (0.5 * math.log((1.0 + E) / (1.0 - E)) - E)
K1 = ALPHA0 / (2.0 - ALPHA0)                       # added-mass coeff  ~0.08156

# Reference values for tests (CLAUDE.md Sec. 4, tol 1e-3).
REF = {
    "e": 0.96825,
    "alpha0": 0.151,
    "k1": 0.0816,
    "cp0": -0.170,
    "ue_max_over_uinf": 1.082,
}


def cp_surface(eta):
    """Surface pressure coefficient, .tex eq.(Cp) (Lamb Art.114 eq.13).

    Cp(eta) = 1 - (1+k1)^2 (1-eta^2)/(1 - e^2 eta^2),   eta = x/a in [-1, 1].
    Accepts float or NumPy array. Closed form -- no autodiff, no network.
    Cp_min = cp_surface(0) ~ -0.170.
    """
    return 1.0 - (1.0 + K1) ** 2 * (1.0 - eta**2) / (1.0 - E**2 * eta**2)


def surface_area_star():
    """Exact non-dimensional area of the prolate spheroid (a*, b*):
        S = 2 pi b*^2 (1 + (a*/(b* e)) arcsin e).
    Used as the analytic target for the Monte-Carlo area test (geometry.py)."""
    return 2.0 * math.pi * B_STAR**2 * (1.0 + (A_STAR / (B_STAR * E)) * math.asin(E))


# ----------------------------------------------------------------------------
# Synthetic-data protocol (data_gen.py).  Frozen-dataset parameters.
# ----------------------------------------------------------------------------
N_TAPS = 40
THETA_MIN_DEG = 5.0
THETA_MAX_DEG = 128.0            # < THETA_SEP_DEG so every tap survives the filter
AZIMUTH_STEP_DEG = 137.5        # golden angle (helical layout, .tex H4a)

# sigma ~ TruncNormal(loc=0, scale=0.025, low=0, high=0.05)  in Cp units.
SIGMA_LOC = 0.0
SIGMA_SCALE = 0.025
SIGMA_LOW = 0.0
SIGMA_HIGH = 0.05
PER_TAP_SIGMA = False           # default; True => one sigma_i per tap (DEVIATIONS.md)

# Separate, fixed seeds (CLAUDE.md Sec. 2.2).  Train seeds live in the configs.
SEED_SIGMA = 20240601           # (a) sigma draw
SEED_NOISE = 20240602           # (b) per-tap noise

# Evaluation sets (evaluate.py) -- generated once, NEVER used in training.
SEED_EVAL_FIELD = 20240701      # 10k 3D field points for rel_L2_field
N_EVAL_FIELD = 10000
N_ETA_DENSE = 600               # dense eta grid for the Cp(eta) figure (>= 500)
