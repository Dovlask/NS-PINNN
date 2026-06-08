"""Generate and FREEZE the synthetic Cp dataset (CLAUDE.md Sec. 5).

Pure NumPy / SciPy -- NO JAX, NO network.  Run ONCE; the output
(data/cp_synthetic.csv + data/metadata.json) is committed and never regenerated
during tuning.  Training reads only the CSV; sigma is recorded in the metadata
but NEVER read by train.py (CLAUDE.md Sec. 5.4).

Protocol
--------
* 40 pressure taps in a HELIX over the surface: meridional angle theta cosine-
  clustered near the nose in [5, 128] deg, azimuth in golden-angle steps
  (137.5 deg).  eta_i = cos(theta_i).  Only taps with theta < theta_sep are kept.
* sigma ~ TruncNormal(loc=0, scale=0.025, [0, 0.05])  (single draw, seed_sigma).
* eps_i ~ N(0, sigma^2)  i.i.d.  (seed_noise).   cp_med_i = Cp(eta_i) + eps_i.

Reproducible from the seeds alone (the git hash / timestamp are provenance only).
"""
import os
import csv
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
from scipy.stats import truncnorm

import constants as C

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "cp_synthetic.csv")
META_PATH = os.path.join(DATA_DIR, "metadata.json")


def helical_taps():
    """Return (eta, theta_deg, phi_deg) for the helical tap layout (.tex H4a)."""
    k = np.arange(C.N_TAPS)
    # cosine clustering near the nose: dense at theta_min, coarse toward the tail
    theta = C.THETA_MIN_DEG + (C.THETA_MAX_DEG - C.THETA_MIN_DEG) * (
        1.0 - np.cos(0.5 * np.pi * k / (C.N_TAPS - 1))
    )
    phi = (k * C.AZIMUTH_STEP_DEG) % 360.0
    eta = np.cos(np.deg2rad(theta))
    return eta, theta, phi


def draw_sigma():
    """Single sigma ~ TruncNormal(loc, scale, [low, high]) (CLAUDE.md Sec. 5.2)."""
    a = (C.SIGMA_LOW - C.SIGMA_LOC) / C.SIGMA_SCALE
    b = (C.SIGMA_HIGH - C.SIGMA_LOC) / C.SIGMA_SCALE
    dist = truncnorm(a, b, loc=C.SIGMA_LOC, scale=C.SIGMA_SCALE)
    if C.PER_TAP_SIGMA:
        return dist.rvs(size=C.N_TAPS, random_state=np.random.default_rng(C.SEED_SIGMA))
    return float(dist.rvs(random_state=np.random.default_rng(C.SEED_SIGMA)))


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    eta, theta, phi = helical_taps()
    keep = theta < C.THETA_SEP_DEG            # H4a filter (all pass here by design)
    eta, theta, phi = eta[keep], theta[keep], phi[keep]
    n = len(eta)

    # surface cartesian coordinates of each tap (a* = 2, b* = 0.5)
    x = C.A_STAR * eta
    rho = C.B_STAR * np.sqrt(1.0 - eta**2)
    y = rho * np.cos(np.deg2rad(phi))
    z = rho * np.sin(np.deg2rad(phi))

    cp_true = C.cp_surface(eta)               # .tex eq.(Cp) -- analytic ground truth

    sigma = draw_sigma()
    noise_rng = np.random.default_rng(C.SEED_NOISE)
    if C.PER_TAP_SIGMA:
        eps = noise_rng.normal(0.0, sigma)    # sigma is a vector here
        sigma_record = sigma.tolist()
        sigma_rms = float(np.sqrt(np.mean(sigma**2)))
    else:
        eps = noise_rng.normal(0.0, sigma, size=n)
        sigma_record = sigma
        sigma_rms = sigma
    cp_med = cp_true + eps

    # ---- freeze CSV ----
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "eta", "theta_deg", "phi_deg", "cp_true", "cp_med"])
        for i in range(n):
            w.writerow([f"{x[i]:.10e}", f"{y[i]:.10e}", f"{z[i]:.10e}",
                        f"{eta[i]:.10e}", f"{theta[i]:.6f}", f"{phi[i]:.6f}",
                        f"{cp_true[i]:.10e}", f"{cp_med[i]:.10e}"])

    # ---- freeze metadata ----
    meta = {
        "case": "prolate_spheroid a/b=4, alpha=0 (axisymmetric potential flow)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash(),
        "physical": {
            "T_air_K": C.T_AIR, "rho_air": C.RHO_AIR, "mu_air": C.MU_AIR,
            "nu_air": C.NU_AIR, "U_inf": C.U_INF, "a": C.A_DIM, "b": C.B_DIM,
            "aspect_ratio": C.ASPECT, "gamma": C.GAMMA, "R_gas": C.R_GAS,
        },
        "hypothesis_checks": {
            "Re_b": C.RE_B, "Re_a": C.RE_A, "Ma": C.MA,
            "Re_b_gt_6e4": bool(C.RE_B > 6e4), "Ma_lt_0p3": bool(C.MA < 0.3),
            "theta_sep_deg": C.THETA_SEP_DEG,
        },
        "nondimensional": {
            "L_ref": C.L_REF, "a_star": C.A_STAR, "b_star": C.B_STAR,
            "r_star_inf": C.R_INF, "e": C.E, "c_star": C.C_STAR,
            "xi0": C.XI0, "alpha0": C.ALPHA0, "k1": C.K1,
        },
        "data_protocol": {
            "N_taps": n, "theta_min_deg": C.THETA_MIN_DEG,
            "theta_max_deg": C.THETA_MAX_DEG, "azimuth_step_deg": C.AZIMUTH_STEP_DEG,
            "per_tap_sigma": C.PER_TAP_SIGMA,
            "sigma": sigma_record, "sigma_rms": sigma_rms,
            "sigma_dist": {"family": "truncnorm", "loc": C.SIGMA_LOC,
                           "scale": C.SIGMA_SCALE, "low": C.SIGMA_LOW,
                           "high": C.SIGMA_HIGH},
        },
        "seeds": {"seed_sigma": C.SEED_SIGMA, "seed_noise": C.SEED_NOISE},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # ---- report (CLAUDE.md Sec. 9.4) ----
    print("Frozen synthetic Cp dataset")
    print(f"  taps kept (theta < {C.THETA_SEP_DEG} deg): {n}/{C.N_TAPS}")
    print(f"  sigma (drawn, seed {C.SEED_SIGMA})        : {sigma_rms:.6f}  [Cp units]")
    print(f"  Re_b = {C.RE_B:.3e}  (> 6e4 -> H1 {'OK' if C.RE_B > 6e4 else 'FAIL'})")
    print(f"  Ma   = {C.MA:.4f}    (< 0.3 -> H3 {'OK' if C.MA < 0.3 else 'FAIL'})")
    print(f"  cp_true range : [{cp_true.min():.4f}, {cp_true.max():.4f}]")
    print(f"  csv      -> {CSV_PATH}")
    print(f"  metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
