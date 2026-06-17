"""Post-training evaluation (CLAUDE.md Sec. 7).  Runs AFTER everything is frozen.

This is the ONLY training-time-adjacent module (besides data_gen.py) that imports
the analytic Lamb solution -- here it is legitimate (no leakage; evaluation only).

Metrics -> results/<run_name>/metrics.json ; figures -> results/<run_name>/figures/.
Acceptance criteria are the ones pre-registered in EXPERIMENTS.md.
"""
import os
import csv
import json

import numpy as np

import jax
import jax.numpy as jnp

import ml_collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
import geometry
import lamb
import constants as C
from train import load_taps

# pre-registered acceptance criteria (EXPERIMENTS.md / CLAUDE.md Sec. 7)
CRIT_REL_L2_FIELD = 0.02
SIGMA_HAT_LO, SIGMA_HAT_HI = 0.5, 1.5
RESIDUAL_ORDERS_BELOW = 1e-2  # >= 2 orders of magnitude below O(1)


def _load_sigma(config):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", config.meta_file)) as f:
        return json.load(f)["data_protocol"]["sigma_rms"]


def _build_model(config):
    """Rebuild the model with the same seeded pools so the checkpoint restores."""
    (tap_xyz, tap_cp), holdout = load_taps(config)
    surf, normals = geometry.sample_surface(config.sampling.n_surface, config.seed_surface)
    far = geometry.sample_farfield(config.sampling.n_farfield, config.seed_far)
    model = models.ProlateSpheroid(config, surf, normals, far, tap_xyz, tap_cp)
    return model, (tap_xyz, tap_cp), holdout, surf, normals, far


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    run_dir = os.path.join(workdir, "results", config.run_name)
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sigma = _load_sigma(config)
    model, (tap_xyz, tap_cp), (ho_xyz, ho_cp), _surf, _normals, _far = _build_model(config)

    # restore_checkpoint reduces the replicated state to a single device and
    # returns it directly -- params are taken as-is (no extra unreplicate).
    model.state = restore_checkpoint(model.state, os.path.join(run_dir, "ckpt"))
    params = model.state.params

    metrics = {}

    # ---- 1. rel_L2_field on the fixed 10k 3D set (never seen in training) ----
    eval_coords = geometry.sample_field_eval(C.N_EVAL_FIELD, C.SEED_EVAL_FIELD)
    g_w = np.asarray(model.velocity_field(params, jnp.asarray(eval_coords)))
    g_lamb = np.asarray(lamb.grad_phi_batch(jnp.asarray(eval_coords)))
    rel_l2 = float(np.linalg.norm(g_w - g_lamb) / np.linalg.norm(g_lamb))
    metrics["rel_L2_field"] = rel_l2

    # ---- 2. Cp on a dense meridian grid (>= 500 eta), NOT at the taps ----
    eta = np.linspace(-0.999, 0.999, C.N_ETA_DENSE)
    xs = C.A_STAR * eta
    rho = C.B_STAR * np.sqrt(1.0 - eta**2)
    surf_grid = np.stack([xs, rho, np.zeros_like(xs)], axis=1)  # az = 0 meridian
    cp_w_grid = np.asarray(model.cp_at(params, jnp.asarray(surf_grid)))
    cp_lamb_grid = C.cp_surface(eta)
    metrics["RMSE_cp_surface"] = float(np.sqrt(np.mean((cp_w_grid - cp_lamb_grid) ** 2)))
    metrics["max_abs_dCp_surface"] = float(np.max(np.abs(cp_w_grid - cp_lamb_grid)))

    # ---- 3. sigma_hat on train and holdout taps ----
    cp_w_train = np.asarray(model.cp_at(params, jnp.asarray(tap_xyz)))
    cp_w_ho = np.asarray(model.cp_at(params, jnp.asarray(ho_xyz)))
    sigma_hat_train = float(np.sqrt(np.mean((cp_w_train - tap_cp) ** 2)))
    sigma_hat_ho = float(np.sqrt(np.mean((cp_w_ho - ho_cp) ** 2)))
    metrics["sigma"] = sigma
    metrics["sigma_hat_train"] = sigma_hat_train
    metrics["sigma_hat_holdout"] = sigma_hat_ho

    # ---- 4. PDE / BC residuals on INDEPENDENT eval sets (own seeds, not the
    #         training pools -- so BC-1/BC-2 are not checked where they were fit) ----
    lap = np.asarray(model.laplacian_at(params, jnp.asarray(eval_coords)))
    surf_e, normals_e = geometry.sample_surface(2000, config.seed_surface + 50000)
    far_e = geometry.sample_farfield(2000, config.seed_far + 50000)
    g_surf = np.asarray(model.velocity_field(params, jnp.asarray(surf_e)))
    bc1 = np.sum(g_surf * normals_e, axis=1)
    g_far = np.asarray(model.velocity_field(params, jnp.asarray(far_e)))
    bc2 = np.linalg.norm(g_far - np.array([1.0, 0.0, 0.0]), axis=1)
    metrics["rms_pde_residual"] = float(np.sqrt(np.mean(lap**2)))
    metrics["rms_bc1_residual"] = float(np.sqrt(np.mean(bc1**2)))
    metrics["rms_bc2_residual"] = float(np.sqrt(np.mean(bc2**2)))

    # ---- pass / fail against the pre-registered criteria ----
    # For the sigma=0 pilot the sigma-relative criteria degenerate, so we use the
    # absolute small-residual bar instead (the pilot is for hyperparameter choice).
    sig_pos = sigma > 0.0
    metrics["noise_free_pilot"] = not sig_pos
    metrics["pass"] = {
        "rel_L2_field_le_2pct": rel_l2 <= CRIT_REL_L2_FIELD,
        "RMSE_cp_surface_le_sigma": (metrics["RMSE_cp_surface"] <= sigma) if sig_pos
        else (metrics["RMSE_cp_surface"] <= RESIDUAL_ORDERS_BELOW),
        "sigma_hat_holdout_in_band": (SIGMA_HAT_LO * sigma <= sigma_hat_ho <= SIGMA_HAT_HI * sigma)
        if sig_pos else (sigma_hat_ho <= RESIDUAL_ORDERS_BELOW),
        "pde_residual_small": metrics["rms_pde_residual"] <= RESIDUAL_ORDERS_BELOW,
        "bc1_residual_small": metrics["rms_bc1_residual"] <= RESIDUAL_ORDERS_BELOW,
        "bc2_residual_small": metrics["rms_bc2_residual"] <= RESIDUAL_ORDERS_BELOW,
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # ================= figures =================
    # (a) Cp(eta): Lamb vs PINN vs taps +- sigma
    tap_eta = tap_xyz[:, 0] / C.A_STAR
    ho_eta = ho_xyz[:, 0] / C.A_STAR
    fig = plt.figure(figsize=(8, 5))
    plt.plot(eta, cp_lamb_grid, "k-", lw=2, label="Lamb (analytic)")
    plt.plot(eta, cp_w_grid, "r--", lw=2, label="PINN")
    plt.errorbar(tap_eta, tap_cp, yerr=sigma, fmt="o", ms=4, color="C0",
                 capsize=2, label="taps (train) $\\pm\\sigma$")
    plt.errorbar(ho_eta, ho_cp, yerr=sigma, fmt="s", ms=4, color="C2",
                 capsize=2, label="taps (holdout)")
    plt.xlabel(r"$\eta = x/a$"); plt.ylabel(r"$C_p$")
    plt.title("Surface pressure coefficient"); plt.legend(); plt.grid(alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "cp_eta.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (b) error map ||grad phi_w - grad phi_lamb|| on the meridian plane
    nx, nz = 200, 160
    xg = np.linspace(-C.R_INF, C.R_INF, nx)
    zg = np.linspace(0.0, C.R_INF, nz)
    XX, ZZ = np.meshgrid(xg, zg)
    P = np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()], axis=1)
    mask = geometry.is_outside(P)
    err = np.full(XX.size, np.nan)
    gw = np.asarray(model.velocity_field(params, jnp.asarray(P[mask])))
    gl = np.asarray(lamb.grad_phi_batch(jnp.asarray(P[mask])))
    err[mask] = np.linalg.norm(gw - gl, axis=1)
    fig = plt.figure(figsize=(9, 5))
    pc = plt.pcolormesh(XX, ZZ, err.reshape(XX.shape), shading="auto", cmap="viridis")
    plt.colorbar(pc, label=r"$\|\nabla\phi_w - \nabla\phi_{Lamb}\|$")
    plt.xlabel("x*"); plt.ylabel("rho*"); plt.title("Velocity error (meridian plane)")
    fig.savefig(os.path.join(fig_dir, "error_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (c) direct Cp comparison on the same meridian plane
    cp_w = np.full(XX.size, np.nan)
    cp_l = np.full(XX.size, np.nan)
    cp_w[mask] = np.asarray(model.cp_at(params, jnp.asarray(P[mask])))
    cp_l[mask] = np.asarray(lamb.cp_field_batch(jnp.asarray(P[mask])))
    valid_cp = np.isfinite(cp_w) & np.isfinite(cp_l)
    cp_min = float(np.nanmin(np.concatenate([cp_w[valid_cp], cp_l[valid_cp]])))
    cp_max = float(np.nanmax(np.concatenate([cp_w[valid_cp], cp_l[valid_cp]])))
    diff_cp = np.abs(cp_w - cp_l)
    diff_max = float(np.nanmax(diff_cp))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    panels = [
        (cp_l.reshape(XX.shape), "Lamb analytic $C_p$", "viridis", cp_min, cp_max),
        (cp_w.reshape(XX.shape), "PINN $C_p$", "viridis", cp_min, cp_max),
        (diff_cp.reshape(XX.shape), r"$|C_p^{PINN} - C_p^{Lamb}|$", "magma", 0.0, diff_max if diff_max > 0 else 1.0),
    ]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.pcolormesh(XX, ZZ, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x*")
        ax.set_ylabel("rho*")
        fig.colorbar(im, ax=ax, shrink=0.88)
    fig.suptitle("Meridian-plane Cp comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "cp_meridian_compare.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (d) 1D exterior cut along the symmetry axis (rho = 0, x > a)
    x_line = np.linspace(C.A_STAR * 1.001, C.R_INF, 500)
    line_coords = np.stack([x_line, np.zeros_like(x_line), np.zeros_like(x_line)], axis=1)
    cp_w_line = np.asarray(model.cp_at(params, jnp.asarray(line_coords)))
    cp_l_line = np.asarray(lamb.cp_field_batch(jnp.asarray(line_coords)))
    fig = plt.figure(figsize=(8, 4.8))
    plt.plot(x_line, cp_l_line, "k-", lw=2, label="Lamb analytic")
    plt.plot(x_line, cp_w_line, "r--", lw=2, label="PINN")
    plt.xlabel(r"$x^*$")
    plt.ylabel(r"$C_p$")
    plt.title("Exterior-axis Cp comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.savefig(os.path.join(fig_dir, "cp_axis_compare.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (e) loss curves
    hist_path = os.path.join(run_dir, "loss_history.csv")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            steps = [int(r["step"]) for r in rows]
            fig = plt.figure(figsize=(8, 5))
            for key in rows[0]:
                if key.endswith("_loss"):
                    plt.semilogy(steps, [float(r[key]) for r in rows], label=key)
            plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3)
            plt.title("Loss components")
            fig.savefig(os.path.join(fig_dir, "loss_curves.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    # (f) residual histogram at the taps with N(0, sigma^2) overlaid
    res = np.concatenate([cp_w_train - tap_cp, cp_w_ho - ho_cp])
    fig = plt.figure(figsize=(7, 5))
    plt.hist(res, bins=20, density=True, alpha=0.6, label="tap residuals")
    if sig_pos:
        xx = np.linspace(res.min(), res.max(), 200)
        plt.plot(xx, np.exp(-xx**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi)),
                 "k-", label=r"$N(0,\sigma^2)$")
    plt.xlabel(r"$C_p^w - C_p^{med}$"); plt.ylabel("density"); plt.legend()
    plt.title(f"Tap residuals vs sigma={sigma:.4f}")
    fig.savefig(os.path.join(fig_dir, "residual_hist.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrics
