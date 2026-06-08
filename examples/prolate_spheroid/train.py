"""Training loop for the prolate-spheroid PINN.

Mirrors examples/ns_steady_cylinder/train.py but logs LOCALLY to
results/<run_name>/ (CSV + JSON) instead of W&B (CLAUDE.md Sec. 3: wandb not
mandatory).  The frozen taps (cp_med) are the ONLY data; Lamb is never imported.

Weighting (CLAUDE.md Sec. 6):
  * scheme in {grad_norm, ntk}: a single normalization at step 0, then either
    kept fixed (config.weighting.adaptive = False -> the "1/residual" baseline)
    or refreshed every update_every_steps (adaptive = True -> paper technique).
Optional final L-BFGS refinement (config.training.use_lbfgs) is isolated and
wrapped: the Adam checkpoint is saved first, so a failure there cannot corrupt it.
"""
import os
import csv
import json
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

import ml_collections

from jaxpi.samplers import SpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
import geometry


def load_taps(split_seed, train_frac=0.8):
    """Load the frozen taps; split 80/20. Training uses cp_med (NOISY) only;
    cp_true is ignored here (it is Lamb -> evaluate.py only)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cp_synthetic.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    cp_med = np.array([float(r["cp_med"]) for r in rows])
    n = len(rows)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_tr = int(round(train_frac * n))
    tr, ho = perm[:n_tr], perm[n_tr:]
    return (xyz[tr], cp_med[tr]), (xyz[ho], cp_med[ho])


def _unreplicate(state):
    return tree_map(lambda x: x[0], state)


def lbfgs_refine(model, state, collocation, max_iter):
    """OPTIONAL full-batch L-BFGS polish (CLAUDE.md Sec. 6). Isolated and untested
    on this hardware -- default OFF. Operates on a single device."""
    import optax

    params = _unreplicate(state).params
    weights = _unreplicate(state).weights
    full_batch = jnp.asarray(collocation)

    def loss_fn(p):
        return model.loss(p, weights, full_batch)

    opt = optax.lbfgs()
    opt_state = opt.init(params)
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def step(p, s):
        v, g = value_and_grad(p, state=s)
        updates, s = opt.update(g, s, p, value=v, grad=g, value_fn=loss_fn)
        p = optax.apply_updates(p, updates)
        return p, s, v

    for i in range(max_iter):
        params, opt_state, v = step(params, opt_state)
    # re-replicate params back into the training state
    from flax import jax_utils
    new_state = _unreplicate(state).replace(params=params)
    return jax_utils.replicate(new_state)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()
    run_dir = os.path.join(workdir, "results", config.run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(run_dir, exist_ok=True)

    # save an exact copy of the config (reproducibility, CLAUDE.md Sec. 2.7)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    # ---- data and fixed pools (seeded once per run) ----
    (tap_xyz, tap_cp), _holdout = load_taps(config.seed_split)
    surf, normals = geometry.sample_surface(config.sampling.n_surface, config.seed_surface)
    far = geometry.sample_farfield(config.sampling.n_farfield, config.seed_far)
    collocation = geometry.sample_collocation(config.sampling.n_collocation, config.seed_collocation)

    # ---- model ----
    model = models.ProlateSpheroid(config, surf, normals, far, tap_xyz, tap_cp)
    evaluator = models.ProlateSpheroidEvaluator(config, model)

    res_sampler = iter(
        SpaceSampler(jnp.asarray(collocation), config.training.batch_size_per_device,
                     rng_key=random.PRNGKey(config.seed_collocation))
    )

    # ---- step-0 weight normalization ----
    adaptive = bool(config.weighting.adaptive)
    if config.weighting.scheme in ["grad_norm", "ntk"]:
        batch0 = next(res_sampler)
        model.state = model.update_weights(model.state, batch0)

    # ---- CSV loss history ----
    hist_path = os.path.join(run_dir, "loss_history.csv")
    hist_file = open(hist_path, "w", newline="")
    hist_writer = None

    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if adaptive and config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        if jax.process_index() == 0 and step % config.logging.log_every_steps == 0:
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            batch0 = jax.device_get(tree_map(lambda x: x[0], batch))
            log_dict = evaluator(state, batch0)
            log_dict = {k: float(v) for k, v in log_dict.items()}
            log_dict["step"] = step

            if hist_writer is None:
                hist_writer = csv.DictWriter(hist_file, fieldnames=["step"] + sorted(k for k in log_dict if k != "step"))
                hist_writer.writeheader()
            hist_writer.writerow(log_dict)
            hist_file.flush()

            end_time = time.time()
            logger.log_iter(step, start_time, end_time, log_dict)
            start_time = end_time

        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                save_checkpoint(model.state, ckpt_dir, keep=config.saving.num_keep_ckpts)

    # always save the final Adam checkpoint
    save_checkpoint(model.state, ckpt_dir, keep=config.saving.num_keep_ckpts)
    hist_file.close()

    # ---- optional L-BFGS refinement (isolated; Adam ckpt already safe) ----
    if config.training.use_lbfgs:
        try:
            print("L-BFGS refinement...")
            model.state = lbfgs_refine(model, model.state, collocation, config.training.lbfgs_max_iter)
            save_checkpoint(model.state, ckpt_dir, keep=config.saving.num_keep_ckpts)
        except Exception as exc:  # keep the Adam result no matter what
            print(f"[warn] L-BFGS refinement skipped ({type(exc).__name__}: {exc}); "
                  f"Adam checkpoint retained.")

    return model
