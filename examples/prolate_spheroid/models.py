"""PINN model for the prolate-spheroid potential flow (mirrors the jaxpi
ns_steady_cylinder example: subclass of ForwardBVP, pmapped step/weights,
grad_norm / ntk weighting reused from jaxpi.models).

Input  : (x*, y*, z*) non-dimensional, rescaled by r*_inf = 10.
Output : scalar phi*_w.    Velocity g = grad* phi*_w (autodiff).

Loss terms (.tex eqs. loss-pde/bc1/bc2/data), each optional via config.physics
(this drives the A/B/C/D ablations, CLAUDE.md Sec. 8):
    pde  : mean( (lap phi)^2 )           on collocation batch
    bc1  : mean( (g . n_hat)^2 )         on fixed surface pool
    bc2  : mean( ||g - x_hat||^2 )       on fixed far-field pool (r*=10)
    data : mean( (Cp_w - Cp_med)^2 )     on fixed training taps,  Cp_w = 1 - ||g||^2

INTEGRITY: the analytic Lamb solution is NOT imported here -- only the noisy
Cp_med values loaded from the frozen dataset enter the loss (CLAUDE.md Sec. 2.4).
"""
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn

import losses as L
from constants import R_INF


class ProlateSpheroid(ForwardBVP):
    def __init__(self, config, surface_coords, surface_normals, far_coords,
                 tap_coords, tap_cp):
        super().__init__(config)

        self.r_inf = R_INF

        # fixed pools (used in full every step, like the BC sets in the example)
        self.sx, self.sy, self.sz = [jnp.asarray(surface_coords[:, i]) for i in range(3)]
        self.nx, self.ny, self.nz = [jnp.asarray(surface_normals[:, i]) for i in range(3)]
        self.fx, self.fy, self.fz = [jnp.asarray(far_coords[:, i]) for i in range(3)]
        self.tx, self.ty, self.tz = [jnp.asarray(tap_coords[:, i]) for i in range(3)]
        self.tap_cp = jnp.asarray(tap_cp)

        # which terms are active (A/B/C/D ablations)
        self.use_pde = bool(config.physics.use_pde)
        self.use_bc1 = bool(config.physics.use_bc1)
        self.use_bc2 = bool(config.physics.use_bc2)
        self.use_data = bool(config.physics.use_data)

        # batched predictors
        self.lap_pred = vmap(self.laplacian_net, (None, 0, 0, 0))
        self.vel_pred = vmap(self.velocity, (None, 0, 0, 0))
        self.cp_pred = vmap(self.cp_net, (None, 0, 0, 0))

    # ---- network and differential operators -------------------------------
    def u_net(self, params, x, y, z):
        """Scalar potential phi*_w. Input rescaled by r*_inf (autodiff carries the
        chain-rule jacobian -- CLAUDE.md Sec. 4 caveat is handled correctly)."""
        X = jnp.stack([x, y, z]) / self.r_inf
        return self.state.apply_fn(params, X)[0]

    def velocity(self, params, x, y, z):
        u = lambda a, b, c: self.u_net(params, a, b, c)
        return L.gradient(u, x, y, z)  # (3,)

    def laplacian_net(self, params, x, y, z):
        u = lambda a, b, c: self.u_net(params, a, b, c)
        return L.laplacian(u, x, y, z)

    # scalar residual nets (for the NTK weighting)
    def bc1_net(self, params, x, y, z, nx, ny, nz):
        return L.bc1_residual(self.velocity(params, x, y, z), jnp.array([nx, ny, nz]))

    def gx_net(self, params, x, y, z):
        return self.velocity(params, x, y, z)[0]

    def gy_net(self, params, x, y, z):
        return self.velocity(params, x, y, z)[1]

    def gz_net(self, params, x, y, z):
        return self.velocity(params, x, y, z)[2]

    def cp_net(self, params, x, y, z):
        return L.cp_from_grad(self.velocity(params, x, y, z))

    # ---- losses -----------------------------------------------------------
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        loss_dict = {}

        if self.use_pde:
            r = self.lap_pred(params, batch[:, 0], batch[:, 1], batch[:, 2])
            loss_dict["pde"] = jnp.mean(r**2)

        if self.use_bc1:
            gn = vmap(self.bc1_net, (None, 0, 0, 0, 0, 0, 0))(
                params, self.sx, self.sy, self.sz, self.nx, self.ny, self.nz
            )
            loss_dict["bc1"] = jnp.mean(gn**2)

        if self.use_bc2:
            g = self.vel_pred(params, self.fx, self.fy, self.fz)  # (Nf, 3)
            res = g - L.X_HAT
            loss_dict["bc2"] = jnp.mean(jnp.sum(res**2, axis=1))

        if self.use_data:
            cp = self.cp_pred(params, self.tx, self.ty, self.tz)
            loss_dict["data"] = jnp.mean((cp - self.tap_cp) ** 2)

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        ntk_dict = {}

        if self.use_pde:
            ntk_dict["pde"] = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.laplacian_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )

        if self.use_bc1:
            ntk_dict["bc1"] = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0, 0))(
                self.bc1_net, params, self.sx, self.sy, self.sz,
                self.nx, self.ny, self.nz
            )

        if self.use_bc2:
            gx = vmap(ntk_fn, (None, None, 0, 0, 0))(self.gx_net, params, self.fx, self.fy, self.fz)
            gy = vmap(ntk_fn, (None, None, 0, 0, 0))(self.gy_net, params, self.fx, self.fy, self.fz)
            gz = vmap(ntk_fn, (None, None, 0, 0, 0))(self.gz_net, params, self.fx, self.fy, self.fz)
            ntk_dict["bc2"] = jnp.concatenate([gx, gy, gz])

        if self.use_data:
            ntk_dict["data"] = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.cp_net, params, self.tx, self.ty, self.tz
            )

        return ntk_dict

    # ---- evaluation helpers (used by evaluate.py / sigma_hat) --------------
    @partial(jit, static_argnums=(0,))
    def velocity_field(self, params, coords):
        return self.vel_pred(params, coords[:, 0], coords[:, 1], coords[:, 2])

    @partial(jit, static_argnums=(0,))
    def cp_at(self, params, coords):
        return self.cp_pred(params, coords[:, 0], coords[:, 1], coords[:, 2])

    @partial(jit, static_argnums=(0,))
    def laplacian_at(self, params, coords):
        return self.lap_pred(params, coords[:, 0], coords[:, 1], coords[:, 2])


class ProlateSpheroidEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def __call__(self, state, batch):
        # logs per-component losses and weights (BaseEvaluator); no Lamb here.
        self.log_dict = super().__call__(state, batch)
        return self.log_dict
