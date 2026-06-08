"""Analytic Lamb (1932) solution for potential flow past a prolate spheroid
(a/b = 4, alpha = 0), in NON-DIMENSIONAL form (L = 2b).

INTEGRITY (CLAUDE.md Sec. 2.4): this module is the analytic ground truth. It is
imported ONLY by evaluate.py and tests/. It is NEVER imported by train.py or by
the loss -- no leakage into training.

Autodiff is the reference for derivatives (CLAUDE.md Sec. 4): grad phi is taken
with jax.grad of the code written here. Finite differences appear ONLY in the
tests, as an independent oracle.

Equation map (README.md):
    phi(xi,eta)  -> .tex eq.(phi),  Lamb 1932 Art.103 p.139
    Q1, Q1'      -> .tex eq.(Q1),(Q1'),  Legendre 2nd kind
    inverse map  -> .tex Sec. 2 (xi,eta) <-> (x, rho)
    cp_field     -> .tex eq.(Cp-PINN):  Cp = 1 - ||grad* phi*||^2

Implementation note: phi is written directly in terms of the focal distances
r1 = |x - c x_hat|, r2 = |x + c x_hat| (no rho = sqrt(y^2+z^2)).  This keeps phi
smooth for autodiff everywhere in the exterior (the only singular points are the
two foci x = +-c, y = z = 0, which lie strictly inside the body).
"""
import jax
import jax.numpy as jnp

from constants import C_STAR, XI0, K1, E  # shape-only constants (a/b = 4)

_Q1P_XI0 = 0.5 * jnp.log((XI0 + 1.0) / (XI0 - 1.0)) - XI0 / (XI0**2 - 1.0)


def Q1(xi):
    """Legendre function of the second kind, .tex eq.(Q1):
    Q1(xi) = (xi/2) ln((xi+1)/(xi-1)) - 1,   xi > 1."""
    return 0.5 * xi * jnp.log((xi + 1.0) / (xi - 1.0)) - 1.0


def Q1_prime(xi):
    """dQ1/dxi, .tex eq.(Q1'):  (1/2) ln((xi+1)/(xi-1)) - xi/(xi^2-1)."""
    return 0.5 * jnp.log((xi + 1.0) / (xi - 1.0)) - xi / (xi**2 - 1.0)


def cart_to_spheroidal(x, y, z):
    """Inverse map (x,y,z) -> (xi, eta), .tex Sec. 2.
    r1 = sqrt((x-c)^2 + rho^2),  r2 = sqrt((x+c)^2 + rho^2),
    xi = (r1+r2)/(2c),  eta = (r2-r1)/(2c).   rho^2 = y^2 + z^2."""
    rho2 = y**2 + z**2
    r1 = jnp.sqrt((x - C_STAR) ** 2 + rho2)
    r2 = jnp.sqrt((x + C_STAR) ** 2 + rho2)
    xi = (r1 + r2) / (2.0 * C_STAR)
    eta = (r2 - r1) / (2.0 * C_STAR)
    return xi, eta


def spheroidal_to_cart(xi, eta, az=0.0):
    """Forward map (xi, eta, azimuth) -> (x, y, z), .tex Sec. 2.
    x = c xi eta,  rho = c sqrt((xi^2-1)(1-eta^2)),  y = rho cos az, z = rho sin az."""
    x = C_STAR * xi * eta
    rho = C_STAR * jnp.sqrt(jnp.clip((xi**2 - 1.0) * (1.0 - eta**2), 0.0, None))
    return x, rho * jnp.cos(az), rho * jnp.sin(az)


def phi(x, y, z):
    """Non-dimensional velocity potential phi*(x,y,z), .tex eq.(phi):
        phi* = c* eta [ xi - Q1(xi)/Q1'(xi0) ].
    Scalar in, scalar out (autodiff-friendly; smooth on the exterior)."""
    xi, eta = cart_to_spheroidal(x, y, z)
    return C_STAR * eta * (xi - Q1(xi) / _Q1P_XI0)


# grad phi*  by autodiff (the reference velocity field for evaluation).
_grad_phi_scalar = jax.grad(phi, argnums=(0, 1, 2))


def grad_phi(x, y, z):
    """Velocity u* = grad* phi* at a single point, by autodiff. Returns (3,)."""
    gx, gy, gz = _grad_phi_scalar(x, y, z)
    return jnp.array([gx, gy, gz])


def grad_phi_batch(coords):
    """Vectorized velocity field. coords: (N,3) -> (N,3)."""
    return jax.vmap(grad_phi, in_axes=(0, 0, 0))(coords[:, 0], coords[:, 1], coords[:, 2])


def cp_field(x, y, z):
    """Cp from the analytic field, .tex eq.(Cp-PINN): Cp = 1 - ||grad* phi*||^2.
    On the surface this reduces to constants.cp_surface(eta)."""
    g = grad_phi(x, y, z)
    return 1.0 - jnp.dot(g, g)


def cp_field_batch(coords):
    return jax.vmap(cp_field, in_axes=(0, 0, 0))(coords[:, 0], coords[:, 1], coords[:, 2])
