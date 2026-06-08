"""Autodiff tests (CLAUDE.md Sec. 6): the network Laplacian via autodiff vs an
independent finite-difference oracle.  Catches the input-normalization chain-rule
bug (a missing 1/r_inf jacobian would leave the Laplacian wrong by a constant
factor while the loss still converges).  Requires JAX."""
import numpy as np
import jax
import jax.numpy as jnp

import losses as L
from jaxpi.archs import Mlp
from constants import R_INF


def _fd_laplacian(u, x, y, z, h=1e-4):
    return (
        (u(x + h, y, z) + u(x - h, y, z) - 2 * u(x, y, z)) / h**2
        + (u(x, y + h, z) + u(x, y - h, z) - 2 * u(x, y, z)) / h**2
        + (u(x, y, z + h) + u(x, y, z - h) - 2 * u(x, y, z)) / h**2
    )


def test_laplacian_harmonic_is_zero():
    # phi = x^2 - z^2 is harmonic (lap = 2 - 2 = 0)
    u = lambda x, y, z: x**2 - z**2
    for p in [(0.3, -1.1, 0.7), (2.0, 0.5, -0.4), (-1.0, 0.2, 1.3)]:
        lap = float(L.laplacian(u, *p))
        assert abs(lap) < 1e-6


def test_gradient_matches_fd():
    u = lambda x, y, z: jnp.sin(x) * y + z**3
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y, z = rng.uniform(-1, 1, 3)
        g = np.asarray(L.gradient(u, float(x), float(y), float(z)))
        gx_fd = (float(u(x + 1e-5, y, z)) - float(u(x - 1e-5, y, z))) / 2e-5
        gz_fd = (float(u(x, y, z + 1e-5)) - float(u(x, y, z - 1e-5))) / 2e-5
        assert abs(g[0] - gx_fd) < 1e-4
        assert abs(g[2] - gz_fd) < 1e-4


def test_network_laplacian_matches_fd():
    # network snapshot with the SAME input normalization used in training
    net = Mlp(num_layers=2, hidden_dim=16, out_dim=1, activation="tanh")
    params = net.init(jax.random.PRNGKey(0), jnp.ones(3))

    def u(x, y, z):
        return net.apply(params, jnp.stack([x, y, z]) / R_INF)[0]

    rng = np.random.default_rng(1)
    for _ in range(15):
        x, y, z = rng.uniform(-3, 3, 3)
        lap_ad = float(L.laplacian(u, float(x), float(y), float(z)))
        lap_fd = _fd_laplacian(lambda a, b, c: float(u(a, b, c)), float(x), float(y), float(z), h=1e-3)
        assert abs(lap_ad - lap_fd) < 1e-3
