"""Tests for the analytic Lamb solution (CLAUDE.md Sec. 4).

Finite differences are used here ONLY as an independent oracle for the autodiff
gradient and for the Laplacian.  Requires JAX (run on the GPU machine)."""
import numpy as np
import jax.numpy as jnp

import lamb
import constants as C


def test_reference_constants():
    assert abs(C.E - C.REF["e"]) < 1e-3
    assert abs(C.ALPHA0 - C.REF["alpha0"]) < 1e-3
    assert abs(C.K1 - C.REF["k1"]) < 1e-3
    assert abs(float(C.cp_surface(0.0)) - C.REF["cp0"]) < 1e-3
    assert abs((1.0 + C.K1) - C.REF["ue_max_over_uinf"]) < 1e-3


def test_inverse_map_round_trip():
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(5000):
        xi = 1.0 + 10 ** rng.uniform(-6, 1.2)
        eta = rng.uniform(-1, 1)
        az = rng.uniform(0, 2 * np.pi)
        cases.append((xi, eta, az))
    # include the axis (eta = +-1) and a near-surface point
    cases += [(C.XI0, 1.0, 0.0), (C.XI0, -1.0, 0.0), (3.0, 1.0, 0.0),
              (C.XI0, 0.0, 0.0), (1.0000001, 0.5, 1.0)]
    max_err = 0.0
    for xi, eta, az in cases:
        x, y, z = lamb.spheroidal_to_cart(xi, eta, az)
        xi2, eta2 = lamb.cart_to_spheroidal(x, y, z)
        max_err = max(max_err, abs(float(xi2) - xi), abs(float(eta2) - eta))
    assert max_err < 1e-10


def _exterior_points(n, rng):
    pts = []
    while len(pts) < n:
        p = rng.uniform(-6, 6, size=3)
        rr = p[0] ** 2 / C.A_STAR**2 + (p[1] ** 2 + p[2] ** 2) / C.B_STAR**2
        if rr > 1.5 and np.hypot(p[1], p[2]) > 0.4 and np.linalg.norm(p) < 8.0:
            pts.append(p)
    return np.array(pts)


def test_laplacian_zero_with_O_h2_rate():
    rng = np.random.default_rng(1)
    pts = _exterior_points(300, rng)

    def lap(h):
        vals = []
        for p in pts:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            s = (
                float(lamb.phi(x + h, y, z) + lamb.phi(x - h, y, z) - 2 * lamb.phi(x, y, z)) / h**2
                + float(lamb.phi(x, y + h, z) + lamb.phi(x, y - h, z) - 2 * lamb.phi(x, y, z)) / h**2
                + float(lamb.phi(x, y, z + h) + lamb.phi(x, y, z - h) - 2 * lamb.phi(x, y, z)) / h**2
            )
            vals.append(s)
        return np.sqrt(np.mean(np.array(vals) ** 2))

    r1, r2, r3 = lap(1e-2), lap(5e-3), lap(2.5e-3)
    assert r3 < 1e-6                       # Laplacian ~ 0
    assert 3.5 < r1 / r2 < 4.5             # O(h^2)
    assert 3.5 < r2 / r3 < 4.5


def test_bc1_on_surface():
    rng = np.random.default_rng(2)
    vals = []
    for _ in range(1000):
        eta = rng.uniform(-0.97, 0.97)
        az = rng.uniform(0, 2 * np.pi)
        x, y, z = lamb.spheroidal_to_cart(C.XI0, eta, az)
        x, y, z = float(x), float(y), float(z)
        n = np.array([2 * x / C.A_STAR**2, 2 * y / C.B_STAR**2, 2 * z / C.B_STAR**2])
        n = n / np.linalg.norm(n)
        g = np.asarray(lamb.grad_phi(x, y, z))
        vals.append(abs(float(np.dot(g, n))))
    assert np.sqrt(np.mean(np.array(vals) ** 2)) < 1e-5


def test_farfield_decay_exponent():
    rng = np.random.default_rng(3)
    rs = np.array([3.0, 5.0, 8.0, 12.0, 20.0]) * (C.A_STAR / 2)
    defs = []
    for R in rs:
        v = []
        for _ in range(150):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            p = R * d
            g = np.asarray(lamb.grad_phi(float(p[0]), float(p[1]), float(p[2])))
            v.append(np.linalg.norm(g - np.array([1.0, 0, 0])))
        defs.append(np.mean(v))
    slope = np.polyfit(np.log(rs), np.log(defs), 1)[0]
    assert abs(slope + 3.0) < 0.25         # O((a/r)^3)
