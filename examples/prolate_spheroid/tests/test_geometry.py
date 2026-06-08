"""Geometry / sampling tests (NumPy only -- runnable without JAX)."""
import numpy as np

import geometry as g
import constants as C


def test_surface_area_monte_carlo():
    s_an = C.surface_area_star()
    s_mc = g.surface_area_monte_carlo(2_000_000, seed=0)
    assert abs(s_mc - s_an) / s_an < 0.005   # < 0.5% (CLAUDE.md Sec. 6)


def test_surface_points_on_surface():
    coords, normals = g.sample_surface(4000, seed=123)
    assert np.max(np.abs(g.implicit(coords))) < 1e-10
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-10)


def test_collocation_is_exterior():
    col = g.sample_collocation(20000, seed=7)
    assert g.is_outside(col).all()
    assert np.max(np.linalg.norm(col, axis=1)) <= C.R_INF + 1e-9
    assert col.shape == (20000, 3)


def test_farfield_on_sphere():
    ff = g.sample_farfield(2000, seed=9)
    assert np.allclose(np.linalg.norm(ff, axis=1), C.R_INF, atol=1e-9)


def test_field_eval_exterior_and_seeded():
    a = g.sample_field_eval(5000, seed=C.SEED_EVAL_FIELD)
    b = g.sample_field_eval(5000, seed=C.SEED_EVAL_FIELD)
    assert g.is_outside(a).all()
    assert np.array_equal(a, b)   # deterministic from the seed
