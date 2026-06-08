"""Sampling of the exterior domain Omega* = {surface <= point <= sphere r*=10}.

Pure NumPy, seeded (CLAUDE.md Sec. 2.2 / Sec. 6): collocation, surface and
far-field pools are generated ONCE per run from explicit seeds. The surface pool
is UNIFORM IN AREA (rejection with the exact area element), verified by the
Monte-Carlo area test against S = 2 pi b*^2 (1 + (a*/(b* e)) arcsin e).

Geometry (.tex Sec. 1):  f(x) = x^2/a*^2 + (y^2+z^2)/b*^2 - 1,  outward normal
n_hat = grad f / ||grad f||,  grad f = (2x/a*^2, 2y/b*^2, 2z/b*^2).
"""
import numpy as np

from constants import A_STAR, B_STAR, R_INF, surface_area_star

_A2 = A_STAR**2
_B2 = B_STAR**2


def implicit(points):
    """f(x) for a batch (N,3):  >0 outside the spheroid, <0 inside, =0 on it."""
    p = np.atleast_2d(points)
    return p[:, 0] ** 2 / _A2 + (p[:, 1] ** 2 + p[:, 2] ** 2) / _B2 - 1.0


def is_outside(points):
    """Boolean mask: strictly outside the spheroid AND inside the far sphere."""
    p = np.atleast_2d(points)
    inside_far = np.sum(p**2, axis=1) <= R_INF**2
    return (implicit(p) > 0.0) & inside_far


def surface_normals(points):
    """Outward unit normals at surface points (N,3) -> (N,3), .tex BC-1."""
    p = np.atleast_2d(points)
    g = np.stack([2 * p[:, 0] / _A2, 2 * p[:, 1] / _B2, 2 * p[:, 2] / _B2], axis=1)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def _area_element(u):
    """|r_u x r_v| for r(u,v) = (a cos u, b sin u cos v, b sin u sin v):
        b sin u sqrt(b^2 cos^2 u + a^2 sin^2 u).   (derivation in README.md)"""
    return B_STAR * np.sin(u) * np.sqrt(_B2 * np.cos(u) ** 2 + _A2 * np.sin(u) ** 2)


def sample_surface(n, seed):
    """n points UNIFORM IN AREA on the spheroid surface (rejection sampling).
    Returns (coords (n,3), normals (n,3))."""
    rng = np.random.default_rng(seed)
    w_max = A_STAR * B_STAR * 1.0001  # max of _area_element over u (at u = pi/2)
    coords = []
    while len(coords) < n:
        m = 2 * (n - len(coords)) + 64
        u = rng.uniform(0.0, np.pi, m)
        v = rng.uniform(0.0, 2.0 * np.pi, m)
        keep = rng.uniform(0.0, w_max, m) < _area_element(u)
        u, v = u[keep], v[keep]
        x = A_STAR * np.cos(u)
        y = B_STAR * np.sin(u) * np.cos(v)
        z = B_STAR * np.sin(u) * np.sin(v)
        coords.extend(np.stack([x, y, z], axis=1).tolist())
    coords = np.array(coords[:n])
    return coords, surface_normals(coords)


def sample_farfield(n, seed):
    """n points uniform on the far sphere ||x*|| = R_INF (.tex BC-2)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return R_INF * d


def sample_collocation(n, seed, shell_thickness=0.5):
    """n collocation points in Omega* (.tex eq.(Laplace) domain).

    CLAUDE.md Sec. 6: ~50% uniform in the exterior volume, ~50% in a thin shell
    near the surface (and nose), to resolve the curvature there. Interior points
    are rejected.  Returns (n,3)."""
    rng = np.random.default_rng(seed)
    n_shell = n // 2
    n_vol = n - n_shell

    # (a) uniform in the exterior ball of radius R_INF (rejection on the body)
    vol = []
    while len(vol) < n_vol:
        m = 3 * (n_vol - len(vol)) + 64
        p = rng.uniform(-R_INF, R_INF, size=(m, 3))
        keep = is_outside(p)
        vol.extend(p[keep].tolist())
    vol = np.array(vol[:n_vol])

    # (b) thin shell: surface points pushed outward by d ~ U(0, shell_thickness)
    surf, nrm = sample_surface(n_shell, seed + 1)
    d = rng.uniform(0.0, shell_thickness, size=(n_shell, 1))
    shell = surf + d * nrm
    shell = shell[is_outside(shell)]
    # top up the few rejected (pushed past r*=10 near the tail) with volume pts
    while len(shell) < n_shell:
        s, nn = sample_surface(n_shell - len(shell), seed + 2 + len(shell))
        dd = rng.uniform(0.0, shell_thickness, size=(len(s), 1))
        cand = s + dd * nn
        shell = np.vstack([shell, cand[is_outside(cand)]])
    shell = shell[:n_shell]

    return np.vstack([vol, shell])


def sample_field_eval(n, seed):
    """n exterior points for the rel_L2_field metric (evaluate.py).
    Generated once, with its OWN seed, NEVER used in training."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        m = 3 * (n - len(pts)) + 64
        p = rng.uniform(-R_INF, R_INF, size=(m, 3))
        keep = is_outside(p)
        pts.extend(p[keep].tolist())
    return np.array(pts[:n])


def surface_area_monte_carlo(n, seed):
    """Monte-Carlo estimate of the surface area (test target, < 0.5% error).
    Uniform (u,v) average of the area element times the (u,v) domain area."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, np.pi, n)
    return (np.pi * 2.0 * np.pi) * np.mean(_area_element(u))


if __name__ == "__main__":
    s_mc = surface_area_monte_carlo(2_000_000, 0)
    s_an = surface_area_star()
    print(f"S analytic = {s_an:.6f}, S monte-carlo = {s_mc:.6f}, "
          f"rel err = {abs(s_mc - s_an) / s_an:.2e}")
