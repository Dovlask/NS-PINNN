"""Pure residual operators (.tex Sec. 'Funcao de Perda', eqs. loss-pde/bc1/bc2/data).

These are plain functions of a scalar field callable u(x,y,z) and of velocity
vectors g = grad u.  models.py uses EXACTLY these in training, and
tests/test_autodiff.py validates them against finite differences on a known
harmonic function -- so the autodiff path that runs on the GPU is the tested one.

All quantities are non-dimensional (L = 2b):  Cp = 1 - ||grad* phi*||^2.
"""
import jax
import jax.numpy as jnp

X_HAT = jnp.array([1.0, 0.0, 0.0])  # free-stream direction (BC-2)


def gradient(u, x, y, z):
    """grad of a scalar field u(x,y,z) at one point, by autodiff. Returns (3,)."""
    gx, gy, gz = jax.grad(u, argnums=(0, 1, 2))(x, y, z)
    return jnp.array([gx, gy, gz])


def laplacian(u, x, y, z):
    """Laplacian u_xx + u_yy + u_zz at one point (.tex eq.(Laplace)), by autodiff.
    Trace of the 3x3 Hessian -- exact for the code as written (CLAUDE.md Sec. 4)."""
    h = jax.hessian(u, argnums=(0, 1, 2))(x, y, z)
    return h[0][0] + h[1][1] + h[2][2]


def cp_from_grad(g):
    """Cp = 1 - ||g||^2  (.tex eq.(Cp-PINN)), g = grad* phi*."""
    return 1.0 - jnp.dot(g, g)


def bc1_residual(g, n):
    """Impermeability residual g . n_hat  (.tex BC-1, should vanish on surface)."""
    return jnp.dot(g, n)


def bc2_residual(g):
    """Far-field residual g - x_hat  (.tex BC-2). Returns (3,)."""
    return g - X_HAT
