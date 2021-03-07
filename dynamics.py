"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import jax
import jax.numpy as jnp

# System constants
g_acc = 9.81    # gravitational acceleration
β = (0.1, 1.)   # drag coefficients


def prior(q, dq, g_acc=g_acc):
    """TODO: docstring."""
    nq = 3
    sinϕ, cosϕ = jnp.sin(q[2]), jnp.cos(q[2])
    H = jnp.eye(nq)
    C = jnp.zeros((nq, nq))
    g = jnp.array([0., g_acc, 0.])
    # R = jnp.array([
    #     [cosϕ, -sinϕ, 0],
    #     [sinϕ,  cosϕ, 0],
    #     [0,    0,     1],
    # ])
    B = jnp.array([
        [-sinϕ, 0, cosϕ],
        [cosϕ,  0, sinϕ],
        [0,     1,    0],
    ])
    return H, C, g, B


def plant(q, dq, u, f_ext, prior=prior):
    """TODO: docstring."""
    H, C, g, B = prior(q, dq)
    ddq = jax.scipy.linalg.solve(H, f_ext + B@u - C@dq - g, sym_pos=True)
    return ddq


def disturbance(q, dq, w, β=β):
    """TODO: docstring."""
    β = jnp.asarray(β)
    ϕ, dx, dy = q[2], dq[0], dq[1]
    sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)
    R = jnp.array([
        [cosϕ, -sinϕ],
        [sinϕ,  cosϕ]
    ])
    v = R.T @ jnp.array([dx - w, dy])
    f_ext = - jnp.array([*(R @ (β * v * jnp.abs(v))), 0.])
    return f_ext
