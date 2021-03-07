"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import pickle
from math import pi, inf
import os
import argparse
import time
import numpy as np
from itertools import product
from tqdm.auto import tqdm


def enumerated_product(*args):
    """TODO: docstring."""
    yield from zip(
        product(*(range(len(x)) for x in args)),
        product(*args)
    )


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('seed', help='seed for pseudo-random number generation',
                    type=int)
parser.add_argument('M', help='number of trajectories to sub-sample',
                    type=int)
parser.add_argument('--use_x64', help='use 64-bit precision',
                    action='store_true')
args = parser.parse_args()

# Set precision
if args.use_x64:
    os.environ['JAX_ENABLE_X64'] = 'True'

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
from jax.experimental.ode import odeint                           # noqa: E402
from utils import random_ragged_spline, spline, params_to_posdef  # noqa: E402
from dynamics import prior, plant, disturbance                    # noqa: E402


# Initialize PRNG key (with offset from original seed to make sure we do not
# sample the same reference trajectories in the training set)
key = jax.random.PRNGKey(args.seed + 20)

hparams = {
    'seed':        args.seed,     #
    'use_x64':     args.use_x64,  #
    'num_subtraj': args.M,        # number of trajectories sub-sampled

    'w_min': 0.,     # minimum wind velocity in inertial `x`-direction
    'w_max': 10.,    # maximum wind velocity in inertial `x`-direction
    'a': 5.,         # shape parameter `a` for beta distribution
    'b': 7.,         # shape parameter `b` for beta distribution

    # Reference trajectory generation
    'T':            10.,                  # time horizon for each reference
    'dt':           1e-2,                 # numerical integration time step
    'num_refs':     200,                  # reference trajectories to generate
    'num_knots':    6,                    # knot points per reference spline
    'poly_orders':  (9, 9, 6),            # spline orders for each DOF
    'deriv_orders': (4, 4, 2),            # smoothness objective for each DOF
    'min_step':     (-2., -2., -pi/6),    #
    'max_step':     (2., 2., pi/6),       #
    'min_ref':      (-inf, -inf, -pi/3),  #
    'max_ref':      (inf, inf, pi/3),     #
}


if __name__ == "__main__":
    print('Testing ... ', flush=True)
    start = time.time()

    # Generate reference trajectories
    key, *subkeys = jax.random.split(key, 1 + hparams['num_refs'])
    subkeys = jnp.vstack(subkeys)
    in_axes = (0, None, None, None, None, None, None, None, None)
    min_ref = jnp.asarray(hparams['min_ref'])
    max_ref = jnp.asarray(hparams['max_ref'])
    t_knots, knots, coefs = jax.vmap(random_ragged_spline, in_axes)(
        subkeys,
        hparams['T'],
        hparams['num_knots'],
        hparams['poly_orders'],
        hparams['deriv_orders'],
        jnp.asarray(hparams['min_step']),
        jnp.asarray(hparams['max_step']),
        0.7*min_ref,
        0.7*max_ref,
    )
    r_knots = jnp.dstack(knots)
    num_dof = 3

    # Sampled-time simulator
    @jax.jit
    @jax.partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
    def simulate(ts, w, t_knots, coefs, params,
                 min_ref=min_ref, max_ref=max_ref,
                 plant=plant, prior=prior, disturbance=disturbance):
        """TODO: docstring."""
        # Construct spline reference trajectory
        def reference(t):
            r = jnp.array([spline(t, t_knots, c) for c in coefs])
            r = jnp.clip(r, min_ref, max_ref)
            return r

        # Required derivatives of the reference trajectory
        def ref_derivatives(t):
            ref_vel = jax.jacfwd(reference)
            ref_acc = jax.jacfwd(ref_vel)
            r = reference(t)
            dr = ref_vel(t)
            ddr = ref_acc(t)
            return r, dr, ddr

        # Adaptation law
        def adaptation_law(q, dq, r, dr, params=params):
            # Regressor features
            y = jnp.concatenate((q, dq))
            for W, b in zip(params['W'], params['b']):
                y = jnp.tanh(W@y + b)

            # Auxiliary signals
            Λ, P = params['Λ'], params['P']
            e, de = q - r, dq - dr
            s = de + Λ@e

            dA = P @ jnp.outer(s, y)
            return dA, y

        # Controller
        def controller(q, dq, r, dr, ddr, f_hat, params=params):
            # Auxiliary signals
            Λ, K = params['Λ'], params['K']
            e, de = q - r, dq - dr
            s = de + Λ@e
            v, dv = dr - Λ@e, ddr - Λ@de

            # Control input and adaptation law
            H, C, g, B = prior(q, dq)
            τ = H@dv + C@v + g - f_hat - K@s
            u = jnp.linalg.solve(B, τ)
            return u, τ

        # Closed-loop ODE for `x = (q, dq)`, with a zero-order hold on
        # the controller
        def ode(x, t, u, w=w):
            q, dq = x
            f_ext = disturbance(q, dq, w)
            ddq = plant(q, dq, u, f_ext)
            dx = (dq, ddq)
            return dx

        # Simulation loop
        def loop(carry, input_slice):
            t_prev, q_prev, dq_prev, u_prev, A_prev, dA_prev = carry
            t = input_slice
            qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]),
                             u_prev)
            q, dq = qs[-1], dqs[-1]
            r, dr, ddr = ref_derivatives(t)

            # Integrate adaptation law via trapezoidal rule
            dA, y = adaptation_law(q, dq, r, dr)
            A = A_prev + (t - t_prev)*(dA_prev + dA)/2

            # Compute force estimate and control input
            f_hat = A @ y
            u, τ = controller(q, dq, r, dr, ddr, f_hat)

            carry = (t, q, dq, u, A, dA)
            output_slice = (q, dq, u, τ, r, dr)
            return carry, output_slice

        # Initial conditions
        t0 = ts[0]
        r0, dr0, ddr0 = ref_derivatives(t0)
        q0, dq0 = r0, dr0
        dA0, y0 = adaptation_law(q0, dq0, r0, dr0)
        A0 = jnp.zeros((q0.size, y0.size))
        f0 = A0 @ y0
        u0, τ0 = controller(q0, dq0, r0, dr0, ddr0, f0)

        # Run simulation loop
        carry = (t0, q0, dq0, u0, A0, dA0)
        carry, output = jax.lax.scan(loop, carry, ts[1:])
        q, dq, u, τ, r, dr = output

        # Prepend initial conditions
        q = jnp.vstack((q0, q))
        dq = jnp.vstack((dq0, dq))
        u = jnp.vstack((u0, u))
        τ = jnp.vstack((τ0, τ))
        r = jnp.vstack((r0, r))
        dr = jnp.vstack((dr0, dr))

        return q, dq, u, τ, r, dr

    # Sample wind velocities from the test distribution
    w_min = 0.   # minimum wind velocity in inertial `x`-direction
    w_max = 10.  # maximum wind velocity in inertial `x`-direction
    a = 5.       # shape parameter `a` for beta distribution
    b = 7.       # shape parameter `b` for beta distribution
    key, subkey = jax.random.split(key, 2)
    w = w_min + (w_max - w_min)*jax.random.beta(subkey, a, b,
                                                (hparams['num_refs'],))

    # Simulate tracking for each `w`
    T, dt = hparams['T'], hparams['dt']
    ts = jnp.arange(0, T + dt, dt)  # same times for each trajectory

    # Try out different gains
    test_results = {
        'w': w, 'w_min': w_min, 'w_max': w_max,
        'beta_params': (a, b),
        'gains': {
            'Λ': (1.,),
            'K': (1., 10.),
            'P': (1., 10.),
        }
    }
    grid_shape = (len(test_results['gains']['Λ']),
                  len(test_results['gains']['K']),
                  len(test_results['gains']['P']))

    # Our method with meta-learned gains
    print('  ours (meta) ...', flush=True)
    filename = os.path.join('train_results', 'ours',
                            'seed={}_M={}.pkl'.format(hparams['seed'],
                                                      hparams['num_subtraj']))
    with open(filename, 'rb') as file:
        train_results = pickle.load(file)
    params = {
        'W': train_results['model']['W'],
        'b': train_results['model']['b'],
        'Λ': params_to_posdef(train_results['controller']['Λ']),
        'K': params_to_posdef(train_results['controller']['K']),
        'P': params_to_posdef(train_results['controller']['P']),
    }
    q, dq, u, τ, r, dr = simulate(ts, w, t_knots, coefs, params)
    e = np.concatenate((q - r, dq - dr), axis=-1)
    rms_e = np.sqrt(np.mean(np.sum(e**2, axis=-1), axis=-1))
    rms_u = np.sqrt(np.mean(np.sum(u**2, axis=-1), axis=-1))
    test_results['ours_meta'] = {
        'params':    params,
        'rms_error': rms_e,
        'rms_ctrl':  rms_u,
    }

    for method in ('pid', 'lstsq', 'ours'):
        test_results[method] = np.empty(grid_shape, dtype=object)
        print('  {} ...'.format(method), flush=True)
        if method == 'pid':
            params = {
                'W': [jnp.zeros((1, 2*num_dof)), ],
                'b': [jnp.inf * jnp.ones((1,)), ],
            }
        else:
            filename = os.path.join(
                'train_results', method,
                'seed={}_M={}.pkl'.format(hparams['seed'],
                                          hparams['num_subtraj'])
            )
            with open(filename, 'rb') as file:
                train_results = pickle.load(file)
            params = {
                'W': train_results['model']['W'],
                'b': train_results['model']['b'],
            }

        for (i, j, l), (λ, k, p) in tqdm(enumerated_product(
            test_results['gains']['Λ'],
            test_results['gains']['K'],
            test_results['gains']['P']), total=np.prod(grid_shape)
        ):
            params['Λ'] = λ * jnp.eye(num_dof)
            params['K'] = k * jnp.eye(num_dof)
            params['P'] = p * jnp.eye(num_dof)
            q, dq, u, τ, r, dr = simulate(ts, w, t_knots, coefs, params)
            e = np.concatenate((q - r, dq - dr), axis=-1)
            rms_e = np.sqrt(np.mean(np.sum(e**2, axis=-1), axis=-1))
            rms_u = np.sqrt(np.mean(np.sum(u**2, axis=-1), axis=-1))
            test_results[method][i, j, l] = {
                'params':    params,
                'rms_error': rms_e,
                'rms_ctrl':  rms_u,
            }

    # Save
    output_filename = os.path.join(
        'test_results', "seed={:d}_M={:d}.pkl".format(hparams['seed'],
                                                      hparams['num_subtraj'])
    )
    with open(output_filename, 'wb') as file:
        pickle.dump(test_results, file)

    end = time.time()
    print('done! ({:.2f} s)'.format(end - start))
