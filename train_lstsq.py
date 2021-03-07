"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

from tqdm.auto import tqdm
import pickle
import time
import warnings
import os
import argparse

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

import jax                                  # noqa: E402
import jax.numpy as jnp                     # noqa: E402
from jax.experimental import optimizers     # noqa: E402
from utils import tree_normsq               # noqa: E402
from dynamics import prior                  # noqa: E402


# Initialize PRNG key
key = jax.random.PRNGKey(args.seed)

# Hyperparameters
hparams = {
    'seed':        args.seed,     #
    'use_x64':     args.use_x64,  #
    'num_subtraj': args.M,        # number of trajectories to sub-sample
    'num_hlayers':        2,      # number of hidden layers
    'hdim':               32,     # number of hidden units per layer
    'train_frac':         0.75,   # fraction per trajectory for training
    'ridge_frac':         0.25,   # (fraction of samples used in the ridge
                                  #  regression solution per trajectory)
    'regularizer_l2':     1e-4,   # coefficient for L2-regularization
    'regularizer_ridge':  1e-6,   # (coefficient for L2-regularization of
                                  #  least-squares solution)
    'learning_rate':      1e-2,   # step size for gradient optimization
    'num_steps':          5000,   # number of epochs
}


if __name__ == "__main__":
    # DATA PROCESSING ########################################################
    # Load raw data and arrange in samples of the form
    # `(t, x, u, t_next, x_next)` for each trajectory, where `x := (q,dq)`
    with open('training_data.pkl', 'rb') as file:
        raw = pickle.load(file)
    num_dof = raw['q'].shape[-1]       # number of degrees of freedom
    num_traj = raw['q'].shape[0]       # total number of raw trajectories
    num_samples = raw['t'].size - 1    # number of transitions per trajectory
    t = jnp.tile(raw['t'][:-1], (num_traj, 1))
    t_next = jnp.tile(raw['t'][1:], (num_traj, 1))
    x = jnp.concatenate((raw['q'][:, :-1], raw['dq'][:, :-1]), axis=-1)
    x_next = jnp.concatenate((raw['q'][:, 1:], raw['dq'][:, 1:]), axis=-1)
    u = raw['u'][:, :-1]

    data = {'t': t, 'x': x, 'u': u, 't_next': t_next, 'x_next': x_next}

    # Shuffle and sub-sample trajectories
    if hparams['num_subtraj'] > num_traj:
        warnings.warn('Cannot sub-sample {:d} trajectories! '
                      'Capping at {:d}.'.format(hparams['num_subtraj'],
                                                num_traj))
        hparams['num_subtraj'] = num_traj

    key, subkey = jax.random.split(key, 2)
    shuffled_idx = jax.random.permutation(subkey, num_traj)
    hparams['subtraj_idx'] = shuffled_idx[:hparams['num_subtraj']]
    data = jax.tree_util.tree_map(
        lambda a: jnp.take(a, hparams['subtraj_idx'], axis=0),
        data
    )

    # META-TRAIN MODEL #######################################################
    # Map over time index
    @jax.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def lstsq_coefs(params, t, x, u, t_next, x_next, prior=prior):
        """TODO: docstring."""
        num_dof = x.size // 2
        q, dq = x[:num_dof], x[num_dof:]
        dq_next = x_next[num_dof:]
        H, C, g, B = prior(q, dq)

        # Regressor
        phi = x
        for W, b in zip(params['W'], params['b']):
            phi = jnp.tanh(W@phi + b)

        # Euler integration of the dynamics from `t` to `t_next` yields
        # a linear equation of the form `A@z = b`, where `A` is the last
        # layer applied to our regressor
        dt = t_next - t
        z = dt*phi
        b = H@(dq_next - dq) + dt*(C@dq + g - B@u)
        return z, b

    # Map over trajectory index
    @jax.partial(jax.vmap, in_axes=(None, 0, None, None, 0, 0, 0, 0, 0))
    def trajectory_loss(params, key, num_ridge_samples, regularizer_ridge,
                        t, x, u, t_next, x_next):
        """TODO: docstring."""
        # Compute least-squares coefficients and shuffle them
        Z, B = lstsq_coefs(params, t, x, u, t_next, x_next)
        num_samples, num_features = Z.shape
        idx = jax.random.permutation(key, num_samples)
        Z = Z[idx]
        B = B[idx]

        # Solve for the last layer as the least-squares solution
        # on a subset of the data
        Z_ls = Z[:num_ridge_samples]
        B_ls = B[:num_ridge_samples]
        ZTZ_λI = (Z_ls.T@Z_ls).at[jnp.diag_indices(num_features)].add(
            regularizer_ridge
        )
        ZTB = Z_ls.T@B_ls
        AT = jax.scipy.linalg.solve(ZTZ_λI/num_ridge_samples,
                                    ZTB/num_ridge_samples, sym_pos=True)

        # Compute loss on ALL of the data
        loss = jnp.sum((Z@AT - B)**2)
        return loss

    @jax.partial(jax.jit, static_argnums=(3,))
    def loss(params, regularizer_l2, keys, num_ridge_samples,
             regularizer_ridge, t, x, u, t_next, x_next):
        """TODO: docstring."""
        num_traj, num_samples = t.shape
        normalizer = num_traj * num_samples
        traj_losses = trajectory_loss(params, keys, num_ridge_samples,
                                      regularizer_ridge,
                                      t, x, u, t_next, x_next)
        loss = (jnp.sum(traj_losses)
                + regularizer_l2*tree_normsq(params)) / normalizer
        return loss

    # Initialize model parameters
    num_hlayers = hparams['num_hlayers']
    hdim = hparams['hdim']
    if num_hlayers >= 1:
        shapes = [(hdim, 2*num_dof), ] + (num_hlayers-1)*[(hdim, hdim), ]
    else:
        shapes = []
    key, *subkeys = jax.random.split(key, 1 + 2*num_hlayers)
    keys_W = subkeys[:num_hlayers]
    keys_b = subkeys[num_hlayers:]
    params = {
        # hidden layer weights
        'W': [0.1*jax.random.normal(keys_W[i], shapes[i])
              for i in range(num_hlayers)],
        # hidden layer biases
        'b': [0.1*jax.random.normal(keys_b[i], (shapes[i][0],))
              for i in range(num_hlayers)],
    }

    # Shuffle samples in time along each trajectory, then split each
    # trajectory into training and validation sets
    key, *subkeys = jax.random.split(key, 1 + hparams['num_subtraj'])
    subkeys = jnp.asarray(subkeys)
    shuffled_data = jax.tree_util.tree_map(
        lambda a: jax.vmap(jax.random.permutation)(subkeys, a),
        data
    )
    num_train_samples = int(hparams['train_frac'] * num_samples)
    num_valid_samples = num_samples - num_train_samples
    train_data = jax.tree_util.tree_map(lambda a: a[:, :num_train_samples],
                                        shuffled_data)
    valid_data = jax.tree_util.tree_map(lambda a: a[:, num_train_samples:],
                                        shuffled_data)

    # Initialize gradient-based optimizer (ADAM)
    num_ridge_samples = int(hparams['ridge_frac']*num_train_samples)
    learning_rate = hparams['learning_rate']
    init_opt, update_opt, get_params = optimizers.adam(learning_rate)
    opt_state = init_opt(params)
    step_idx = 0
    best_idx = 0
    best_loss = jnp.inf
    best_params = params

    @jax.partial(jax.jit, static_argnums=(4,))
    def step(idx, opt_state, regularizer_l2, keys, num_ridge_samples,
             regularizer_ridge, batch):
        """TODO: docstring."""
        params = get_params(opt_state)
        grads = jax.grad(loss, argnums=0)(params, regularizer_l2, keys,
                                          num_ridge_samples,
                                          regularizer_ridge, **batch)
        opt_state = update_opt(idx, grads, opt_state)
        return opt_state

    # Pre-compile before training
    print('MODEL META-TRAINING: Pre-compiling ... ', end='', flush=True)
    start = time.time()
    _ = step(step_idx, opt_state, hparams['regularizer_l2'],
             subkeys, num_ridge_samples,
             hparams['regularizer_ridge'], train_data)
    _ = loss(params, 0., subkeys, num_valid_samples,
             hparams['regularizer_ridge'], **valid_data)
    end = time.time()
    print('done ({:.2f} s)! Now training ...'.format(end - start))
    start = time.time()

    # Do gradient descent
    for _ in tqdm(range(hparams['num_steps'])):
        key, *subkeys = jax.random.split(key, 1 + hparams['num_subtraj'])
        subkeys = jnp.asarray(subkeys)
        opt_state = step(step_idx, opt_state, hparams['regularizer_l2'],
                         subkeys, num_ridge_samples,
                         hparams['regularizer_ridge'], train_data)
        new_params = get_params(opt_state)
        new_loss = loss(new_params, 0., subkeys, num_valid_samples,
                        hparams['regularizer_ridge'], **valid_data)
        step_idx += 1
        if new_loss < best_loss:
            best_idx = step_idx
            best_loss = new_loss
            best_params = new_params

    # Save hyperparameters and model
    results = {
        'best_step_idx': best_idx,
        'hparams': hparams,
        'model':   best_params
    }
    output_name = "seed={:d}_M={:d}".format(
        hparams['seed'], hparams['num_subtraj']
    )
    output_path = os.path.join('train_results', 'lstsq', output_name + '.pkl')
    with open(output_path, 'wb') as file:
        pickle.dump(results, file)

    end = time.time()
    print('done ({:.2f} s)! Best step index: {}'.format(end - start,
                                                        best_idx))
