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
from math import pi, inf
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

import jax                                          # noqa: E402
import jax.numpy as jnp                             # noqa: E402
from jax.experimental import optimizers             # noqa: E402
from dynamics import prior                          # noqa: E402
from utils import (tree_normsq, rk38_step, epoch,   # noqa: E402
                   odeint_fixed_step, random_ragged_spline, spline,
                   params_to_cholesky, params_to_posdef)


# Initialize PRNG key
key = jax.random.PRNGKey(args.seed)

# Hyperparameters
hparams = {
    'seed':        args.seed,     #
    'use_x64':     args.use_x64,  #
    'num_subtraj': args.M,        # number of trajectories to sub-sample

    # For training the model ensemble
    'ensemble': {
        'num_hlayers':    2,     # number of hidden layers in each model
        'hdim':           32,    # number of hidden units per layer
        'train_frac':     0.75,  # fraction of each trajectory for training
        'batch_frac':     0.25,  # fraction of training data per batch
        'regularizer_l2': 1e-4,  # coefficient for L2-regularization
        'learning_rate':  1e-2,  # step size for gradient optimization
        'num_epochs':     1000,  # number of epochs
    },
    # For meta-training
    'meta': {
        'num_hlayers':       2,          # number of hidden layers
        'hdim':              32,         # number of hidden units per layer
        'train_frac':        0.75,       #
        'learning_rate':     1e-2,       # step size for gradient optimization
        'num_steps':         500,        # maximum number of gradient steps
        'regularizer_l2':    1e-4,       # coefficient for L2-regularization
        'regularizer_ctrl':  1e-3,       #
        'regularizer_error': 0.,         #
        'T':                 5.,         # time horizon for each reference
        'dt':                1e-2,       # time step for numerical integration
        'num_refs':          10,         # reference trajectories to generate
        'num_knots':         6,          # knot points per reference spline
        'poly_orders':       (9, 9, 6),  # spline orders for each DOF
        'deriv_orders':      (4, 4, 2),  # smoothness objective for each DOF
        'min_step':          (-2., -2., -pi/6),    #
        'max_step':          (2., 2., pi/6),       #
        'min_ref':           (-inf, -inf, -pi/3),  #
        'max_ref':           (inf, inf, pi/3),     #
    },
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

    # MODEL ENSEMBLE TRAINING ################################################
    # Loss function along a trajectory
    def ode(x, t, u, params, prior=prior):
        """TODO: docstring."""
        num_dof = x.size // 2
        q, dq = x[:num_dof], x[num_dof:]
        H, C, g, B = prior(q, dq)

        # Each model in the ensemble is a feed-forward neural network
        # with zero output bias
        f = x
        for W, b in zip(params['W'], params['b']):
            f = jnp.tanh(W@f + b)
        f = params['A'] @ f
        ddq = jax.scipy.linalg.solve(H, B@u + f - C@dq - g, sym_pos=True)
        dx = jnp.concatenate((dq, ddq))
        return dx

    def loss(params, regularizer, t, x, u, t_next, x_next, ode=ode):
        """TODO: docstring."""
        num_samples = t.size
        dt = t_next - t
        x_next_est = jax.vmap(rk38_step, (None, 0, 0, 0, 0, None))(
            ode, dt, x, t, u, params
        )
        loss = (jnp.sum((x_next_est - x_next)**2)
                + regularizer*tree_normsq(params)) / num_samples
        return loss

    # Parallel updates for each model in the ensemble
    @jax.partial(jax.jit, static_argnums=(4, 5))
    @jax.partial(jax.vmap, in_axes=(None, 0, None, 0, None, None))
    def step(idx, opt_state, regularizer, batch, get_params, update_opt,
             loss=loss):
        """TODO: docstring."""
        params = get_params(opt_state)
        grads = jax.grad(loss, argnums=0)(params, regularizer, **batch)
        opt_state = update_opt(idx, grads, opt_state)
        return opt_state

    @jax.jit
    @jax.vmap
    def update_best_ensemble(old_params, old_loss, new_params, batch):
        """TODO: docstring."""
        new_loss = loss(new_params, 0., **batch)  # do not regularize
        best_params = jax.tree_util.tree_multimap(
            lambda x, y: jnp.where(new_loss < old_loss, x, y),
            new_params,
            old_params
        )
        best_loss = jnp.where(new_loss < old_loss, new_loss, old_loss)
        return best_params, best_loss, new_loss

    # Initialize model parameters
    num_models = hparams['num_subtraj']  # one model per trajectory
    num_hlayers = hparams['ensemble']['num_hlayers']
    hdim = hparams['ensemble']['hdim']
    if num_hlayers >= 1:
        shapes = [(hdim, 2*num_dof), ] + (num_hlayers-1)*[(hdim, hdim), ]
    else:
        shapes = []
    key, *subkeys = jax.random.split(key, 1 + 2*num_hlayers + 1)
    keys_W = subkeys[:num_hlayers]
    keys_b = subkeys[num_hlayers:-1]
    key_A = subkeys[-1]
    ensemble = {
        # hidden layer weights
        'W': [0.1*jax.random.normal(keys_W[i], (num_models, *shapes[i]))
              for i in range(num_hlayers)],
        # hidden layer biases
        'b': [0.1*jax.random.normal(keys_b[i], (num_models, shapes[i][0]))
              for i in range(num_hlayers)],
        # last layer weights
        'A': 0.1*jax.random.normal(key_A, (num_models, num_dof, hdim))
    }

    # Shuffle samples in time along each trajectory, then split each
    # trajectory into training and validation sets (i.e., for each model)
    key, *subkeys = jax.random.split(key, 1 + num_models)
    subkeys = jnp.asarray(subkeys)
    shuffled_data = jax.tree_util.tree_map(
        lambda a: jax.vmap(jax.random.permutation)(subkeys, a),
        data
    )
    num_train_samples = int(hparams['ensemble']['train_frac'] * num_samples)
    ensemble_train_data = jax.tree_util.tree_map(
        lambda a: a[:, :num_train_samples],
        shuffled_data
    )
    ensemble_valid_data = jax.tree_util.tree_map(
        lambda a: a[:, num_train_samples:],
        shuffled_data
    )

    # Initialize gradient-based optimizer (ADAM)
    learning_rate = hparams['ensemble']['learning_rate']
    batch_size = int(hparams['ensemble']['batch_frac'] * num_train_samples)
    num_batches = num_train_samples // batch_size
    init_opt, update_opt, get_params = optimizers.adam(learning_rate)
    opt_states = jax.vmap(init_opt)(ensemble)
    get_ensemble = jax.jit(jax.vmap(get_params))
    step_idx = 0
    best_idx = jnp.zeros(num_models)

    # Pre-compile before training
    print('ENSEMBLE TRAINING: Pre-compiling ... ', end='', flush=True)
    start = time.time()
    batch = next(epoch(key, ensemble_train_data, batch_size,
                       batch_axis=1, ragged=False))
    _ = step(step_idx, opt_states, hparams['ensemble']['regularizer_l2'],
             batch, get_params, update_opt)
    inf_losses = jnp.broadcast_to(jnp.inf, (num_models,))
    best_ensemble, best_losses, _ = update_best_ensemble(ensemble,
                                                         inf_losses,
                                                         ensemble,
                                                         ensemble_valid_data)
    _ = get_ensemble(opt_states)
    end = time.time()
    print('done ({:.2f} s)!'.format(end - start))

    # Do gradient descent
    for _ in tqdm(range(hparams['ensemble']['num_epochs'])):
        key, subkey = jax.random.split(key, 2)
        for batch in epoch(subkey, ensemble_train_data, batch_size,
                           batch_axis=1, ragged=False):
            opt_states = step(step_idx, opt_states,
                              hparams['ensemble']['regularizer_l2'],
                              batch, get_params, update_opt)
            new_ensemble = get_ensemble(opt_states)
            old_losses = best_losses
            best_ensemble, best_losses, valid_losses = update_best_ensemble(
                best_ensemble, best_losses, new_ensemble, batch
            )
            step_idx += 1
            best_idx = jnp.where(old_losses == best_losses,
                                 best_idx, step_idx)

    # META-TRAINING ##########################################################
    def ode(z, t, meta_params, params, reference, prior=prior):
        """TODO: docstring."""
        x, A, c = z
        num_dof = x.size // 2
        q, dq = x[:num_dof], x[num_dof:]
        r = reference(t)
        dr = jax.jacfwd(reference)(t)
        ddr = jax.jacfwd(jax.jacfwd(reference))(t)

        # Regressor features
        y = x
        for W, b in zip(meta_params['W'], meta_params['b']):
            y = jnp.tanh(W@y + b)

        # Parameterized control and adaptation gains
        gains = jax.tree_util.tree_map(
            lambda x: params_to_posdef(x),
            meta_params['gains']
        )
        Λ, K, P = gains['Λ'], gains['K'], gains['P']

        # Auxiliary signals
        e, de = q - r, dq - dr
        v, dv = dr - Λ@e, ddr - Λ@de
        s = de + Λ@e

        # Controller and adaptation law
        H, C, g, B = prior(q, dq)
        f_hat = A@y
        τ = H@dv + C@v + g - f_hat - K@s
        u = jnp.linalg.solve(B, τ)
        dA = P @ jnp.outer(s, y)

        # Apply control to "true" dynamics
        f = x
        for W, b in zip(params['W'], params['b']):
            f = jnp.tanh(W@f + b)
        f = params['A'] @ f
        ddq = jax.scipy.linalg.solve(H, τ + f - C@dq - g, sym_pos=True)
        dx = jnp.concatenate((dq, ddq))

        # Estimation loss
        # chol_P = params_to_cholesky(meta_params['gains']['P'])
        # f_error = f_hat - f
        # loss_est = f_error@jax.scipy.linalg.cho_solve((chol_P, True),
        #                                               f_error)

        # Integrated cost terms
        dc = jnp.array([
            e@e + de@de,                # tracking loss
            u@u,                        # control loss
            (f_hat - f)@(f_hat - f),    # estimation loss
        ])

        # Assemble derivatives
        dz = (dx, dA, dc)
        return dz

    # Simulate adaptive control loop on each model in the ensemble
    def ensemble_sim(meta_params, ensemble_params, reference, T, dt, ode=ode):
        """TODO: docstring."""
        # Initial conditions
        r0 = reference(0.)
        dr0 = jax.jacfwd(reference)(0.)
        num_dof = r0.size
        num_features = meta_params['W'][-1].shape[0]
        x0 = jnp.concatenate((r0, dr0))
        A0 = jnp.zeros((num_dof, num_features))
        c0 = jnp.zeros(3)
        z0 = (x0, A0, c0)

        # Integrate the adaptive control loop using the meta-model
        # and EACH model in the ensemble along the same reference
        in_axes = (None, None, None, None, None, None, 0)
        ode = jax.partial(ode, reference=reference)
        z, t = jax.vmap(odeint_fixed_step, in_axes)(ode, z0, 0., T, dt,
                                                    meta_params,
                                                    ensemble_params)
        x, A, c = z
        return t, x, A, c

    # Initialize meta-model parameters
    num_hlayers = hparams['meta']['num_hlayers']
    hdim = hparams['meta']['hdim']
    if num_hlayers >= 1:
        shapes = [(hdim, 2*num_dof), ] + (num_hlayers-1)*[(hdim, hdim), ]
    else:
        shapes = []
    key, *subkeys = jax.random.split(key, 1 + 2*num_hlayers + 3)
    subkeys_W = subkeys[:num_hlayers]
    subkeys_b = subkeys[num_hlayers:-3]
    subkeys_gains = subkeys[-3:]
    meta_params = {
        # hidden layer weights
        'W': [0.1*jax.random.normal(subkeys_W[i], shapes[i])
              for i in range(num_hlayers)],
        # hidden layer biases
        'b': [0.1*jax.random.normal(subkeys_b[i], (shapes[i][0],))
              for i in range(num_hlayers)],
        'gains': {  # vectorized control and adaptation gains
            'Λ': 0.1*jax.random.normal(subkeys_gains[0],
                                       ((num_dof*(num_dof + 1)) // 2,)),
            'K': 0.1*jax.random.normal(subkeys_gains[1],
                                       ((num_dof*(num_dof + 1)) // 2,)),
            'P': 0.1*jax.random.normal(subkeys_gains[2],
                                       ((num_dof*(num_dof + 1)) // 2,)),
        }
    }

    # Initialize spline coefficients for each reference trajectory
    num_refs = hparams['meta']['num_refs']
    key, *subkeys = jax.random.split(key, 1 + num_refs)
    subkeys = jnp.vstack(subkeys)
    in_axes = (0, None, None, None, None, None, None, None, None)
    min_ref = jnp.asarray(hparams['meta']['min_ref'])
    max_ref = jnp.asarray(hparams['meta']['max_ref'])
    t_knots, knots, coefs = jax.vmap(random_ragged_spline, in_axes)(
        subkeys,
        hparams['meta']['T'],
        hparams['meta']['num_knots'],
        hparams['meta']['poly_orders'],
        hparams['meta']['deriv_orders'],
        jnp.asarray(hparams['meta']['min_step']),
        jnp.asarray(hparams['meta']['max_step']),
        0.7*min_ref,
        0.7*max_ref,
    )
    # x_coefs, y_coefs, θ_coefs = coefs
    # x_knots, y_knots, θ_knots = knots
    r_knots = jnp.dstack(knots)

    # Simulate the adaptive control loop for each model in the ensemble and
    # each reference trajectory (i.e., spline coefficients)
    @jax.partial(jax.vmap, in_axes=(None, None, 0, 0, None, None))
    def simulate(meta_params, ensemble_params, t_knots, coefs, T, dt,
                 min_ref=min_ref, max_ref=max_ref):
        """TODO: docstring."""
        # Define a reference trajectory in terms of spline coefficients
        def reference(t):
            r = jnp.array([spline(t, t_knots, c) for c in coefs])
            r = jnp.clip(r, min_ref, max_ref)
            return r
        t, x, A, c = ensemble_sim(meta_params, ensemble_params,
                                  reference, T, dt)
        return t, x, A, c

    @jax.partial(jax.jit, static_argnums=(4, 5))
    def loss(meta_params, ensemble_params, t_knots, coefs, T, dt,
             regularizer_l2, regularizer_ctrl, regularizer_error):
        """TODO: docstring."""
        # Simulate on each model for each reference trajectory
        t, x, A, c = simulate(meta_params, ensemble_params, t_knots,
                              coefs, T, dt)

        # Sum final costs over reference trajectories and ensemble models
        # Note `c` has shape (`num_refs`, `num_models`, `T // dt`, 3)
        c_final = jnp.sum(c[:, :, -1, :], axis=(0, 1))

        # Form a composite loss by weighting the different cost integrals,
        # and normalizing by the number of models, number of reference
        # trajectories, and time horizon
        num_refs = c.shape[0]
        num_models = c.shape[1]
        normalizer = T * num_refs * num_models
        tracking_loss, control_loss, estimation_loss = c_final
        l2_penalty = tree_normsq((meta_params['W'], meta_params['b']))
        loss = (tracking_loss
                + regularizer_ctrl*control_loss
                + regularizer_error*estimation_loss
                + regularizer_l2*l2_penalty) / normalizer
        aux = {
            # for each model in ensemble
            'tracking_loss':   jnp.sum(c[:, :, -1, 0], axis=0) / num_refs,
            'control_loss':    jnp.sum(c[:, :, -1, 1], axis=0) / num_refs,
            'estimation_loss': jnp.sum(c[:, :, -1, 2], axis=0) / num_refs,
            'l2_penalty':      l2_penalty,
            'eigs_Λ':
                jnp.diag(params_to_cholesky(meta_params['gains']['Λ']))**2,
            'eigs_K':
                jnp.diag(params_to_cholesky(meta_params['gains']['K']))**2,
            'eigs_P':
                jnp.diag(params_to_cholesky(meta_params['gains']['P']))**2,
        }
        return loss, aux

    # Shuffle and split ensemble into training and validation sets
    train_frac = hparams['meta']['train_frac']
    num_train_models = int(train_frac * num_models)
    key, subkey = jax.random.split(key, 2)
    model_idx = jax.random.permutation(subkey, num_models)
    train_model_idx = model_idx[:num_train_models]
    valid_model_idx = model_idx[num_train_models:]
    train_ensemble = jax.tree_util.tree_map(lambda x: x[train_model_idx],
                                            best_ensemble)
    valid_ensemble = jax.tree_util.tree_map(lambda x: x[valid_model_idx],
                                            best_ensemble)

    # Split reference trajectories into training and validation sets
    num_train_refs = int(train_frac * num_refs)
    train_t_knots = jax.tree_util.tree_map(lambda a: a[:num_train_refs],
                                           t_knots)
    train_coefs = jax.tree_util.tree_map(lambda a: a[:num_train_refs], coefs)
    valid_t_knots = jax.tree_util.tree_map(lambda a: a[num_train_refs:],
                                           t_knots)
    valid_coefs = jax.tree_util.tree_map(lambda a: a[num_train_refs:], coefs)

    # Initialize gradient-based optimizer (ADAM)
    learning_rate = hparams['meta']['learning_rate']
    init_opt, update_opt, get_params = optimizers.adam(learning_rate)
    opt_state = init_opt(meta_params)
    step_idx = 0
    best_idx = 0
    best_loss = jnp.inf
    best_meta_params = meta_params

    @jax.partial(jax.jit, static_argnums=(5, 6))
    def step(idx, opt_state, ensemble_params, t_knots, coefs, T, dt,
             regularizer_l2, regularizer_ctrl, regularizer_error):
        """TODO: docstring."""
        meta_params = get_params(opt_state)
        grads, aux = jax.grad(loss, argnums=0, has_aux=True)(
            meta_params, ensemble_params, t_knots, coefs, T, dt,
            regularizer_l2, regularizer_ctrl, regularizer_error
        )
        opt_state = update_opt(idx, grads, opt_state)
        return opt_state, aux

    # Pre-compile before training
    print('META-TRAINING: Pre-compiling ... ', end='', flush=True)
    dt = hparams['meta']['dt']
    T = hparams['meta']['T']
    regularizer_l2 = hparams['meta']['regularizer_l2']
    regularizer_ctrl = hparams['meta']['regularizer_ctrl']
    regularizer_error = hparams['meta']['regularizer_error']
    start = time.time()
    _ = step(0, opt_state, train_ensemble, train_t_knots, train_coefs, T, dt,
             regularizer_l2, regularizer_ctrl, regularizer_error)
    _ = loss(meta_params, valid_ensemble, valid_t_knots, valid_coefs, T, dt,
             0., 0., 0.)
    end = time.time()
    print('done ({:.2f} s)! Now training ...'.format(
          end - start))
    start = time.time()

    # Do gradient descent
    for _ in tqdm(range(hparams['meta']['num_steps'])):
        opt_state, train_aux = step(
            step_idx, opt_state, train_ensemble, train_t_knots, train_coefs,
            T, dt, regularizer_l2, regularizer_ctrl, regularizer_error
        )
        new_meta_params = get_params(opt_state)
        valid_loss, valid_aux = loss(
            new_meta_params, valid_ensemble, valid_t_knots, valid_coefs,
            T, dt, 0., 0., 0.
        )
        if valid_loss < best_loss:
            best_meta_params = new_meta_params
            best_loss = valid_loss
            best_idx = step_idx
        step_idx += 1

    # Save hyperparameters, ensemble, model, and controller
    output_name = "seed={:d}_M={:d}".format(hparams['seed'], num_models)
    results = {
        'best_step_idx': best_idx,
        'hparams': hparams,
        'ensemble': best_ensemble,
        'model': {
            'W': best_meta_params['W'],
            'b': best_meta_params['b'],
        },
        'controller': best_meta_params['gains'],
    }
    output_path = os.path.join('train_results', 'ours', output_name + '.pkl')
    with open(output_path, 'wb') as file:
        pickle.dump(results, file)

    end = time.time()
    print('done ({:.2f} s)! Best step index: {}'.format(end - start,
                                                        best_idx))
