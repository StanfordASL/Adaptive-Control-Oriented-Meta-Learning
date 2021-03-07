"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

if __name__ == "__main__":
    import pickle
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
    from utils import spline, random_ragged_spline
    from dynamics import prior, plant, disturbance

    # Seed random numbers
    seed = 0
    key = jax.random.PRNGKey(seed)

    # Generate smooth trajectories
    num_traj = 500
    T = 30
    num_knots = 6
    poly_orders = (9, 9, 6)
    deriv_orders = (4, 4, 2)
    min_step = jnp.array([-2., -2., -jnp.pi/6])
    max_step = jnp.array([2., 2., jnp.pi/6])
    min_knot = jnp.array([-jnp.inf, -jnp.inf, -jnp.pi/3])
    max_knot = jnp.array([jnp.inf, jnp.inf, jnp.pi/3])

    key, *subkeys = jax.random.split(key, 1 + num_traj)
    subkeys = jnp.vstack(subkeys)
    in_axes = (0, None, None, None, None, None, None, None, None)
    t_knots, knots, coefs = jax.vmap(random_ragged_spline, in_axes)(
        subkeys, T, num_knots, poly_orders, deriv_orders,
        min_step, max_step, min_knot, max_knot
    )
    # x_coefs, y_coefs, ϕ_coefs = coefs
    r_knots = jnp.dstack(knots)

    # Sampled-time simulator
    @jax.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def simulate(ts, w, t_knots, coefs,
                 plant=plant, prior=prior, disturbance=disturbance):
        """TODO: docstring."""
        # Construct spline reference trajectory
        def reference(t):
            x_coefs, y_coefs, ϕ_coefs = coefs
            x = spline(t, t_knots, x_coefs)
            y = spline(t, t_knots, y_coefs)
            ϕ = spline(t, t_knots, ϕ_coefs)
            ϕ = jnp.clip(ϕ, -jnp.pi/3, jnp.pi/3)
            r = jnp.array([x, y, ϕ])
            return r

        # Required derivatives of the reference trajectory
        def ref_derivatives(t):
            ref_vel = jax.jacfwd(reference)
            ref_acc = jax.jacfwd(ref_vel)
            r = reference(t)
            dr = ref_vel(t)
            ddr = ref_acc(t)
            return r, dr, ddr

        # Feedback linearizing PD controller
        def controller(q, dq, r, dr, ddr):
            kp, kd = 10., 0.1
            e, de = q - r, dq - dr
            dv = ddr - kp*e - kd*de
            H, C, g, B = prior(q, dq)
            τ = H@dv + C@dq + g
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
            t_prev, q_prev, dq_prev, u_prev = carry
            t = input_slice
            qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]),
                             u_prev)
            q, dq = qs[-1], dqs[-1]
            r, dr, ddr = ref_derivatives(t)
            u, τ = controller(q, dq, r, dr, ddr)
            carry = (t, q, dq, u)
            output_slice = (q, dq, u, τ, r, dr)
            return carry, output_slice

        # Initial conditions
        t0 = ts[0]
        r0, dr0, ddr0 = ref_derivatives(t0)
        q0, dq0 = r0, dr0
        u0, τ0 = controller(q0, dq0, r0, dr0, ddr0)

        # Run simulation loop
        carry = (t0, q0, dq0, u0)
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

    # Sample wind velocities from the training distribution
    w_min = 0.  # minimum wind velocity in inertial `x`-direction
    w_max = 6.  # maximum wind velocity in inertial `x`-direction
    a = 5.      # shape parameter `a` for beta distribution
    b = 9.      # shape parameter `b` for beta distribution
    key, subkey = jax.random.split(key, 2)
    w = w_min + (w_max - w_min)*jax.random.beta(subkey, a, b, (num_traj,))

    # Simulate tracking for each `w`
    dt = 0.01
    t = jnp.arange(0, T + dt, dt)  # same times for each trajectory
    q, dq, u, τ, r, dr = simulate(t, w, t_knots, coefs)

    data = {
        'seed': seed, 'prng_key': key,
        't': t, 'q': q, 'dq': dq,
        'u': u, 'r': r, 'dr': dr,
        't_knots': t_knots, 'r_knots': r_knots,
        'w': w, 'w_min': w_min, 'w_max': w_max,
        'beta_params': (a, b),
    }

    with open('training_data.pkl', 'wb') as file:
        pickle.dump(data, file)
