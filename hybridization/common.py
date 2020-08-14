'''common functions for scaled hybridiation'''

import numpy as np
from scipy.integrate import RK45

def jac_func(x, dx=1e-8):
    '''
    numeric jacobian, based on code from: 
    https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python
    '''
    
    n = len(x)
    func = der_func(None, x)
    jac = np.zeros((n, n))
    
    for j in range(n):  # through columns to allow for vector addition
        dxj = (abs(x[j])*dx if x[j] != 0 else dx)
        x_plus = np.array([(xi if k != j else xi + dxj) for k, xi in enumerate(x)], dtype=float)
        jac[:, j] = (der_func(None, x_plus) - func) / dxj
        
    return jac

def der_func(_, state):
    '''derivative function used by RK45

    This is the coupled vanderpol which can be called with any (even) number of dimensions. It is
    a generalization of the coupling from this paper:

    Rand and Holmes "Bifurcation of periodic motions in two weakly coupled van der pol oscillators"
    International Journal of Non-Linear Mechanics, 1980

    x_i' = y_i
    y_i' = (1 - x_i*x_i) * y_i - x_i + (x_{i-1} - x_i) [if x_{i-1} exists] + (x_{i+1} - x_i) [if x_{i+1} exists]
    '''

    assert len(state.shape) == 1, "expected 1d np.array as input state"
    dims = state.size
    assert dims % 2 == 0
    
    # variable order is x0, y0, x1, y1, ...

    rv = np.zeros((dims,), dtype=float)

    for n in range(dims):
        if n % 2 == 0:
            #x_i' = y_i
            yi = state[n+1]
            rv[n] = yi
        else:
            # y_i' = (1 - x_i*x_i) * y_i - x_i + (x_{i-1} - x_i) + (x_{i+1} - x_i)
            xi = state[n-1]
            yi = state[n]
            rv[n] = (1 - xi*xi) * yi - xi

            if n-3 >= 0: # if x_{i-1} exists
                xprev = state[n-3]

                rv[n] += (xprev - xi)

            if n+1 < dims: # if x_{i+1} exists
                xnext = state[n+1]
                rv[n] += (xnext - xi)

    return rv

def make_linear_der_func(mat, vec):
    'make a linear derivative function for rk45'

    return lambda _, state: mat.dot(state) + vec

def simulate_linear(center_state, x0_list, tmax, sim_step):
    '''simulate a linearized version of the vanderpol system

    returns an numpy array of states for each x0, where rows are time steps and cols are variables
    '''

    states_list = [[x0] for x0 in x0_list]

    num_steps = int(round(tmax / sim_step))

    rtol = 1e-5
    atol = 1e-7

    for s in range(num_steps):
        if s + 1 % 100 == 0:
            print(f"Simulate Linear Step {s+1} / {num_steps}")

        # use center to create linear approximation
        jac = jac_func(center_state)

        center_linear = jac.dot(center_state)
        center_nonlinear = der_func(None, center_state)

        affine_term = center_nonlinear - center_linear
        linear_der_func = make_linear_der_func(jac, affine_term)

        # simulate center with linear approximation
        rk45 = RK45(linear_der_func, 0, center_state, sim_step, rtol=rtol, atol=atol)
        while rk45.status == 'running':
            rk45.step()

        # make sure the solver didn't fail
        assert rk45.status == 'finished'
        center_state = rk45.y
        
        for states in states_list:
            rk45 = RK45(linear_der_func, 0, states[-1], sim_step, rtol=rtol, atol=atol)

            while rk45.status == 'running':
                rk45.step()

            # make sure the solver didn't fail
            assert rk45.status == 'finished'
            
            states.append(rk45.y)

    rv = []

    for states in states_list:
        rv.append(np.array(states, dtype=float))

    return rv

def simulate(x0, tmax, sim_step, scale_func=None, stop_func=None):
    '''simulate the vanderpol system from the given initial state

    returns an numpy array of states, where rows are time steps and cols are variables
    '''

    times = [0]
    states = [x0]

    scaled_der_func = der_func

    if scale_func is not None:
        scaled_der_func = lambda t, x: scale_func(t, x) * der_func(t, x)

    rtol = 1e-5
    atol = 1e-7
    max_step = 0.01
    rk45 = RK45(scaled_der_func, times[-1], states[-1], tmax, rtol=rtol, atol=atol, max_step=max_step)

    stop_sim = False

    while rk45.status == 'running' and not stop_sim:
        rk45.step()

        if rk45.t > times[-1] + sim_step:
            dense_output = rk45.dense_output()

            while rk45.t > times[-1] + sim_step and not stop_sim:
                t = times[-1] + sim_step
                times.append(t)
                states.append(dense_output(t))

                if stop_func is not None:
                    stop_sim = stop_func(states)

    # make sure the solver didn't fail
    assert stop_sim or rk45.status == 'finished'

    return np.array(states, dtype=float)

def simulate_dynamics_scaled(center_state, scale_distance, scale_factor, x0_list, tmax, stop_func, sim_step):
    '''simulate a dynamics-scaled version of the vanderpol system

    returns a 2-tuple (A, B) where 
        A: a numpy array of states for each x0, where rows are time steps and cols are variables
        B: a list (one for each time step) of (point, vec) used for scaling, where point is in front of center and
        vec is the normalized direction towards the center (opposite of gradient)
        C: a list of center_states at each time step
    '''

    states_list = [[x0] for x0 in x0_list]

    num_steps = int(round(tmax / sim_step))

    rv_b = []

    scale_norm_vec = None
    scale_offset = None

    rtol = 1e-5
    atol = 1e-7

    def scaled_der_func(t, state):
        'scaled derivative func, uses scale_norm_vec and scale_offset'

        scale = state.dot(scale_norm_vec) - scale_offset

        assert scale > 0

        return der_func(t, state) * scale * scale_factor

    center_states = [center_state]
    
    for s in range(num_steps):
        if (s + 1) % 100 == 0:
            print(f"Simulate Dynamics Scaled Step {s+1} / {num_steps}")

        # use center to create linear approximation
        gradient = der_func(None, center_state)

        # normalize gradient
        gradient /= np.linalg.norm(gradient)

        pt = center_state + gradient * scale_distance
        scale_norm_vec = -gradient

        scale_offset = scale_norm_vec.dot(pt)

        rv_b.append((pt, scale_norm_vec))

        # scale function is vec.dot(new_point) - offset... should always be positive
        
        # simulate center with linear approximation
        rk45 = RK45(scaled_der_func, 0, center_state, sim_step, rtol=rtol, atol=atol)
        while rk45.status == 'running':
            rk45.step()

        # make sure the solver didn't fail
        assert rk45.status == 'finished'
        center_state = rk45.y

        center_states.append(center_state)

        if stop_func(center_states):
            center_states = center_states[:-1]
            break
        
        for states in states_list:
        
            rk45 = RK45(scaled_der_func, 0, states[-1], sim_step, rtol=rtol, atol=atol)

            while rk45.status == 'running':
                rk45.step()

            # make sure the solver didn't fail
            assert rk45.status == 'finished'
            
            states.append(rk45.y)

    rv_a = []

    for states in states_list:
        rv_a.append(np.array(states, dtype=float))

    rv_c = np.array(center_states, dtype=float)

    if len(rv_b) < rv_c.shape[0]:
        rv_b.append(rv_b[-1]) # only needed when tmax is reached

    return rv_a, rv_b, rv_c
