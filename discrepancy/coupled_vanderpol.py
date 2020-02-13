'''
Discrepancy Function with Scaling Experiments for Coupled vanderpol

Stanley Bak, Jan 2020
'''

import random
import math

import numpy as np
from scipy.integrate import RK45
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def der_func(_, state):
    '''derivative function used by RK45

    This is the coupled vanderpol which can be called with any (even) number of dimensions. It is
    a generalization of the coupling from this paper:

    Rand and Holmes "Bifurcation of periodic motions in two weakly coupled van der pol oscillators"
    International Journal of Non-Linear Mechanics, 1980

    x_i' = y_i
    y_i' = (1 - x_i*x_i) * y_i - x_i + (x_{i-1} - x_i) [if x_{i-1} exists] + (x_{i+1} - x_i) [if x_{i+1} exists]
    '''
    
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
            # y_i' = (1 - x_i*x_i) * y_i - x_i + (x_{i-1} - x_i) [if x_{i-1} exists] + (x_{i+1} - x_i) [if x_{i+1} exists]
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

def simulate(x0, tmax, sim_step):
    '''simulate the vanderpol system from the given initial state

    returns an numpy array of states, where rows are time steps and cols are variables
    '''

    times = [0]
    states = [x0]

    rk45 = RK45(der_func, times[-1], states[-1], tmax)

    while rk45.status == 'running':
        rk45.step()

        if rk45.t > times[-1] + sim_step:
            dense_output = rk45.dense_output()

            while rk45.t > times[-1] + sim_step:
                t = times[-1] + sim_step
                times.append(t)
                states.append(dense_output(t))

    # make sure the solver didn't fail
    assert rk45.status == 'finished'

    return np.array(states, dtype=float)

def simulate_track(x0, sim_step, center_sim):
    '''simulate the vanderpol system from the given initial state,
    trying to adjust the time to stay as close as possible to center_sim

    returns an numpy array of states, where rows are time steps and cols are variables
    '''

    states = []
    times = []

    target_index = 0
    target = center_sim[target_index]
    target_grad = der_func(None, target)
    target_level = np.dot(target, target_grad)
    
    cur_state = x0
    cur_time = 0

    ep = 1e-6
    rk45 = RK45(der_func, 0, x0, np.inf)
    rk45.step()

    stuck_counter = 0
    last_states = 0

    while rk45.status == 'running':
        # check if advancing time moves us closer to target
        min_time_level = np.dot(cur_state, target_grad)

        if len(states) == last_states:
            stuck_counter += 1

            if stuck_counter > 100:
                while len(states) < len(center_sim):
                    states.append(cur_state)

                break
        else:
            last_states = len(states)
            stuck_counter = 0

        #print(f"\n{len(states)}: target_level = {target_level}")
        #print(f"min_time_level = {min_time_level}")

        if min_time_level < target_level:
            # we're moving closer to target
            
            # check if other endpoint is also moving closer
            max_time_level = np.dot(rk45.y, target_grad)

            #print(f"max_time_level = {max_time_level}")
            
            if min_time_level < target_level and max_time_level < target_level:
                # next!
                cur_state = rk45.y
                cur_time = rk45.t
                rk45.step()
                #print(f"step to {cur_time} and {cur_state}!")
                #print(f"target_grad = {target_grad}")
            else:
                assert min_time_level <= target_level <= max_time_level
                # closest point is within the current step, find it!
                d = rk45.dense_output()

                #print("finding closest point")

                min_time = cur_time
                max_time = d.t_max
                tol = 1e-6

                while True:
                    frac = (target_level - min_time_level) / (max_time_level - min_time_level)
                    assert 0 <= frac < 1.0

                    mid_time = min_time + frac * (max_time - min_time)
                    mid_state = d(mid_time)
                    mid_level = np.dot(mid_state, target_grad)

                    diff = abs(mid_level - target_level)

                    if diff < tol:
                        #print("found!")
                        break

                    if mid_level < target_level:
                        min_time = mid_time
                    else:
                        max_time = mid_time

                # mid_state is now the closest
                states.append(mid_state)
                times.append(mid_time)

                target_index += 1

                if target_index == len(center_sim):
                    break

                target = center_sim[target_index]
                target_grad = der_func(None, target)
                target_level = np.dot(target, target_grad)

                cur_state = mid_state
                cur_time = mid_time

                #print(f"updated cur_state (min_time) level to be: {np.dot(cur_state, target_grad)}")
        else:
            # frozen
            #print("frozen!")
            states.append(cur_state)

            assert not times or cur_time >= times[-1]
            times.append(cur_time)
            target_index += 1

            if target_index == len(center_sim):
                break

            target = center_sim[target_index]
            target_grad = der_func(None, target)
            target_level = np.dot(target, target_grad)

    # make sure the solver didn't fail
    assert rk45.status in ['finished', 'running']

    return np.array(states, dtype=float)

def get_center_orth(center):
    'get the xs and ys for plotting the center orthogonal line'

    cx = center[0]
    cy = center[1]

    center_grad = der_func(None, center)
    
    # for plotting we only use x and y
    gx = center_grad[0]
    gy = center_grad[1]

    # normalize
    desired_len = 25.0
    glen = math.sqrt(gx * gx + gy * gy) / desired_len
    glen = max(1e-6, glen)
    gx /= glen
    gy /= glen
    gx1 = cx - gy
    gx2 = cx + gy
    gy1 = cy + gx
    gy2 = cy - gx

    xs = [gx1, gx2]
    ys = [gy1, gy2]

    return xs, ys

def max_lines(tuples):
    'convert initdist, cur_dist tuples to max lines verts'

    tuples.sort()
    last_x = tuples[-1][0]

    points = np.array(tuples + [(0, 0)], dtype=float)
    assert points.shape[1] == 2, "2 columns expected"
    
    hull = ConvexHull(points, qhull_options='QJ')

    xs = []
    ys = []

    # we dont want the entire hull    
    hull_xs = points[hull.vertices, 0]
    hull_ys = points[hull.vertices, 1]

    index = 0

    while True:
        x, y = hull_xs[index], hull_ys[index]
        
        if xs or (x == 0 and y == 0):
            xs.append(x)
            ys.append(y)

            if x == last_x:
                break

        # decrement to get clockwise order
        index -= 1

        if index < 0:
            index = hull_xs.size - 1

    return xs, ys

def make_anim(use_scaling):
    'make animation (main function)'

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16

    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))

    random.seed(0) # determinstic random numbrers

    dims = 2
    init = np.array([0.35] * dims, dtype=float)
    tmax = 15
    step = 0.01 # 0.01
    rand_radius = 1.0 # 0.5, random perturbation around initial state 
    npoints = 50 # number of simulations

    all_sims = []
    dots = []
    lines = []

    # center simulation
    center_sim = simulate(init.copy(), tmax, step)
    center_line = axes[0].plot(center_sim[:, 0], center_sim[:, 1], 'g-', lw=0.8)[0]

    cindex = 0
    cx = center_sim[cindex, 0]
    cy = center_sim[cindex, 1]    
    center_dot = axes[0].plot([cx], [cy], 'g.', markersize=10)[0]

    xs, ys = get_center_orth(center_sim[cindex])
    center_orth_line = axes[0].plot(xs, ys, 'g:')[0]

    if not use_scaling:
        center_orth_line.set_visible(False)
        center_dot.set_visible(False)
        center_line.set_visible(False)

    for index in range(npoints):
        x0 = init.copy()

        for i, x in enumerate(x0):
            r = random.random() * 2 * rand_radius - rand_radius
            x0[i] = x + r

        print(f"Simulating {index+1}/{npoints}...")
        if use_scaling:
            states = simulate_track(x0, step, center_sim)
        else:
            states = simulate(x0, tmax, step)

        assert states.size == center_sim.size

        all_sims.append(states)

        line = axes[0].plot(states[:, 0], states[:, 1], 'k-', lw=0.15)[0]
        lines.append(line)

        dot = axes[0].plot([states[0, 0]], [states[0, 1]], 'b.', markersize=6)[0]
        dots.append(dot)

    time_text = axes[0].text(0.05, 0.95, 'Time: 0', horizontalalignment='left', fontsize=14,
                             verticalalignment='top', transform=axes[0].transAxes)

    label = "" if not use_scaling else "Scaled "

    axes[0].set_title(f"Vanderpol {label}Simulation ({init.size}D)")
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)

    ###############################
    index = 0
    init_dists = []
    cur_dists = []

    for p1 in range(npoints):
        for p2 in range(npoints):
            if p1 == p2:
                continue

            init_dist = np.linalg.norm(all_sims[p1][0] - all_sims[p2][0])
            cur_dist = np.linalg.norm(all_sims[p1][index] - all_sims[p2][index])

            init_dists.append(init_dist)
            cur_dists.append(cur_dist)

    # find ymax
    ymax = 0
    max_div = 20
    max_index = all_sims[0].shape[0] - 1

    for div in range(max_div):
        index = min(int(div * max_index / max_div), max_index)

        for p1 in range(npoints):
            for p2 in range(npoints):
                if p1 == p2:
                    continue

                cur_dist = np.linalg.norm(all_sims[p1][index] - all_sims[p2][index])

                if cur_dist > ymax:
                    ymax = cur_dist

    axes[1].set_ylim(0, 1.1 * ymax)

    ## disc func upper bound
    if npoints > 1:
        tuples = list(zip(init_dists, cur_dists))

        xs, ys = max_lines(tuples)
    else:
        xs = []
        ys = []

    disc_func = axes[1].plot(xs, ys, 'r-', lw=4, zorder=1)[0]
    dists = axes[1].plot(init_dists, cur_dists, 'b.', markersize=6, zorder=2)[0]

    axes[1].set_title(f'Piecewise Discrepancy Function')
    axes[1].set_xlabel('Initial Distance')
    axes[1].set_ylabel(f'Current Distance')

    freeze_frames = 40
    num_frames = states.shape[0] + 2*freeze_frames

    def animate(f):
        'animate function'

        if (f+1) % 10 == 0:
            print(f"Frame: {f+1} / {num_frames}")

        f = f - freeze_frames

        f = max(0, f)
        f = min(f, max_index)

        time_text.set_text(f"Time: {(f * step):.2f}")

        cindex = f
        cx = center_sim[cindex, 0]
        cy = center_sim[cindex, 1]
        center_dot.set_data([cx], [cy])

        xs, ys = get_center_orth(center_sim[cindex])
        center_orth_line.set_data(xs, ys)

        center_line.set_data([center_sim[:f+1, 0]], [center_sim[:f+1, 1]])

        for n, dot in enumerate(dots):
            states = all_sims[n]
            dot.set_data([states[f, 0]], [states[f, 1]])

        for n, line in enumerate(lines):
            states = all_sims[n]
            line.set_data([states[:f+1, 0]], [states[:f+1, 1]])

        cur_dists = []

        for p1 in range(npoints):
            for p2 in range(npoints):
                if p1 == p2:
                    continue

                cur_dist = np.linalg.norm(all_sims[p1][f] - all_sims[p2][f])
                cur_dists.append(cur_dist)

            dists.set_data(init_dists, cur_dists)

        ## disc func upper bound
        if npoints > 1:
            tuples = list(zip(init_dists, cur_dists))

            xs, ys = max_lines(tuples)
            disc_func.set_data(xs, ys)

        return dots + lines + [time_text, dists, disc_func, center_line, center_dot, center_orth_line]

    interval = 10 # 10
    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, \
                                      interval=interval, blit=True, repeat=True)

    make_mp4 = True # set to False to plot live
    
    if make_mp4:
        writer = animation.writers['ffmpeg'](fps=60, metadata=dict(artist='Stanley Bak'), bitrate=1800) # was 1800

        for i in range(2):
            axes[i].xaxis.set_tick_params(labelsize=16)
            axes[i].yaxis.set_tick_params(labelsize=16)

        label = "original" if not use_scaling else "scaled"
        filename = f'discrepancy_{label}.mp4'
        print(f"Making {filename}...")
        my_anim.save(filename, writer=writer)
    else:
        plt.show()

if __name__ == "__main__":
    # create both animations
    make_anim(False)
    make_anim(True)
