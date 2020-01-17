'''
Discrepancy Function with Scaling Experiments for Coupled vanderpol

Stanley Bak, Jan 2020
'''

import random

import numpy as np
from scipy.integrate import RK45

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
    

def main():
    'main entry point'

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))

    random.seed(0) # determinstic random numbrers

    dims = 10
    init = np.array([0.35] * dims, dtype=float)
    tmax = 15
    step = 0.005

    rand_radius = 0.5 # random perturbation around initial state
    npoints = 50 # number of simulations
    all_sims = []

    dots = []
    lines = []

    for _ in range(npoints):
        x0 = init.copy()

        for i, x in enumerate(x0):
            r = x + random.random() * 2 * rand_radius - rand_radius            
            x0[i] = x + r

        states = simulate(x0, tmax, step)
        all_sims.append(states)
        
        line = axes[0].plot(states[:, 0], states[:, 1], 'k-', lw=0.2)[0]
        lines.append(line)
        #mid_index = states.shape[0] // 2
        #last_index = states.shape[0] - 1

        dot = axes[0].plot([states[0, 0]], [states[0, 1]], 'b.')[0]
        dots.append(dot)

        #plt.plot([states[mid_index, 0]], [states[mid_index, 1]], 'go', label=f'T={tmax/2}' if p == 0 else None)
        #plt.plot([states[last_index, 0]], [states[last_index, 1]], 'bo', label=f'T={tmax}' if p == 0 else None)

    time_text = axes[0].text(0.05, 0.95, 'Time: 0', horizontalalignment='left', fontsize=14,
                             verticalalignment='top', transform=axes[0].transAxes)

    axes[0].set_title(f"Coupled Vanderpol Simulation ({init.size}D)")
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    #axes[0].legend()

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

    dists = axes[1].plot(init_dists, cur_dists, 'b.')[0]

    ## disc func upper bound
    tuples = list(zip(init_dists, cur_dists))
    tuples.sort()

    xs = [0]
    ys = [0]

    for init, cur in tuples:
        xs.append(init)
                
        if cur > ys[-1]:
            ys.append(cur)
        else:
            ys.append(ys[-1])
        
    disc_func = axes[1].plot(xs, ys, 'r-', lw=4)[0]

    axes[1].set_title(f'Piecewise Discrepancy Function')
    axes[1].set_xlabel('Initial Distance')
    axes[1].set_ylabel(f'Current Distance')

    freeze_frames = 40

    def animate(f):
        'animate function'

        f = f - freeze_frames

        f = max(0, f)
        f = min(f, max_index)

        print(f"{f} / {max_index}")

        time_text.set_text(f"Time: {(f * step):.2f}")
        
        for n, dot in enumerate(dots):
            states = all_sims[n]
            dot.set_data([states[f, 0]], [states[f, 1]])

        for n, line in enumerate(lines):
            states = all_sims[n]
            line.set_data([states[:f, 0]], [states[:f, 1]])

        cur_dists = []

        for p1 in range(npoints):
            for p2 in range(npoints):
                if p1 == p2:
                    continue

                cur_dist = np.linalg.norm(all_sims[p1][f] - all_sims[p2][f])
                cur_dists.append(cur_dist)

            dists.set_data(init_dists, cur_dists)

        ## disc func upper bound
        tuples = list(zip(init_dists, cur_dists))
        tuples.sort()

        xs = [0]
        ys = [0]

        for init, cur in tuples:
            xs.append(init)

            if cur > ys[-1]:
                ys.append(cur)
            else:
                ys.append(ys[-1])

        disc_func.set_data(xs, ys)

        return dots + lines + [time_text, dists, disc_func]

    my_anim = animation.FuncAnimation(fig, animate, frames=states.shape[0] + 2*freeze_frames, \
                                      interval=10, blit=True, repeat=True)

    writer = animation.writers['ffmpeg'](fps=60, metadata=dict(artist='Stanley Bak'), bitrate=1200) # was 1800

    for i in range(2):
        axes[i].xaxis.set_tick_params(labelsize=16)
        axes[i].yaxis.set_tick_params(labelsize=16)

    my_anim.save('discrepancy.mp4', writer=writer)
    #plt.show()

if __name__ == "__main__":
    main()
