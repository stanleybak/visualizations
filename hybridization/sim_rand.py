'''sim_rand for scaled vanderpol'''

import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from common import simulate, der_func

def sim_rand(make_mp4=True):
    'make simulation animations animations'

    #plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))

    dims = 2
    init = np.array([0.0] * dims, dtype=float)
    tmax = 10
    step = 0.01 # 0.01
    rand_radius = 3.0
    npoints = 50 # number of simulations

    all_sims = []
    dots = []
    lines = []

    ax.set_title("Vanderpol Oscillator")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    rmin = -3
    rmax = 3
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(rmin, rmax)

    ################
    # quiver

    v_step = 0.2
    vec_min = 1.0
    vec_max = 2.0

    x_range = np.arange(rmin, rmax, v_step)
    y_range = np.arange(rmin, rmax, v_step)

    big_x, big_y = np.meshgrid(x_range, y_range)
    u_list = []
    v_list = []

    for x_row, y_row in zip(big_x, big_y):
        for x, y in zip(x_row, y_row):
            state = np.array([x, y], dtype=float)
            vec = der_func(None, state)
            norm = np.linalg.norm(vec)

            if norm > 1e-6:
                # scale the vector

                if norm < vec_min:
                    vec = vec / norm * vec_min

                    assert abs(np.linalg.norm(vec) - vec_min) < 1e-6
                elif norm > vec_max:
                    vec = vec / norm * vec_max
                    

            u_list.append(vec[0])
            v_list.append(vec[1])

    ax.quiver(big_x, big_y, u_list, v_list, color='grey')
    plt.savefig('vectorfield.png')
    ########################

    for index in range(npoints):
        x0 = init.copy()

        for i, x in enumerate(x0):
            r = random.random() * 2 * rand_radius - rand_radius
            x0[i] = x + r

        print(f"Simulating {index+1}/{npoints}...")
        states = simulate(x0, tmax, step)

        all_sims.append(states)

        line = ax.plot(states[:, 0], states[:, 1], 'k-', lw=1.0, zorder=1)[0]
        lines.append(line)

        dot = ax.plot([states[0, 0]], [states[0, 1]], 'ro', markersize=7, zorder=2)[0]
        dots.append(dot)

    time_text = ax.text(0.05, 0.95, 'Time: 0', horizontalalignment='left', fontsize=14,
                             verticalalignment='top', transform=ax.transAxes)

    freeze_frames = 40
    num_frames = states.shape[0] + 2*freeze_frames

    time_text.set_visible(False)


    def animate(f):
        'animate function'

        if (f+1) % 10 == 0:
            print(f"Frame: {f+1} / {num_frames}")

        f = f - freeze_frames

        f = max(0, f)
        max_index = len(all_sims[0]) - 1
        f = min(f, max_index)

        time_text.set_text(f"Time: {(f * step):.2f}")

        for n, dot in enumerate(dots):
            states = all_sims[n]
            dot.set_data([states[f, 0]], [states[f, 1]])

        for n, line in enumerate(lines):
            states = all_sims[n]
            line.set_data([states[:f+1, 0]], [states[:f+1, 1]])

        return dots + lines + [time_text]

    interval = 10 # 10
    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, \
                                      interval=interval, blit=True, repeat=True)
    
    if make_mp4:
        writer = animation.writers['ffmpeg'](fps=100, metadata=dict(artist='Stanley Bak'), bitrate=1800) # was 1800

        #ax.xaxis.set_tick_params(labelsize=16)
        #ax.yaxis.set_tick_params(labelsize=16)

        filename = 'sim_rand.mp4'
        print(f"Making {filename}...")
        my_anim.save(filename, writer=writer)
    else:
        plt.show()
