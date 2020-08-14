'''sim_linear for scaled vanderpol'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from common import simulate_linear, der_func

def sim_linear(make_mp4=True, tmax=None):
    'make simulation animations animations'

    #plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))

    if tmax is None:
        tmax = 12
        
    step = 0.01 # 0.01

    init_box = [[0.3, 0.7], [0.3, 0.7]]
    mesh_points = 65 #50 # points per row / column in initial set
    
    #ax.set_title("Vanderpol Oscillator")
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

    ax.quiver(big_x, big_y, u_list, v_list, color='lightgrey')

    ########################

    cx = (init_box[0][1] + init_box[0][0]) / 2
    cy = (init_box[1][1] + init_box[1][0]) / 2
    center_state = np.array([cx, cy], dtype=float)
    
    dx = (init_box[0][1] - init_box[0][0]) / (mesh_points - 1)
    dy = (init_box[1][1] - init_box[1][0]) / (mesh_points - 1)
    x0_list = []

    for xindex in range(mesh_points):
        for yindex in range(mesh_points):
            x = init_box[0][0] + dx * xindex
            y = init_box[1][0] + dy * yindex

            x0_list.append(np.array([x, y], dtype=float))

    all_sims = simulate_linear(center_state, x0_list, tmax, step)

    # drawing elements
    dots = []
    lines = []
    
    for states in all_sims:
        line = ax.plot(states[:, 0], states[:, 1], '-', lw=3.0, zorder=1)[0]
        line.set_color('grey')
        lines.append(line)

        dot = ax.plot([states[0, 0]], [states[0, 1]], 'ro', markersize=3, zorder=2)[0]
        #dot.set_color('lime')
        #dot.set_visible(False)
        dots.append(dot)

    # make cur_set indices
    curset_indices = []

    # top
    for i in range(mesh_points):
        curset_indices.append(i)

    # right
    for i in range(1, mesh_points):
        curset_indices.append(mesh_points * (i+1) - 1)

    # bottom
    for i in range(1, mesh_points):
        curset_indices.append(mesh_points**2 - 1 - i)

    # left
    for i in range(1, mesh_points):
        curset_indices.append(mesh_points**2 - (mesh_points * (i+1)))

    curset_xs = []
    curset_ys = []
                
    for i in curset_indices:
        curset_xs.append(all_sims[i][0, 0])
        curset_ys.append(all_sims[i][0, 1])

    curset_line = ax.plot(curset_xs, curset_ys, '-', lw=4.0, zorder=3)[0]
    curset_line.set_color('lime')
    curset_line.set_visible(False)

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

        curset_xs = []
        curset_ys = []

        for i in curset_indices:
            curset_xs.append(all_sims[i][f, 0])
            curset_ys.append(all_sims[i][f, 1])

        curset_line.set_data(curset_xs, curset_ys)

        return dots + lines + [time_text, curset_line]

    plt.tight_layout()
    
    interval = 5 # 10
    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, \
                                      interval=interval, blit=True, repeat=True)
    
    if make_mp4:
        writer = animation.writers['ffmpeg'](fps=100, metadata=dict(artist='Stanley Bak'), bitrate=1800) # was 1800

        #ax.xaxis.set_tick_params(labelsize=16)
        #ax.yaxis.set_tick_params(labelsize=16)

        filename = 'sim_linear.mp4'
        print(f"Making {filename}...")
        my_anim.save(filename, writer=writer)
    else:
        plt.show()
