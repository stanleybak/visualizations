'''sim_reachable using scaled dynamics for half the state space'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import numpy as np

from common import simulate, der_func

def sim_scaled(make_mp4=True, scale_func_str='quarter'):
    'sim scaled with the desired scale func'

    print(f"Running sim_scaled('{scale_func_str}')")

    filename = f"scaled_{scale_func_str}.mp4" if make_mp4 else None

    # map scale_func_str -> (scale_func, plot_extra_func, in_loop_func)
    func_map = {}

    func_map['none'] = get_noscale_funcs()
    func_map['line'] = get_line_funcs()
    func_map['quarter'] = get_quarter_funcs()
    func_map['onoff'] = get_onoff_funcs()
    func_map['two'] = get_two_funcs()

    scale_func, plot_extra_func, in_loop_func = func_map[scale_func_str]

    sim_scaled_generic(filename, scale_func, plot_extra_func, in_loop_func)

def get_noscale_funcs():
    'returns (scale_func, plot_extra_func, in_loop_func) for scaling="none"'

    scale_func = lambda _t, _state: 1.0
    plot_extra_func = lambda _ax: None
    in_loop_func = lambda _t: []

    return scale_func, plot_extra_func, in_loop_func

def get_two_funcs():
    'returns (scale_func, plot_extra_func, in_loop_func) for scaling="two"'

    scale_func = lambda _t, _state: 2.0
    plot_extra_func = lambda _ax: None
    in_loop_func = lambda _t: []

    return scale_func, plot_extra_func, in_loop_func

def get_quarter_funcs():
    'returns (scale_func, plot_extra_func, in_loop_func) for scaling="quarter"'

    # plot bounds
    rmin = -3
    rmax = 3

    in_loop_func = lambda _t: []

    scale_func = lambda _t, state: 0.5 if state[0] < 0 and state[1] < 0 else 1.0

    def plot_extra_func(ax):
        'extra plotting'

        # add patches
        patches = []
        width = rmax - rmin
        rect = mpatches.Rectangle((rmin, rmin), width/2, width/2)
        patches.append(rect)

        patch_collection = PatchCollection(patches, zorder=0, fc='lime', alpha=0.5)
        ax.add_collection(patch_collection)

    return scale_func, plot_extra_func, in_loop_func

def get_line_funcs():
    'returns (scale_func, plot_extra_func, in_loop_func) for scaling="line"'

    # plot bounds
    rmin = -3
    rmax = 3

    in_loop_func = lambda _t: []

    linex = -2.5

    # scale distance to x=-2.5 line
    scale_func = lambda _t, state: 0.8 * (state[0] - linex)

    def plot_extra_func(ax):
        'extra plotting'

        startx = linex
        endx = linex + 12 * 0.025 * 5

        start_alpha = 0.8
        end_alpha = 0.0
        step = 0.01

        curx = startx

        xs = [linex, linex]
        ys = [rmin, rmax]
        ax.plot(xs, ys, 'k--', lw=2)

        while curx < endx:
            frac = (endx - curx) / (endx - startx)
            # frac is 1.0 at start and 0.0 at end

            curx += step
            cur_alpha = start_alpha * frac + end_alpha * (1.0 - frac)

            # add patches
            patches = []
            rect = mpatches.Rectangle((curx, rmin), step, rmax - rmin)
            patches.append(rect)

            patch_collection = PatchCollection(patches, zorder=0, fc='lime', alpha=cur_alpha)
            ax.add_collection(patch_collection)

    return scale_func, plot_extra_func, in_loop_func

def get_onoff_funcs():
    'returns (scale_func, plot_extra_func, in_loop_func) for scaling="onoff"'

    # plot bounds
    rmin = -3
    rmax = 3

    changes_per_sec = 2

    def scale_func(t, state):
        'scale with time step'

        scale = 1.0

        if state[0] < 0 and state[1] < 0:
            state = int(t * changes_per_sec) % 2

            if state == 0:
                scale = 0.5
            else:
                scale = 1.0

        return scale

    pc_list = []

    def plot_extra_func(ax):
        'extra plotting'

        # add patches
        patches = []
        width = rmax - rmin
        rect = mpatches.Rectangle((rmin, rmin), width/2, width/2)
        patches.append(rect)

        patch_collection = PatchCollection(patches, zorder=0, fc='lime', alpha=0.5)
        pc_list.append(patch_collection)
        ax.add_collection(patch_collection)

    def in_loop_func(cur_time):
        'in anim loop func'

        state = int(cur_time * changes_per_sec) % 2

        if state == 0:
            pc_list[0].set_alpha(0.5)
        else:
            pc_list[0].set_alpha(0.0)

        return pc_list

    return scale_func, plot_extra_func, in_loop_func

def sim_scaled_generic(filename, scale_func, plot_extra_func, in_loop_func):
    'make simulation animations animations'

    #plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))

    tmax = 30.0
    step = 0.01

    init_box = [[0.3, 0.7], [0.3, 0.7]]
    mesh_points = 65 # points per row / column in initial set
    
    #ax.set_title("Vanderpol Oscillator")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    rmin = -3
    rmax = 3
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(rmin, rmax)

    plot_extra_func(ax)

    def stop_func(state_list):
        'stop sim condition, returns true or false'

        rv = False
        yindex = 1

        # stop if y just crossed < 2.0 and exists past state with y = 1.9
        if len(state_list) >= 2:
            if state_list[-1][yindex] < 2.0 <= state_list[-2][yindex]:
                for state_prev, state in zip(state_list[:-3], state_list[1:-2]):
                    if state[yindex] < 1.7 <= state_prev[yindex]:
                        rv = True
                        break

        return rv

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

    all_sims = []
    dots = []
    lines = []

    dx = (init_box[0][1] - init_box[0][0]) / (mesh_points - 1)
    dy = (init_box[1][1] - init_box[1][0]) / (mesh_points - 1)

    min_sim_len = np.inf

    for xindex in range(mesh_points):
        for yindex in range(mesh_points):
            x = init_box[0][0] + dx * xindex
            y = init_box[1][0] + dy * yindex

            x0 = np.array([x, y], dtype=float)

            print(f"Simulating {len(all_sims) + 1}/{mesh_points**2}... ", end='', flush=True)
            states = simulate(x0, tmax, step, scale_func=scale_func, stop_func=stop_func)
            print(f"{len(states)} steps")

            min_sim_len = min(min_sim_len, len(states))

            all_sims.append(states)

            line = ax.plot(states[:, 0], states[:, 1], '-', lw=3.0, zorder=1)[0]
            line.set_color('grey')
            lines.append(line)

            dot = ax.plot([states[0, 0]], [states[0, 1]], 'ro', markersize=3, zorder=2)[0]
            #dot.set_color('lime')
            #dot.set_visible(False)
            dots.append(dot)

    print(f"Trimming sims to min_sim_len={min_sim_len} ({min_sim_len * step} sec)")

    for i, sim in enumerate(all_sims):
        all_sims[i] = sim[:min_sim_len]

    assert all_sims[0].shape[0] == min_sim_len

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

    time_text = ax.text(0.05, 0.95, 'Time: 0', horizontalalignment='left', fontsize=14,
                             verticalalignment='top', transform=ax.transAxes)

    normal_frames = all_sims[0].shape[0]
    freeze_frames = 40 if normal_frames > 40 else 0
    num_frames = normal_frames + 2*freeze_frames

    time_text.set_visible(False)

    def animate(f):
        'animate function'

        if (f+1) % 10 == 0:
            print(f"Frame: {f+1} / {num_frames}")

        f = f - freeze_frames

        f = max(0, f)
        max_index = len(all_sims[0]) - 1
        f = min(f, max_index)

        cur_time = f * step
        time_text.set_text(f"Time: {cur_time:.2f}")

        for n, dot in enumerate(dots):
            states = all_sims[n]

            if f < states.shape[0]:
                dot.set_data([states[f, 0]], [states[f, 1]])

        for n, line in enumerate(lines):
            states = all_sims[n]

            i = min(states.shape[0], f+1)
                
            line.set_data([states[:i, 0]], [states[:i, 1]])

        rv = dots + lines + [time_text]

        rv += in_loop_func(cur_time)

        return rv

    plt.tight_layout()

    interval = 5 # 10
    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, \
                                      interval=interval, blit=True, repeat=True)
    
    if filename is not None:
        writer = animation.writers['ffmpeg'](fps=100, metadata=dict(artist='Stanley Bak'), bitrate=1800) # was 1800

        #ax.xaxis.set_tick_params(labelsize=16)
        #ax.yaxis.set_tick_params(labelsize=16)

        print(f"Making {filename}...")
        my_anim.save(filename, writer=writer)
    else:
        plt.show()
