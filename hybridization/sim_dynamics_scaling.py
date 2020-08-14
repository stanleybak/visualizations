'''sim_reachable using scaled dynamics for half the state space'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import numpy as np

from common import simulate_dynamics_scaled, der_func

def sim_dynamics_scaling(make_mp4=True):
    'make dynamics scaling animations'

    filename = 'dynamics_scaling.mp4' if make_mp4 else None
    print("sim_dynamics_scaling started")

    fig, ax = plt.subplots(figsize=(8, 8))

    tmax = 30.0
    step = 0.01 # 0.01

    init_box = [[0.3, 0.7], [0.3, 0.7]]
    mesh_points = 65 # points per row / column in initial set

    ##
    scale_distance = 0.5
    scale_factor = 2.0

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

    ##
    
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

    all_sims, scale_tuples, center_states = simulate_dynamics_scaled(center_state, scale_distance, scale_factor, x0_list, tmax, stop_func, step)

    assert center_states.shape == all_sims[0].shape, f"center_states.shape = {center_states.shape}, all_sims[0].shape = {all_sims[0].shape}"
    assert len(scale_tuples) == center_states.shape[0], f"len(scale_tuples) = {len(scale_tuples)}, all_sims[0].shape[0] = {all_sims[0].shape[0]}"

    # drawing elements
    dots = []
    lines = []

    center_dot = ax.plot([center_states[0, 0]], [center_states[0, 1]], 'bo', markersize=6, zorder=3)[0]
    center_dot.set_visible(False)

    pt = scale_tuples[0][0]
    forward_dot = ax.plot([pt[0]], [pt[1]], 'bo', markersize=6, zorder=3)[0]
    forward_dot.set_visible(False)

    vec = scale_tuples[0][1]
    dist = 9
    orth = dist * np.array([vec[1], -vec[0]], dtype=float)
    pt1 = pt + orth
    pt2 = pt - orth
    
    xs = [pt1[0], pt2[0]]
    ys = [pt1[1], pt2[1]]
    forward_line = ax.plot(xs, ys, 'k--', lw=2)[0]

    ##
    fac = 4
    green_step = 0.1 / fac
    green_num = 6 * fac
    start_alpha = 0.8
    end_alpha = 0.0
    
    patches = []
    #patch_collections = []

    last_pt = pt
    # assume last_pt, pt1 and pt2 are already defined as loop precondition
    
    for step in range(1, green_num+1):
        new_pt = last_pt + vec * green_step

        pt4 = new_pt + orth
        pt3 = new_pt - orth

        xs = [pt1[0], pt2[0], pt3[0], pt4[0]]
        ys = [pt1[1], pt2[1], pt3[1], pt4[1]]

        frac = step / green_num
        cur_alpha = end_alpha * frac + start_alpha * (1.0 - frac)

        xy = np.array([xs, ys], dtype=float).transpose()

        poly = mpatches.Polygon(xy, fc='lime', alpha=cur_alpha, zorder=0)
        patches.append(poly)
        ax.add_patch(poly)

        #pc = PatchCollection([poly], zorder=0, fc='lime', alpha=cur_alpha)
        #ax.add_collection(pc)
        #patch_collections.append(pc)

        # update for next iteration
        pt1 = pt4
        pt2 = pt3
        last_pt = new_pt

    ##
    
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

    freeze_frames = 40
    num_frames = states.shape[0] + 2*freeze_frames

    def animate(f):
        'animate function'

        if (f+1) % 10 == 0:
            print(f"Frame: {f+1} / {num_frames}")

        f = f - freeze_frames

        f = max(0, f)
        max_index = len(all_sims[0]) - 1
        f = min(f, max_index)

        center_dot.set_data([center_states[f, 0]], [center_states[f, 1]])

        pt = scale_tuples[f][0]
        forward_dot.set_data([pt[0]], [pt[1]])

        vec = scale_tuples[f][1]
        orth = 9 * np.array([vec[1], -vec[0]], dtype=float)
        pt1 = pt + orth
        pt2 = pt - orth

        xs = [pt1[0], pt2[0]]
        ys = [pt1[1], pt2[1]]
        forward_line.set_data(xs, ys)

        ########
        # patch collections
        last_pt = pt
        # assume last_pt, pt1 and pt2 are already defined as loop precondition

        for step in range(1, green_num+1):
            new_pt = last_pt + vec * green_step

            pt4 = new_pt + orth
            pt3 = new_pt - orth

            xs = [pt1[0], pt2[0], pt3[0], pt4[0]]
            ys = [pt1[1], pt2[1], pt3[1], pt4[1]]

            xy = np.array([xs, ys], dtype=float).transpose()

            patches[step-1].set_xy(xy)
            #poly = mpatches.Polygon(xy)
            #patches.append(poly)

            #pc = PatchCollection([poly], zorder=0, fc='lime', alpha=cur_alpha)
            #ax.add_collection(pc)
            #patch_collections.append(pc)

            # update for next iteration
            pt1 = pt4
            pt2 = pt3
            last_pt = new_pt

        ########

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

        return dots + lines + patches + [curset_line, center_dot, forward_dot, forward_line]

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
