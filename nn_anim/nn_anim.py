'''
Neural Network Execution / Verification Animation

Stanley Bak, Jan 2020
'''

import random
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches

from acasxu import load_init_network

#
# We keep chaning the projection, which is bad. If we don't though, the points get projected to zero and the
# animation looks bad.
#

def proj(p, layer, is_x):
    '''return the projection of a point

    different layer are projections onto different dimensions, as it makes a better visualization
    '''

    xdim = 3
    ydim = 4

    if layer == 0:
        xdim = 0
        ydim = 28
    elif layer == 1:
        xdim = 43
        ydim = 45
    elif layer == 2:
        xdim = 15
        ydim = 19
    elif layer == 3:
        xdim = 9
        ydim = 23
    elif layer == 4:
        xdim = 32
        ydim = 6
    elif layer == 5:
        xdim = 27
        ydim = 21

    rv = p[xdim] if is_x else p[ydim]

    # zoom in slightly
    if layer > 5:
        if is_x:
            rv *= 18
        else:
            rv *= 12
    elif layer == -1:
        rv *= 3
    elif layer in [1, 2]:
        if is_x:
            rv *= 3

    return rv

def main():
    'main entry point'

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['path.simplify'] = False

    init_box, nn, _ = load_init_network((5, 1), "3")
    #print("Warning: trimming nn to less layers")
    #nn.layers = nn.layers[:1]
    
    dims = len(init_box)
    assert dims == 5

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    axes.set_xlim(-0.2, 4.5)
    axes.set_ylim(-0.2, 3)

    frame_text = axes.text(0.05, 0.95, 'Frame: 0', horizontalalignment='left', fontsize=14,
                           verticalalignment='top', transform=axes.transAxes)
    frame_text.set_visible(False)

    layer_text = axes.text(0.0, 0.0, 'Init', horizontalalignment='center', fontsize=14, color='blue',
                           verticalalignment='bottom')

    layer_text.set_visible(False)

    random.seed(1) # determinstic random numbers

    npoints = 500

    # in each layer
    all_points = []
    init = []
    all_points.append(init)

    dots = []

    label_positions = []
    min_x = np.inf
    max_x = -np.inf
    max_y = -np.inf

    for _ in range(npoints):
        p = np.zeros((dims, ), dtype=float)

        for i, (lb, ub) in enumerate(init_box):
            p[i] = lb + random.random() * (ub - lb)

        init.append(p)
        x = proj(p, -1, True)
        y = proj(p, -1, False)

        min_x = min(min_x, x)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

        dot = axes.plot([x], [y], 'b.', markersize=3.0)[0]
        axes.plot([x], [y], 'r.', markersize=0.5, zorder=0.5)
        dots.append(dot)

    label_positions.append((min_x + (max_x - min_x) / 2, max_y))

    axes.set_title("Execution of ACASXu Neural Network 5-1, Property 3")
    axes.set_xlabel(f'Projection 1')
    axes.set_ylabel(f'Projection 2')

    # worst-case point
    wpt = np.array([-0.30353116, 0.0095493, 0.49338032, 0.5, 0.4184347])
    wpts = [wpt]

    for i, layer in enumerate(nn.layers):
        print(f"executing layer {i}...")
        layer_list = []
        min_x = np.inf
        max_x = -np.inf
        max_y = -np.inf

        wpts.append(layer.execute(wpts[-1]))

        #if i > 0:
        #    print("got past layer 1, exiting")
        #    exit(1)

        for p in all_points[-1]:
            p = layer.execute(p)

            layer_list.append(p)

            y = proj(p, i, False)
            x = proj(p, i, True)
                           
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

            axes.plot([x], [y], 'r.', markersize=0.5, zorder=0.5)

        if min_x < 0.5:
            min_x = 0.5 # push label to the right a little
            
        label_positions.append((min_x + (max_x - min_x) / 2, max_y))
        all_points.append(layer_list)

    # set position of desired output text
    x = proj(wpts[-1], len(nn.layers), True)
    y = proj(wpts[-1], len(nn.layers), False)

    desired_output_text = axes.text(x + 0.1, y, 'Desired\nOutput', horizontalalignment='left', fontsize=14, color='g',
                                    verticalalignment='center')
    desired_output_text.set_visible(False)

    desired_circle = plt.Circle((x, y), 0.07, color='g', fill=False)
    axes.add_artist(desired_circle)
    desired_circle.set_visible(False)

    x = proj(wpts[0], -1, True)
    y = proj(wpts[0], -1, False)

    generated_input_text = axes.text(x + 0.1, y, 'Generated\nInput', horizontalalignment='left', fontsize=14,
                                     color='g', verticalalignment='center')
    generated_input_text.set_visible(False)

    wdot = axes.plot([x], [y], 'go', markersize=5.0)[0]
    wdot.set_visible(False)

    ## reach sets
    reach_verts = []
    
    # manually add init
    x0, x1 = [3 * v for v in init_box[3]]
    y0, y1 = [3 * v for v in init_box[4]]

    init_verts = [[[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]]
    reach_verts.append(init_verts)

    # load others
    print("loading reach sets")
    with gzip.open('reach_set_verts.pickle.gz', 'rb') as f:
        other_layers = pickle.load(f)

        for layer in other_layers:
            reach_verts.append(layer)

    print("loaded")

    #colors = ['0.8', 'r', 'g', 'b', 'y', '0.5', 'k']
    colors = ['0.6']
    polys = []

    for i, verts in enumerate(reach_verts):
        col = colors[i % len(colors)]
                
        poly = PolyCollection(verts, facecolor=col, zorder=1)
        polys.append(poly)
        axes.add_collection(poly)
        poly.set_visible(False)

        #for pts in verts:
        #    axes.add_line(Line2D([pt[0] for pt in pts], [pt[1] for pt in pts], color=col,
        #                         marker='.', ms=1, mec=col, solid_capstyle='projecting', zorder=2))

    frames_per_tick = 50
    half_tick = frames_per_tick // 2
    
    num_frames = 2*frames_per_tick * len(all_points)
    num_frames += frames_per_tick # extra pause at start
    done_sim_frames = num_frames

    num_frames += frames_per_tick # show desired output
    num_frames += frames_per_tick # show generatred input
    start_wc_sim = num_frames

    num_frames += half_tick * len(all_points) # no pauses generatred input
    num_frames += frames_per_tick # extra pause at the end

    # legend
    r = patches.Rectangle((-10, -10), 1, 1, facecolor='0.6', label='Computed Output Set')
    axes.add_patch(r)

    dot = axes.plot([-10], [-10], 'b.', markersize=5.0, label='Random Executions')[0]

    plt.legend()
    
    def animate(f):
        'animate function'

        if f % frames_per_tick == 0:
            print(f"tick {f // frames_per_tick} / {num_frames // frames_per_tick}")
        else:
            print(".", end='', flush=True)

        if f == 0:
            for poly in polys:
                poly.set_visible(False)

            desired_output_text.set_visible(False)
            desired_circle.set_visible(False)
            generated_input_text.set_visible(False) # FALSE
            wdot.set_visible(False)

        if f > done_sim_frames:
            desired_output_text.set_visible(True)
            desired_circle.set_visible(True)

        if done_sim_frames + frames_per_tick < f < start_wc_sim:
            generated_input_text.set_visible(True)
            wdot.set_visible(True)

        if f > start_wc_sim:
            wdot.set_visible(True)
            generated_input_text.set_visible(False)

            # animate wdot
            frac = (f - start_wc_sim) / half_tick

            cur_layer = 0
            
            while frac > 1:
                frac -= 1
                cur_layer += 1

            if cur_layer + 1 >= len(wpts):
                # frozen at end
                x = proj(wpts[-1], len(nn.layers), True)
                y = proj(wpts[-1], len(nn.layers), False)

                wdot.set_data([x], [y])
            else:
                p_prev = wpts[cur_layer]
                p_next = wpts[cur_layer + 1]
                xfrac, yfrac = interpolate(p_prev, cur_layer-1, p_next, frac)
                
                wdot.set_data([xfrac], [yfrac])

        f -= frames_per_tick # extra pause at start

        label = f"Frame: {f}"

        freeze = True
        cur_index = 0

        while f > frames_per_tick:
            f -= frames_per_tick

            freeze = not freeze

            if freeze:
                cur_index += 1

        if cur_index >= len(all_points) - 1: # past the end
            freeze = True
            cur_index = len(all_points) - 1

        label += f" cur_index={cur_index} freeze={freeze}"
        #print(label)
        #frame_text.set_text(label)
            
        if freeze:
            cur_layer = all_points[cur_index]
            polys[cur_index].set_visible(True)
            layer_index = cur_index-1

            if layer_index == -1:
                label = "Initial Points"
            elif layer_index == 6:
                label = 'After Output Layer'
            else:
                label = f'After Layer #{layer_index+1}'

            layer_text.set_visible(True)
            layer_text.set_text(label)
            x, y = label_positions[cur_index]
            layer_text.set_x(x)
            layer_text.set_y(y + 0.02)

            for p, dot in zip(cur_layer, dots):
                dot.set_data([proj(p, layer_index, True)], [proj(p, layer_index, False)])
                dot.set_color('blue')
        else:
            layer_text.set_visible(False)
                        
            # interpolating
            frac = f / frames_per_tick

            polys[cur_index].set_visible(False)
            fprev = cur_index
            fnext = cur_index + 1
            
            prev_layer = all_points[fprev]
            next_layer = all_points[fnext]

            for p_prev, p_next, dot in zip(prev_layer, next_layer, dots):
                xfrac, yfrac = interpolate(p_prev, fprev-1, p_next, frac)
                
                dot.set_data([xfrac], [yfrac])
                dot.set_color('black')

        return dots + polys + [frame_text, layer_text, wdot, desired_output_text, desired_circle,
                               generated_input_text, wdot]

    save_mp4 = True

    #print("mp4 hardcoded frames limit")
    #num_frames = 500

    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, \
                                      interval=10, blit=True, repeat=True)

    if save_mp4:
        writer = animation.writers['ffmpeg'](fps=30, metadata=dict(artist='Stanley Bak'), bitrate=1800)

        my_anim.save('nn_anim.mp4', writer=writer)
    else:
        plt.show()

def interpolate(p_prev, layer_prev, p_next, frac):
    'returns interpolated x and y'

    x0 = proj(p_prev, layer_prev, True)
    y0 = proj(p_prev, layer_prev, False)

    x1 = proj(p_next, layer_prev+1, True)
    y1 = proj(p_next, layer_prev+1, False)

    xfrac = (1 - frac) * x0 + frac * x1
    yfrac = (1 - frac) * y0 + frac * y1

    return xfrac, yfrac
        
if __name__ == "__main__":
    main()
