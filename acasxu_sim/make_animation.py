'''
Simulate the flocking neural network
'''

import random
import math
import sys

import numpy as np
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from scipy import ndimage

import matplotlib
matplotlib.use('TkAgg') # set backend

from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D

from network import FullyConnectedLayer, weights_biases_to_nn

def load_networks():
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''

    nets = []

    for net in range(1, 6):
        filename = f"ACASXU_run2a_{net}_1_batch_2000.mat"

        #print(f"Loading {filename}...")
        matfile = loadmat(filename)

        weights = matfile['W'][0]
        biases = matfile['b'][0]

        for b in biases:
            b.shape = (len(b),)

        range_for_scaling = matfile['range_for_scaling'][0]
        means_for_scaling = matfile['means_for_scaling'][0]

        network = weights_biases_to_nn(weights, biases)

        scale_outputs = False

        if scale_outputs:
            # add one more layer to the network for the output scaling
            output_mult, output_bias = range_for_scaling[-1], means_for_scaling[-1]
            output_mult_mat = np.identity(5) * output_mult
            output_bias_vec = np.array([output_bias] * 5, dtype=float)
            layer_num = len(network.layers)
            network.layers.append(FullyConnectedLayer(layer_num, output_mult_mat, output_bias_vec, False))

        nets.append([network, range_for_scaling, means_for_scaling])

    return nets

def run_network(network_tuple, x, stdout=False):
    'run the network and return the output'

    net, range_for_scaling, means_for_scaling = network_tuple

    # normalize input
    for i in range(5):
        x[i] = (x[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"input (after scaling): {x}")
        
    return net.execute(x)

class State():
    'state of execution container'

    nets = load_networks()
    img = plt.imread('airplane.png')

    plane_size = 1500

    def __init__(self):

        init_vec = [0, 0, math.pi/2, 0, 25000, -math.pi/2, 0]
        
        self.vec = np.array(init_vec, dtype=float) # copy
        self.time = 0
        self.vec_list = None # set when simulation() is called
        self.artists_dict = {} # set when make_artists is called

        # the network runs at some update rate
        self.nn_update_rate = 2.0
                
        self.next_nn_update = 0

        self.command = 0
        self.commands = [] # commands history
        self.vel = 807 # ft/sec ?

    def randomize(self):
        'randomize the initial state (for batch simulation)'

        rad = 5000
        self.vec[0] += random.random() * 2*rad - rad

        y_rad = 200
        self.vec[1] += random.random() * 2*y_rad - y_rad

    def artists_list(self):
        'return list of artists'

        return list(self.artists_dict.values())

    def set_plane_visible(self, vis):
        'set ownship plane visibility status'

        self.artists_dict['dot0'].set_visible(not vis)
        self.artists_dict['circle0'].set_visible(False) # circle always False
        self.artists_dict['lc0'].set_visible(True)
        self.artists_dict['plane0'].set_visible(vis)
        
    def update_artists(self, axes):
        '''update artists in self.artists_dict to be consistant with self.vec, returns a list of artists'''

        assert self.artists_dict
        rv = []

        x1, y1, theta1, x2, y2, theta2, _ = self.vec

        for i, x, y, theta in zip([0, 1], [x1, x2], [y1, y2], [theta1, theta2]):
            key = f'plane{i}'

            if key in self.artists_dict:
                plane = self.artists_dict[key]
                rv.append(plane)

                if plane.get_visible():
                    theta_deg = (theta - math.pi / 2) / math.pi * 180 # original image is facing up, not right
                    original_size = list(State.img.shape)
                    img_rotated = ndimage.rotate(State.img, theta_deg, order=1)
                    rotated_size = list(img_rotated.shape)
                    ratios = [r / o for r, o in zip(rotated_size, original_size)]
                    plane.set_data(img_rotated)

                    size = State.plane_size
                    width = size * ratios[0]
                    height = size * ratios[1]
                    box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
                    tbox = TransformedBbox(box, axes.transData)
                    plane.bbox = tbox

            key = f'dot{i}'
            if key in self.artists_dict:
                dot = self.artists_dict[f'dot{i}']
                cir = self.artists_dict[f'circle{i}']
                rv += [dot, cir]

                dot.set_data([x], [y])
                cir.set_center((x, y))

        # line collection
        lc = self.artists_dict['lc0']
        rv.append(lc)

        self.update_lc_artist(lc)

        return rv

    def update_lc_artist(self, lc):
        'update line collection artist based on current state'
        
        paths = lc.get_paths()
        colors = []
        lws = []
        paths.clear()
        last_command = -1
        codes = []
        verts = []

        for i, vec in enumerate(self.vec_list):
            if np.linalg.norm(vec - self.vec) < 1e-6:
                # done
                break

            cmd = self.commands[i]

            # command[i] is the line from i to (i+1)
            if cmd != last_command:
                if codes:
                    paths.append(Path(verts, codes))

                codes = [Path.MOVETO]
                verts = [(vec[0], vec[1])]
                
                if cmd == 1: # weak left
                    lws.append(2)
                    colors.append('b')
                elif cmd == 2: # weak right
                    lws.append(2)
                    colors.append('c')
                elif cmd == 3: # strong left
                    lws.append(2)
                    colors.append('g')
                elif cmd == 4: # strong right
                    lws.append(2)
                    colors.append('r')
                else:
                    assert cmd == 0 # coc
                    lws.append(2)
                    colors.append('k')

            codes.append(Path.LINETO)
            verts.append((self.vec_list[i+1][0], self.vec_list[i+1][1]))

        # add last one
        if codes:
            paths.append(Path(verts, codes))

        lc.set_lw(lws)
        lc.set_color(colors)

    def make_artists(self, axes, show_intruder):
        'make self.artists_dict'

        assert self.vec_list is not None

        posa_list = [(v[0], v[1], v[2]) for v in self.vec_list]
        posb_list = [(v[3], v[4], v[5]) for v in self.vec_list]
        
        pos_lists = [posa_list, posb_list]

        if show_intruder:
            pos_lists.append(posb_list)

        for i, pos_list in enumerate(pos_lists):
            x, y, theta = pos_list[0]
            
            l = axes.plot(*zip(*pos_list), f'c-', lw=0, zorder=1)[0]
            l.set_visible(False)
            self.artists_dict[f'line{i}'] = l

            if i == 0:
                lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(lc)
                self.artists_dict[f'lc{i}'] = lc

            # only sim_index = 0 gets intruder aircraft
            if i == 0 or (i == 1 and show_intruder):
                size = State.plane_size
                box = Bbox.from_bounds(x - size/2, y - size/2, size, size)
                tbox = TransformedBbox(box, axes.transData)
                box_image = BboxImage(tbox, zorder=2)

                theta_deg = (theta - math.pi / 2) / math.pi * 180 # original image is facing up, not right
                img_rotated = ndimage.rotate(State.img, theta_deg, order=1)

                box_image.set_data(img_rotated)
                axes.add_artist(box_image)
                self.artists_dict[f'plane{i}'] = box_image

            if i == 0:
                dot = axes.plot([x], [y], f'k.', markersize=6.0, zorder=2)[0]
                self.artists_dict[f'dot{i}'] = dot

                rad = 1500
                c = patches.Ellipse((x, y), rad, rad, color='k', lw=3.0, fill=False)
                axes.add_patch(c)
                self.artists_dict[f'circle{i}'] = c

    def step(self, dt):
        'execute one time step and update the model'

        tol = 1e-6

        if self.next_nn_update < tol:
            assert abs(self.next_nn_update) < tol, f"time step doesn't sync with nn update time. " + \
                                                   f"next update: {self.next_nn_update}"
            # update command
            self.update_command()

            self.next_nn_update = self.nn_update_rate

        self.next_nn_update -= dt
        self.commands.append(self.command)

        #x1, y1, theta1, x2, y2, theta2, t = self.vec
        
        if self.command == 0: # coc
            y = 0
        elif self.command == 1: # weak left
            y = 1.5
        elif self.command == 2: # weak right
            y = -1.5
        elif self.command == 3: # strong left
            y = 3.0
        else: # strong right
            assert self.command == 4
            y = -3.0

        dtheta1 = (y / 180 * math.pi)

        def der(_t, y):
            'get continuous-time derivative'

            _, _, theta1, _, _, theta2, _ = y

            dx1 = self.vel * math.cos(theta1)
            dy1 = self.vel * math.sin(theta1)

            dtheta2 = 0
            dx2 = self.vel * math.cos(theta2)
            dy2 = self.vel * math.sin(theta2)

            return dx1, dy1, dtheta1, dx2, dy2, dtheta2, 1

        res = solve_ivp(der, [0, dt], self.vec, t_eval=[dt])
        assert res.success
        
        new_state = np.ravel(res.y)

        self.vec = new_state
        self.time += dt

    def simulate(self, dt, tmax):
        '''simulate system

        saves result in self.vec_list
        '''

        t = 0
        rv = [self.vec.copy()]

        while t + 1e-6 < tmax:
            self.step(dt)

            rv.append(self.vec.copy())

            t += dt

        self.vec_list = rv

    def update_command(self):
        'update command based on current state'''

        x1, y1, theta1, x2, y2, theta2, _ = self.vec

        rho = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        dy = y2 - y1
        dx = x2 - x1

        theta = math.atan2(dy, dx)
        psi = theta2 - theta1
        v_own = v_int = self.vel

        theta -= theta1

        while theta < -math.pi:
            theta += 2 * math.pi

        while theta > math.pi:
            theta -= 2 * math.pi

        if psi < -math.pi:
            psi += 2 * math.pi

        while psi > math.pi:
            psi -= 2 * math.pi

        # 0: rho, distance
        # 1: theta, angle to intruder relative to ownship heading
        # 2: psi, heading of intruder relative to ownship heading
        # 3: v_own, speed of ownship
        # 4: v_int, speed in intruder

        # min inputs: 0, -3.1415, -3.1415, 100, 0
        # max inputs: 60760, 3.1415, 3,1415, 1200, 1200

        stdout = False #self.time < 0.5

        if stdout:
            print(f"\nstate at time {self.time}, x1: {x1}, y1: {y1}, " + \
              f"heading1: {theta1}, x2: {x2}, y2: {y2}, heading2: {theta2}")

            print(f"input (before scaling): rho: {rho}, theta: {theta}, psi: {psi}, v_own: {v_own}, v_int: {v_int}")

        if rho > 60760:
            self.command = 0
        else:
            last_command = self.command

            net = State.nets[last_command]

            state = [rho, theta, psi, v_own, v_int]

            res = run_network(net, state, stdout)
            self.command = np.argmin(res)

            names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

            if stdout:
                print(f".{self.time:.2f}: unscaled network output ({names[self.command]}): {res}")

def main():
    'main entry point'

    random.seed(0) # deterministic random numbers

    plt.style.use('bmh')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['path.simplify'] = False

    # decrease dt to make it more smooth
    dt = 0.1
    tmax = 20

    # number of aircraft in pass 3 of the animation
    count = 100 if len(sys.argv) < 2 else int(sys.argv[1])

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    axes.axis('equal')

    axes.set_title("ACAS Xu Simulations")
    axes.set_xlabel(f'X Position (ft)')
    axes.set_ylabel(f'Y Position (ft)')

    time_text = axes.text(0.02, 0.98, 'Time: 0', horizontalalignment='left', fontsize=14,
                          verticalalignment='top', transform=axes.transAxes)
    time_text.set_visible(True)

    states = []

    custom_lines = [Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='c', lw=2),
                    Line2D([0], [0], color='r', lw=2)]

    axes.legend(custom_lines, ['Strong Left', 'Weak Left', 'Clear of Conflict', 'Weak Right', 'Strong Right'], \
                fontsize=14, loc='lower left')

    
    for i in range(count):
        print(f"Simulating {i+1}/{count}")
        s = State()

        if i > 0:
            s.randomize()
            
        states.append(s)
        # todo: add random perturbation

        s.simulate(dt, tmax)
        show_intruder = (i == 0)
        s.make_artists(axes, show_intruder)

    plt.tight_layout()

    save_mp4 = True # False = plot to screen

    if save_mp4:
        print("Saving to MP4 (slow)... you can plot to screen by settings the save_mp4 variable to False.")
    else:
        print("Plotting to screen... you can save_mp4 by setting the save_mp4 variable to True.")

    num_steps = len(states[0].vec_list)
    interval = 20 if save_mp4 else 4
    freeze_frames = 10 if not save_mp4 else 80

    num_runs = 3
    num_frames = num_runs * num_steps + 2 * num_runs * freeze_frames

    def animate(f):
        'animate function single frame function'

        if (f+1) % 10 == 0:
            print(f"Frame: {f+1} / {num_frames}")

        run_index = f // (num_steps + 2 * freeze_frames)

        f = f - run_index * (num_steps + 2*freeze_frames)

        f -= freeze_frames

        if f < 0:
            f = 0

        # post anim freeze
        if f >= num_steps:
            f = num_steps - 1

        if run_index == 0:
            num_states = 1
        elif run_index == 1:
            num_states = 10
        else:
            num_states = len(states)

        if f == 0:
            # initiaze current run_index
            show_plane = num_states <= 10
            for s in states[:num_states]:
                s.set_plane_visible(show_plane)

            for s in states[num_states:]:
                for a in s.artists_list():
                    a.set_visible(False)

        time_text.set_text(f'Time: {f * dt:.1f}')

        artists = [time_text]

        for s in states[:num_states]:
            s.vec = s.vec_list[f]
            artists += s.update_artists(axes)

        for s in states[num_states:]:
            artists += s.artists_list()

        return artists

    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True, repeat=True)

    if save_mp4:
        writer = animation.writers['ffmpeg'](fps=50, metadata=dict(artist='Stanley Bak'), bitrate=1800)

        my_anim.save('acasxu_anim.mp4', writer=writer)
    else:
        plt.show()

if __name__ == "__main__":
    main()
