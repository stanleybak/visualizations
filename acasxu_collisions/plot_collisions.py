'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used for falsification, where the opponent is allowed to maneuver over time
'''

from functools import lru_cache
import time
import math
import argparse

import numpy as np
from scipy import ndimage
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D

import onnxruntime as ort
from numba import njit

def init_plot():
    'initialize plotting style'

    #matplotlib.use('TkAgg') # set backend

    p = 'bak_matplotlib.mlpstyle'
    plt.style.use(['bmh', p])

def load_network(last_cmd):
    '''load the one neural network as a 2-tuple (range_for_scaling, means_for_scaling)'''

    onnx_filename = f"ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"

    #print(f"Loading {mat_filename}...")
    #matfile = loadmat(mat_filename)
    #range_for_scaling = matfile['range_for_scaling'][0]
    #means_for_scaling = matfile['means_for_scaling'][0]
    #mat_filename = f"ACASXU_run2a_1_1_batch_2000.mat"

    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    session = ort.InferenceSession(onnx_filename)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session, range_for_scaling, means_for_scaling

def load_networks():
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''

    nets = []

    for last_cmd in range(5):
        nets.append(load_network(last_cmd))

    return nets


'Lru Cache.'
def get_time_elapse_mat(command1, dt, command2=0):
    '''get the matrix exponential for the given command

    state: x, y, vx, vy, x2, y2, vx2, vy2
    '''

    y_list = [0.0, 1.5, -1.5, 3.0, -3.0]
    y1 = y_list[command1]
    y2 = y_list[command2]

    dtheta1 = (y1 / 180 * np.pi)
    dtheta2 = (y2 / 180 * np.pi)

    a_mat = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0], # x' = vx
        [0, 0, 0, 1, 0, 0, 0, 0], # y' = vy
        [0, 0, 0, -dtheta1, 0, 0, 0, 0], # vx' = -vy * dtheta1
        [0, 0, dtheta1, 0, 0, 0, 0, 0], # vy' = vx * dtheta1
    #
        [0, 0, 0, 0, 0, 0, 1, 0], # x' = vx
        [0, 0, 0, 0, 0, 0, 0, 1], # y' = vy
        [0, 0, 0, 0, 0, 0, 0, -dtheta2], # vx' = -vy * dtheta2
        [0, 0, 0, 0, 0, 0, dtheta2, 0], # vy' = vx * dtheta1
        ], dtype=float)

    return expm(a_mat * dt)

def run_network(network_tuple, x, stdout=False):
    'run the network and return the output'

    session, range_for_scaling, means_for_scaling = network_tuple

    # normalize input
    for i in range(5):
        x[i] = (x[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"input (after scaling): {x}")

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, 1, 1, 5)
    outputs = session.run(None, {'input': in_array})

    return outputs[0][0]

@njit(cache=True)
def state7_to_state5(state7, v_own, v_int):
    """compute rho, theta, psi from state7"""

    assert len(state7) == 7

    x1, y1, theta1, x2, y2, theta2, _ = state7

    rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    dy = y2 - y1
    dx = x2 - x1

    theta = np.arctan2(dy, dx)
    psi = theta2 - theta1

    theta -= theta1

    while theta < -np.pi:
        theta += 2 * np.pi

    while theta > np.pi:
        theta -= 2 * np.pi

    if psi < -np.pi:
        psi += 2 * np.pi

    while psi > np.pi:
        psi -= 2 * np.pi

    return np.array([rho, theta, psi, v_own, v_int])

@njit(cache=True)
def state7_to_state8(state7, v_own, v_int):
    """compute x,y, vx, vy, x2, y2, vx2, vy2 from state7"""

    assert len(state7) == 7

    x1 = state7[0]
    y1 = state7[1]
    vx1 = math.cos(state7[2]) * v_own
    vy1 = math.sin(state7[2]) * v_own

    x2 = state7[3]
    y2 = state7[4]
    vx2 = math.cos(state7[5]) * v_int
    vy2 = math.sin(state7[5]) * v_int

    return np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2])

@lru_cache(maxsize=None)
def get_airplane_img():
    """load airplane image form file"""

    img = plt.imread('airplane.png')

    return img

def init_time_elapse_mats(dt):
    """get value of time_elapse_mats array"""

    rv = []

    for cmd in range(5):
        rv.append([])

        for int_cmd in range(5):
            mat = get_time_elapse_mat(cmd, dt, int_cmd)
            rv[-1].append(mat)

    return rv

@njit(cache=True)
def step_state(state7, v_own, v_int, time_elapse_mat, dt):
    """perform one time step with the given commands"""

    state8_vec = state7_to_state8(state7, v_own, v_int)

    s = time_elapse_mat @ state8_vec

    # extract observation (like theta) from state
    new_time = state7[-1] + dt
    theta1 = math.atan2(s[3], s[2])
    theta2 = math.atan2(s[7], s[6])
    rv = np.array([s[0], s[1], theta1, s[4], s[5], theta2, new_time])

    return rv

class State():
    'state of execution container'

    nets = load_networks()
    plane_size = 4000

    nn_update_rate = 1.0 # todo: make this a parameter
    dt = 1.0

    'Valid range" [100, 1145]'
    #v_own = 800 # ft/sec

    'Valid range: [60,1145]'
    #v_int = 500

    time_elapse_mats = init_time_elapse_mats(dt)

    def __init__(self, init_vec, v_own=800, v_int=500, save_states=False):
        assert len(init_vec) == 7, "init vec should have length 7"

        self.vec = np.array(init_vec, dtype=float) # current state
        self.next_nn_update = 0
        self.command = 0 # initial command
        self.v_own = v_own
        self.v_int = v_int

        # these are set when simulation() if save_states=True
        self.save_states = save_states
        self.vec_list = [] # state history
        self.commands = [] # commands history
        self.int_commands = [] # intruder command history

        # used only if plotting
        self.artists_dict = {} # set when make_artists is called
        self.img = None # assigned if plotting

        # assigned by simulate()
        self.u_list = []
        self.u_list_index = None
        self.min_dist = np.inf

    def __str__(self):
        x1, y1, _theta1, x2, y2, _theta2, _ = self.vec
        rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
        return f'State(v_own: {self.v_own}, v_int: {self.v_int}, rho: {rho})'

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
                    theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                    original_size = list(self.img.shape)
                    img_rotated = ndimage.rotate(self.img, theta_deg, order=1)
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

        int_lc = self.artists_dict['int_lc0']
        rv.append(int_lc)

        self.update_lc_artists(lc, int_lc)

        return rv

    def update_lc_artists(self, own_lc, int_lc):
        'update line collection artist based on current state'

        assert self.vec_list

        for lc_index, lc in enumerate([own_lc, int_lc]):
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

                if lc_index == 0:
                    cmd = self.commands[i]
                else:
                    cmd = self.int_commands[i]

                x = 0 if lc_index == 0 else 3
                y = 1 if lc_index == 0 else 4

                # command[i] is the line from i to (i+1)
                if cmd != last_command:
                    if codes:
                        paths.append(Path(verts, codes))

                    codes = [Path.MOVETO]
                    verts = [(vec[x], vec[y])]

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

                verts.append((self.vec_list[i+1][x], self.vec_list[i+1][y]))

            # add last one
            if codes:
                paths.append(Path(verts, codes))

            lc.set_lw(lws)
            lc.set_color(colors)

    def make_artists(self, axes, show_intruder):
        'make self.artists_dict'

        assert self.vec_list
        self.img = get_airplane_img()

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

                int_lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(int_lc)
                self.artists_dict[f'int_lc{i}'] = int_lc

            # only sim_index = 0 gets intruder aircraft
            if i == 0 or (i == 1 and show_intruder):
                size = State.plane_size
                box = Bbox.from_bounds(x - size/2, y - size/2, size, size)
                tbox = TransformedBbox(box, axes.transData)
                box_image = BboxImage(tbox, zorder=2)

                theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                img_rotated = ndimage.rotate(self.img, theta_deg, order=1)

                box_image.set_data(img_rotated)
                axes.add_artist(box_image)
                self.artists_dict[f'plane{i}'] = box_image

            if i == 0:
                dot = axes.plot([x], [y], 'k.', markersize=6.0, zorder=2)[0]
                self.artists_dict[f'dot{i}'] = dot

                rad = 1500
                c = patches.Ellipse((x, y), rad, rad, color='k', lw=3.0, fill=False)
                axes.add_patch(c)
                self.artists_dict[f'circle{i}'] = c

    def step(self):
        'execute one time step and update the model'

        tol = 1e-6

        if self.next_nn_update < tol:
            assert abs(self.next_nn_update) < tol, f"time step doesn't sync with nn update time. " + \
                      f"next update: {self.next_nn_update}"

            # update command
            self.update_command()

            self.next_nn_update = State.nn_update_rate

        self.next_nn_update -= State.dt
        intruder_cmd = self.u_list[self.u_list_index]

        if self.save_states:
            self.commands.append(self.command)
            self.int_commands.append(intruder_cmd)

        time_elapse_mat = State.time_elapse_mats[self.command][intruder_cmd] #get_time_elapse_mat(self.command, State.dt, intruder_cmd)

        self.vec = step_state(self.vec, self.v_own, self.v_int, time_elapse_mat, State.dt)

    def simulate(self, cmd_list):
        '''simulate system

        saves result in self.vec_list
        also saves self.min_dist
        '''

        self.u_list = cmd_list
        self.u_list_index = None

        assert isinstance(cmd_list, list)
        tmax = len(cmd_list) * State.nn_update_rate

        t = 0.0

        if self.save_states:
            rv = [self.vec.copy()]

        #self.min_dist = 0, math.sqrt((self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2), self.vec.copy()
        prev_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2

        while t + 1e-6 < tmax:
            self.step()

            cur_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2

            if self.save_states:
                rv.append(self.vec.copy())

            t += State.dt

            if cur_dist_sq > prev_dist_sq:
                #print(f"Distance was increasing at time {round(t, 2)}, stopping simulation. Min_dist: {round(prev_dist, 1)}ft")
                break

            prev_dist_sq = cur_dist_sq

        self.min_dist = math.sqrt(prev_dist_sq)

        if self.save_states:
            self.vec_list = rv

        if not self.save_states:
            assert not self.vec_list
            assert not self.commands
            assert not self.int_commands

    def update_command(self):
        'update command based on current state'''

        rho, theta, psi, v_own, v_int = state7_to_state5(self.vec, self.v_own, self.v_int)

        # 0: rho, distance
        # 1: theta, angle to intruder relative to ownship heading
        # 2: psi, heading of intruder relative to ownship heading
        # 3: v_own, speed of ownship
        # 4: v_int, speed in intruder

        # min inputs: 0, -3.1415, -3.1415, 100, 0
        # max inputs: 60760, 3.1415, 3,1415, 1200, 1200

        if rho > 60760:
            self.command = 0
        else:
            last_command = self.command

            net = State.nets[last_command]

            state = [rho, theta, psi, v_own, v_int]

            res = run_network(net, state)
            self.command = np.argmin(res)

            #names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

        if self.u_list_index is None:
            self.u_list_index = 0
        else:
            self.u_list_index += 1

            # repeat last command if no more commands
            self.u_list_index = min(self.u_list_index, len(self.u_list) - 1)

def plot(states, save_mp4):
    """plot a specific simulation"""

    assert states, "states list was empty"

    for s in states:
        s.vec = s.vec_list[0] # for printing the correct state
        print(f"plotting state {s}")

    init_plot()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
    axes.axis('equal')
    axes.set_xlim([-50000, 50000])
    axes.set_ylim([-50000, 50000])

    axes.set_title("Collision Avoidance System Simulations")
    axes.set_xlabel('X Position (ft)')
    axes.set_ylabel('Y Position (ft)')

    time_text = axes.text(0.02, 0.98, 'Time: 0', horizontalalignment='left', fontsize=16,
                          verticalalignment='top', transform=axes.transAxes)
    time_text.set_visible(False)

    custom_lines = [Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='c', lw=2),
                    Line2D([0], [0], color='r', lw=2)]

    axes.legend(custom_lines, ['Strong Left', 'Weak Left', 'Clear of Conflict', 'Weak Right', 'Strong Right'], \
                fontsize=14, loc='lower right')

    for s in states:
        s.make_artists(axes, show_intruder=True)

    plt.tight_layout()

    interval = 20 if not save_mp4 else 50 # ms per frame
    freeze_frames = 10 if not save_mp4 else 10

    num_frames = freeze_frames

    for s in states:
        num_frames += len(s.vec_list) + freeze_frames

    def animate(f):
        'animate function'

        #if not save_mp4:
        #    f *= 5 # multiplier to make animation faster

        if (f+1) % 10 == 0:
            print(f"Frame: {f+1} / {num_frames}")

        sim_index = 0
        f = f - freeze_frames

        while sim_index + 1 < len(states) and f >= len(states[sim_index].vec_list) + freeze_frames:
            f -= freeze_frames
            f -= len(states[sim_index].vec_list)
            sim_index += 1

        cur_state = states[sim_index]
            
        f = max(0, f)
        f = min(f, len(cur_state.vec_list) - 1)
        
        if f == 0:
            for i, s in enumerate(states):
                visible = i == sim_index
                
                for a in s.artists_list():
                    a.set_visible(visible)

                if visible:
                    s.set_plane_visible(True)

        time_text.set_text(f'Time: {f * State.dt:.1f}')

        artists = [time_text]

        cur_state.vec = cur_state.vec_list[f]
        artists += cur_state.update_artists(axes)

        return artists

    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True, repeat=True)

    if save_mp4:
        fps = 1000 // interval
        writer = animation.writers['ffmpeg'](fps=fps, metadata=dict(artist='Stanley Bak'), bitrate=1800)

        my_anim.save('collisions.mp4', writer=writer)
    else:
        plt.show()

def make_random_input(seed, intruder_can_turn=True, num_inputs=100):
    """make a random input for the system"""

    np.random.seed(seed) # deterministic random numbers

    # state vector is: x, y, theta, x2, y2, theta2, time
    init_vec = np.zeros(7)
    init_vec[2] = np.pi / 2 # ownship moving up initially

    radius = 10000 + np.random.random() * 55000 # [10000, 65000]
    angle = np.random.random() * 2 * np.pi
    int_x = radius * np.cos(angle)
    int_y = radius * np.sin(angle)
    int_heading = np.random.random() * 2 * np.pi

    init_vec[3] = int_x
    init_vec[4] = int_y
    init_vec[5] = int_heading

    # intruder commands for every control period (0 to 4)
    if intruder_can_turn:
        cmd_list = []

        for _ in range(num_inputs):
            cmd_list.append(np.random.randint(5))
    else:
        cmd_list = [0] * num_inputs

    # generate random valid velocities
    #init_velo = [np.random.randint(100, 1146),
    #             np.random.randint(60, 1146)]
    init_velo = [np.random.randint(100, 1200),
                 np.random.randint(0, 1200)]

    return init_vec, cmd_list, init_velo

def main():
    'main entry point'

    # parse arguments
    parser = argparse.ArgumentParser(description='Run ACASXU Dublins model simulator.')
    parser.add_argument("--save-mp4", action='store_true', default=False, help="Save plotted mp4 files to disk.")
    args = parser.parse_args()

    intruder_can_turn = False

    save_mp4 = args.save_mp4

    states = []
    interesting_seeds = []
    
    fixed_seeds = [47864, 350121, 1064918, 2408364, 3343828, 4376627, 4864696, 5347079, 5441672, 5758993, 6732555,
                   7078340, 7695200, 8409039, 8792867, 8805668, 9462487, 9940174, 10271554, 10350983, 10361014,
                   10910661, 10937981, 11181270, 11516559, 11520796, 11668276, 11997187, 12158481, 12307677, 12457049,
                   13572705, 13575733, 13668951, 13818967, 13933799, 14727763, 15251105, 15367862, 16532889, 16746180,
                   16761402, 16925934, 17240464, 17259343, 17574941, 17603003, 17864247, 17931056, 17932354, 18123525,
                   18300330, 18314102, 18643877, 18670654, 19493243, 20028446, 20407869, 20560926, 20938360, 20948433,
                   21155614, 21223587, 21442280, 22315177, 23176847, 23455001, 23587267, 23734541, 25565598, 25606743,
                   26589457, 26895016, 27555919, 27615987, 27682959, 29376848, 29395086, 29425865, 29448537, 29722732,
                   29943969, 30066332, 30140180, 30230147, 30231797, 30367618, 30970788, 31314234, 31691492, 32386985,
                   32533938, 32813115, 33385488, 33908749, 33909848, 33996760, 34235143, 35071786, 35645324, 35714381,
                   36351017, 36374716, 36488021, 36673002, 37128175, 37897062, 38227037, 38747871, 39583124, 39918090,
                   40226527, 40684850, 41012928, 41343984, 41519631, 41704917, 42693553, 43000551, 43052399, 44438047,
                   44827392, 45282719, 45907884, 46174234, 46281692, 46583527, 46610699, 46667733, 46765891, 47119273,
                   47369569, 47869383, 49029458, 50260801, 50627979, 50885042, 50885519, 51214438, 51471790, 52091416,
                   52265346, 53047610, 53226028, 54307543, 54510950, 54776571, 55248604, 55353097, 55415194, 55472342,
                   55545704, 56135639, 56531242, 56750554, 57266001, 57677429, 58164174, 58698494, 59268237, 59463889,
                   60055075, 60063934, 60100723, 60284303, 60430497, 60703468, 60942369, 61376538, 61435587, 61815001,
                   61930908, 61975048, 62081979, 62408790, 62773755, 63312672, 63735789, 64795151, 64857085, 65179987,
                   65672583, 66358822, 67474110, 68019457, 68148377, 68940570, 68955210, 69063306, 69662776, 69922660,
                   70403883, 70616733, 70638649, 71420484, 72614621, 72761353, 73160471, 73542722, 74134023, 74563766,
                   74845871, 75305552, 75309385, 76359673, 76399663, 76609912, 76702802, 76736897, 76997458, 77553645,
                   77605336, 77645667, 78548579, 78632461, 79364270, 79848208, 79880311, 80059435, 80630302, 80817403,
                   81568345, 82787488, 82916411, 83040435, 83492801, 83632598, 83814196, 84332465, 84419292, 85062540,
                   85435294, 86846945, 86966576, 87237324, 88421015, 88690714, 88763608, 88843131, 90349458, 90400326,
                   91452288, 91810150, 92185948, 92808261, 93136754, 93348437, 94369951, 94502809, 95487193, 96530445,
                   96705637, 97059651, 97495122, 98569984, 98738031]

    if fixed_seeds:
        max_vel_own_seeds = []
        int_heading_seeds = []

        min_vel_int = np.inf, -1

        for seed in fixed_seeds:
            init_vec, cmd_list, init_velo = make_random_input(seed, intruder_can_turn=intruder_can_turn)

            v_own = init_velo[0]
            v_int = init_velo[1]
            int_heading = init_vec[5]

            int_heading_seeds.append((int_heading, seed))
            max_vel_own_seeds.append((v_own, seed))

            if (v_int, seed) < min_vel_int:
                min_vel_int = (v_int, seed)

            #s = State(init_vec, init_velo[0], init_velo[1], save_states=True)
            #s.simulate(cmd_list)

            #if s.min_dist > 500:
            #    print(f"seed {seed} min_dist was {s.min_dist}")
            #    no_longer_collision_count += 1

        print(f"min_vel_int: {min_vel_int}")

        int_heading_seeds = list(sorted(int_heading_seeds))
        max_vel_own_seeds = list(sorted(max_vel_own_seeds))

        l = len(int_heading_seeds)
        indices = [0, l//5, 2*l//5, 3*l//5, 4*l//5]
        interesting_seeds = [int_heading_seeds[i][1] for i in indices]
        interesting_seeds = [min_vel_int[1], max_vel_own_seeds[-1][1]] + interesting_seeds + [max_vel_own_seeds[-3][1]]

        #interesting_seeds = [max_vel_own[1], min_vel_int[1]] + [s for _, s in sorted_seeds]
    assert interesting_seeds
    #State.dt = 0.1 # smoother animation

    for seed in interesting_seeds:
        init_vec, cmd_list, init_velo = make_random_input(seed, intruder_can_turn=intruder_can_turn)
        s = State(init_vec, init_velo[0], init_velo[1], save_states=True)
        s.simulate(cmd_list)
        states.append(s)

        d = round(s.min_dist, 2)
        print(f"\nSeed {seed} has min_dist {d}ft")

    save_mp4 = True
    plot(states, save_mp4)

if __name__ == "__main__":
    main()
