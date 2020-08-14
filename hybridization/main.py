'''
Vanderpol hybridization with dynamics scaling

Stanley Bak, Aug 2020
'''

import random

import matplotlib
import matplotlib.pyplot as plt

from sim_rand import sim_rand
from sim_linear import sim_linear
from sim_scaled import sim_scaled
from sim_dynamics_scaling import sim_dynamics_scaling

def main():
    'main entry point'

    random.seed(0) # determinstic random numbers
    matplotlib.use('TkAgg') # set backend
    plt.style.use(['bmh', './bak_matplotlib.mlpstyle'])

    make_mp4 = True
 
    sim_rand(make_mp4=make_mp4)
    sim_linear(make_mp4=make_mp4)

    sim_scaled(make_mp4=make_mp4, scale_func_str='none')
    sim_scaled(make_mp4=make_mp4, scale_func_str='two')
    sim_scaled(make_mp4=make_mp4, scale_func_str='quarter')
    sim_scaled(make_mp4=make_mp4, scale_func_str='line')
    sim_scaled(make_mp4=make_mp4, scale_func_str='onoff')

    sim_dynamics_scaling(make_mp4=make_mp4)

if __name__ == "__main__":
    main()
