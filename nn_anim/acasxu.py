'''
Generic ACASXu loading (partially edited for visualization)

Stanley Bak, Jan 2020
'''

import numpy as np
from scipy.io import loadmat

from network import weights_biases_to_nn, FullyConnectedLayer

def load_init_network(net_pair, property_str):
    '''load the network / spec and return it

    the network is based on the net_pair, using specification spec

    returns (init_box, network, spec)
    '''

    assert isinstance(property_str, str)
    assert property_str != "6", "spec 6 is a disjunctive initial set, use 6.1 or 6.2 for the two parts"

    required_spec_net_list = [
        ["5", (1, 1)],
        ["6.1", (1, 1)],
        ["6.2", (1, 1)],
        ["7", (1, 9)],
        ["8", (2, 9)],
        ["9", (3, 3)],
        ["10", (4, 5)]]

    for req_spec, tup in required_spec_net_list:
        if property_str == req_spec:
            assert net == tup, f"spec {property_str} should only be run on net {tup}"

    # load the network and prepare input / output specs
    n1, n2 = net_pair
    folder = '.'
    matfile = loadmat(f'{folder}/ACASXU_run2a_{n1}_{n2}_batch_2000.mat')

    weights = matfile['W'][0]
    biases = matfile['b'][0]

    for b in biases:
        b.shape = (len(b),)

    range_for_scaling = matfile['range_for_scaling'][0]
    means_for_scaling = matfile['means_for_scaling'][0]

    num_inputs = weights[0].shape[1]
    init_lb, init_ub = get_init_box(property_str)

    # normalize input
    for i in range(num_inputs):
        init_lb[i] = (init_lb[i] - means_for_scaling[i]) / range_for_scaling[i]
        init_ub[i] = (init_ub[i] - means_for_scaling[i]) / range_for_scaling[i]

        #print(f"Input {i}: {init_lb[i], init_ub[i]}")

    spec = None #get_spec(property_str)
    
    init_box = list(zip(init_lb, init_ub))
    
    network = weights_biases_to_nn(weights, biases)

    scale_outputs = False

    if scale_outputs:
        # add one more layer to the network for the output scaling
        output_mult, output_bias = range_for_scaling[-1], means_for_scaling[-1]
        output_mult_mat = np.identity(5) * output_mult
        output_bias_vec = np.array([output_bias] * 5, dtype=float)
        layer_num = len(network.layers)
        network.layers.append(FullyConnectedLayer(layer_num, output_mult_mat, output_bias_vec, False))

    return init_box, network, spec

def get_init_box(property_str):
    'get lb, ub lists for the given property'

    if property_str in ("1", "2"):
        init_lb = [55947.691, -3.141592, -3.141592, 1145, 0]
        init_ub = [60760, 3.141592, 3.141592, 1200, 60]
    elif property_str == "3":
        init_lb = [1500, -0.06, 3.1, 980, 960]
        init_ub = [1800, 0.06, 3.141592, 1200, 1200]
    elif property_str == "4":
        init_lb = [1500, -0.06, 0, 1000, 700]
        init_ub = [1800, 0.06, 0, 1200, 800]
    elif property_str == "5":
        init_lb = [250, 0.2, -3.141592, 100, 0]
        init_ub = [400, 0.4, -3.141592 + 0.005, 400, 400]
    elif property_str == "6.1":
        init_lb = [12000, 0.7, -3.141592, 100, 0]
        init_ub = [62000, 3.141592, -3.141592 + 0.005, 1200, 1200]
    elif property_str == "6.2":
        init_lb = [12000, -3.141592, -3.141592, 100, 0]
        init_ub = [62000, -0.7, -3.141592 + 0.005, 1200, 1200]
    elif property_str == "7":
        init_lb = [0, -3.141592, -3.141592, 100, 0]
        init_ub = [60760, 3.141592, 3.141592, 1200, 1200]
    elif property_str == "8":
        init_lb = [0, -3.141592, -0.1, 600, 600]
        init_ub = [60760, -0.75*3.141592, 0.1, 1200, 1200]
    elif property_str == "9":
        init_lb = [2000, -0.4, -3.141592, 100, 0]
        init_ub = [7000, -0.14, -3.141592 + 0.01, 150, 150]
    elif property_str == "10":
        init_lb = [36000, 0.7, -3.141592, 900, 600]
        init_ub = [60760, 3.141592, -3.141592 + 0.01, 1200, 1200]
    else:
        raise RuntimeError(f"init_box undefined for property {property_str}")

    #print("Warning: small init")
    #init_lb[0] = init_ub[0] - 1
    #init_lb[1] = init_ub[1] - 0.001

    return init_lb, init_ub

