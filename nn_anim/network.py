'''
Stanley Bak

Network container classes for nnstar (partialled edited for visualization)
'''

import numpy as np
from scipy.signal import convolve2d

class NeuralNetwork():
    'neural network container'

    def __init__(self, layers):

        assert layers, "layers should be a non-empty list"

        for i, layer in enumerate(layers):
            assert layer.layer_num == i, f"Layer {i} has incorrect layer num: {layer.layer_num}: {layer}"
        
        self.layers = layers
        self.check_io()

        for layer in layers:
            layer.network = self

    def __str__(self):
        return f'[NeuralNetwork with {len(self.layers)} layers with {self.layers[0].get_input_shape()} input and ' + \
          f'{self.get_output_shape()} output]'

    def get_input_shape(self):
        'get the input shape to the first layer'

        return self.layers[0].get_input_shape()

    def get_output_shape(self):
        'get the output shape from the last layer'

        return self.layers[-1].get_output_shape()

    def get_num_inputs(self):
        'get the scalar number of inputs'

        shape = self.get_input_shape()

        rv = 1

        for x in shape:
            rv *= x

        return rv

    def get_num_outputs(self):
        'get the scalar number of outputs'

        shape = self.get_output_shape()

        rv = 1

        for x in shape:
            rv *= x

        return rv

    def execute(self, input_vec, save_branching=False):
        '''execute the neural network with the given input vector

        if save_branching is True, returns (output, branch_list), where branch_list contains one list for each layer,
            and each layer-list is a list of the branching decisions taken by each neuron. For layers with ReLUs, this
            will be True/False values (True if positive branch is taken), for max pooling layers these will be ints, or
            lists of ints (if multiple max values are equal)
        
        otherwise, just returns output
        '''

        if save_branching:
            branch_list = []

        state = np.array(input_vec, dtype=float) # test with float32 dtype?
        assert state.shape == self.get_input_shape(), f"in network.execute(), passed-in shape was {state.shape}, " + \
            f"network expects input of shape {self.get_input_shape()}"

        for layer in self.layers:
            if save_branching:
                state, layer_branch_list = layer.execute(state, save_branching=True)
                branch_list.append(layer_branch_list)
            else:
                state = layer.execute(state)

        assert state.shape == self.get_output_shape()

        rv = (state, branch_list) if save_branching else state
        
        return rv

    def check_io(self):
        'check the neural network for input / output compatibility'

        for i, layer in enumerate(self.layers):
            if i == 0:
                continue

            prev_output_shape = self.layers[i-1].get_output_shape()
            my_input_shape = layer.get_input_shape()

            assert prev_output_shape == my_input_shape, f"output of layer {i-1} was {prev_output_shape}, " + \
              f"and this doesn't match input of layer {i} which is {my_input_shape}"

class FullyConnectedLayer():
    'fully connected layer'

    def __init__(self, layer_num, weights, biases, has_relu, prev_layer_output_shape=None):

        if isinstance(weights, list):
            weights = np.array(weights, dtype=float)

        if isinstance(biases, list):
            biases = np.array(biases, dtype=float)
        
        self.layer_num = layer_num
        self.weights = weights
        self.biases = biases
        self.has_relu = has_relu
        self.prev_layer_output_shape = prev_layer_output_shape

        self.network = None

        assert biases.shape[0] == weights.shape[0], "biases vec in layer " + \
            f"{layer_num} has length {biases.shape[0]}, but weights matrix has height " + \
            f"{weights.shape[0]}"

        assert len(biases.shape) == 1, f'expected 1-d bias vector at layer {layer_num}, got {biases.shape}'
        assert len(weights.shape) == 2

        if prev_layer_output_shape is not None:
            expected_inputs = 1

            for x in prev_layer_output_shape:
                expected_inputs *= x

            assert expected_inputs == weights.shape[1], f"FC Layer weight matrix shape was {weights.shape}, but " + \
                f"prev_layer_output_shape {prev_layer_output_shape} needs {expected_inputs} columns"
        
    def __str__(self):
        desc = "(with ReLU)" if self.has_relu else "(no ReLU)"
        return f'[FullyConnectedLayer {desc} with {self.get_input_shape()} input and {self.get_output_shape()} output]'

    def get_input_shape(self):
        'get the input shape to this layer'

        rv = self.prev_layer_output_shape

        if rv is None:
            rv = (self.weights.shape[1],)

        return rv

    def get_output_shape(self):
        'get the output shape from this layer'

        return (self.weights.shape[0],)

    def transform_star(self, star):
        'apply the linear transformation part of the layer to the passed-in lp_star (not relu)'

        star.a_mat = np.dot(self.weights, star.a_mat)
        star.bias = np.dot(self.weights, star.bias) + self.biases

    def transform_zono(self, zono):
        'apply the linear transformation part of the layer to the passed-in zonotope (not relu)'

        zono.mat_t = np.dot(self.weights, zono.mat_t)
        zono.center = np.dot(self.weights, zono.center) + self.biases

    def execute(self, state, save_branching=False, skip_relu=False):
        '''execute the fully connected layer on a concrete state

        if save_branching is True, returns (output, branch_list), where branch_list is a list of booleans for each
            neuron in the layer that is True if the nonnegative branch of the ReLU was taken, False if negative

        if skip_relu is True, will skip relu part, regardless of if layer has relu or not       
 
        otherwise, just returns output
        '''

        if save_branching:
            branch_list = []

        assert state.shape == self.get_input_shape(), f"state shape to fully connected layer was {state.shape}, " + \
            f"expected {self.get_input_shape()}"

        state = nn_flatten(state)

        rv = np.dot(self.weights, state)

        assert len(self.biases.shape) == 1
        rv = rv + self.biases
        assert len(rv.shape) == 1

        if self.has_relu and not skip_relu:
            next_list = []

            for val in rv:
                if save_branching:
                    branch_list.append(val >= 0)
                    
                next_list.append(max(0.0, float(val)))

            rv = np.array(next_list, dtype=state.dtype)

        assert rv.shape == self.get_output_shape()

        rv = (rv, branch_list) if save_branching else rv

        return rv

def nn_flatten(image):
    'flatten a multichannel image to a 1-d array'

    # note: fortran-style flattening makes Tran's example network classify correctly, so I guess it's the standard
    vec = image.flatten('F')

    return vec

def nn_unflatten(image, shape):
    '''unflatten to a multichannel image from a 1-d array

    this uses reshape, so may not be a copy
    '''

    assert len(image.shape) == 1

    rv = image.reshape(shape, order='F')

    assert rv.shape == shape

    return rv

def convert_weights(weights):
    'convert weights from a list format to an np.array format'

    layers = [] # list of np.array for each layer

    for weight_mat in weights:
        layers.append(np.array(weight_mat, dtype=float))

    # this prevents python from attempting to broadcast the layers together
    rv = np.empty(len(layers), dtype=object)
    rv[:] = layers

    return rv

def convert_biases(biases):
    'convert biases from a list format to an np.array format'

    layers = [] # list of np.array for each layer

    for biases_vec in biases:
        bias_ar = np.array(biases_vec, dtype=float)
        bias_ar.shape = (len(biases_vec),)
        
        layers.append(bias_ar)

    # this prevents python from attempting to broadcast the layers together
    rv = np.empty(len(layers), dtype=object)
    rv[:] = layers

    return rv

def weights_biases_to_nn(weights, biases):
    '''create a NeuralNetwork from a weights and biases matrix

    this assumes every layer is a fully-connected layer followed by a ReLU, except for the last one
    '''

    if isinstance(weights, list):
        weights = convert_weights(weights)

    if isinstance(biases, list):
        biases = convert_biases(biases)

    num_layers = weights.shape[0]
    assert biases.shape[0] == num_layers, f"nn has {num_layers} layers, but biases shape was {biases.shape}"

    layers = []

    for i, (layer_weights, layer_biases) in enumerate(zip(weights, biases)):
        has_relu = i < num_layers - 1
        layers.append(FullyConnectedLayer(i, layer_weights, layer_biases, has_relu))

    return NeuralNetwork(layers)
