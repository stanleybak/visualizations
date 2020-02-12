# Neural Network Execution Visualization

[<img src="https://img.youtube.com/vi/mkHj3hJNM18/maxresdefault.jpg" width="50%">](https://youtu.be/mkHj3hJNM18)

This animation shows a layer-by-layer execution of the ACAS Xu neural network [1], with network 5-1 using the initial
set of states for property 3.

The blue dots are 500 random simulations, sampled uniformly from the initial set (a five dimensional set).

The gray region is the exact reachable set (~9500 regions at the final layer), computed using a geometric set-based execution technique using star sets [2], similar to the method implemented in the `nnv` tool [3]. The actual implementation I used for this is under review and will be released soon. The reachable set computation takes about 10 seconds single threaded (it takes longer to create the vertices than to compute the reach set, about 40 seconds). 

The green point is a worst-case (maximum x value in the output space) execution that can be generated with the approach. This would be very hard to find using random simulations, and shows the power of our set-based method.

The red dots are the same as the blue dots, just shown for all the layers at once without animating.

The output is a 2-d projection of the high-dimensional set at each layer, as I can't visualize 50-dimensional sets directly. See `nn_anim.py` for the projection used at each layer, specifically the `proj` function. In the input and output layers, the x axis is a scaled version of dimension 3 and a the y axis is a scaled version of dimension 4.

Created by Stanley Bak (http://stanleybak.com), Feb 2020

### References

[1] Katz, Guy, et al. "Reluplex: An efficient SMT solver for verifying deep neural networks." International Conference on Computer Aided Verification. Springer, Cham, 2017.

[2] Tran, Hoang-Dung, et al. "Star-Based Reachability Analysis of Deep Neural Networks." International Symposium on Formal Methods. Springer, Cham, 2019.

[3] https://github.com/verivital/nnv
