# Neural Network Execution Visualization

A layer-by-layer execution is performed for the ACAS Xu neural network [1], with network 5-1 using the initial
set for property 3.

The blue dots are 500 random simulations, sampled uniformly from the initial set.

The gray region is the exact reachable set (~9500 regions at the final layer), computed using a geometric reachability technique with star sets [2], similar to the method implemented in the `nnv` tool [3]. The actual implementation I used for this is under review and will be released soon. The reachable set computation takes about 10 seconds single threaded (it takes longer to create the vertices than to compute the reach set, about 40 seconds). The green point is a worst-case (maximum x value) execution that can be generated with the approach.

Prepared by Stanley Bak, Feb 2020

[1] Katz, Guy, et al. "Reluplex: An efficient SMT solver for verifying deep neural networks." International Conference on Computer Aided Verification. Springer, Cham, 2017.

[2] Tran, Hoang-Dung, et al. "Star-Based Reachability Analysis of Deep Neural Networks." International Symposium on Formal Methods. Springer, Cham, 2019.

[3] https://github.com/verivital/nnv
