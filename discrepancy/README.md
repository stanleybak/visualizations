# Verification With Discrepancy Functions

A discrepancy function can be used to verify systems with nonlinear differential equations. The function tells you how far points can deviate over time, based on their initial distances. Individual simulations, then, can prove that all nearby points cannot enter the unsafe set.

This folder computes visualizations of discrepancy functions for the Vanderpol system. There is also a dynamics-scaling version, speeding up or slowing down simulations to stay on the hyperplane (line) orthogonal to the center simulation and gradient (green point and green dotted line). This is similar to what happens with pseudo-invariants [1] and dynamics scaling [2].

The system dynamics are:

x' = y

y' = (1 - x^2) * y - x

[1] "Reducing the Wrapping Effect in Flowpipe Construction using Pseudo-Invariants", S. Bak, Fourth Workshop on Design, Modeling and Evaluation of Cyber Physical Systems (CyPhy 2014)

[2] "Time-Triggered Conversion of Guards for Reachability Analysis of Hybrid Automata", S. Bak, S. Bogomolov, M. Althoff, 15th International Conference on Formal Modelling and Analysis of Timed Systems (FORMATS 2017)
