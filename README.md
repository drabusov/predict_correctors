# predict_correctors
accelerator lattice optimization in one step
Use simplified lattice and make four major steps:
1) Imply gradient errors with a random_seed and dK
2) Optimize the lattice using 24 correctors
3) Create dataset: A. Measure observables before and after correction -> (feature selection) B. Target -> values of correctors
4) Add noise to the data (beta-beating is measured at BPMs with some errors)
5) Train the network
