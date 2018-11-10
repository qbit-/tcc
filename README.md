# TCC
Tensor Structured Couple Cluster theory. This code implements Coupled Cluster 
with Canonical Polyadic Decomposition (CPD) and Tensor Hyper Contraction (THC)
decomposed amplitudes and integrals. This code implements the CC as described in the 
article: https://arxiv.org/abs/1708.02674.
**If you find the code useful, please cite the above article**

Requirements:
* PySCF
* Numpy
* h5py

Examples of calculations can be found in rccsd_*.py files. PySCF is used for
input/SCF calculations/integrals
