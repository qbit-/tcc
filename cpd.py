import numpy as np
from tcc.utils import khatrirao


def cpd_rebuild(factors):
    """
    Rebuild full tensor from it's CPD decomposition
    :param factors: iterable with factor matrices

    >>> a = np.array([[1,3],[2,4]])
    >>> b = np.array([[5,6],[7,8],[9,10]])
    >>> c = np.array([[11,12],[13,14]])
    >>> ref = np.array([  784.,  1058.,  1332.])
    >>> np.allclose(sum(cpd_rebuild((a,b,c))[:,:,1], 1), ref)
    True
    """
    N = len(factors)
    tensor_shape = tuple(factor.shape[0] for factor in factors)
    t = khatrirao(tuple(factors[ii] for ii in range(
        N - 1)), reverse=True).dot(factors[N - 1].transpose())
    return t.reshape(tensor_shape)


# def cpd_contract_cpd(factors_top, factors_bottom, conjugate=False):
#     """
#     Fully contract two CPD decomposed tensors
#     :param factors_top: iterable with CPD decomposition
#                         of the top (left) tensor
#     :param factors_bottom: iterable with CPD decomposition
#                         of the bottom (right) tensor
#     :param conjugate: conjugate the top (left) factor (default: False)

#     >>> raise
#     """

#     from functools import reduce

#     if conjugate:
#         factors_top = [factor.conjugate() for factor in factors_top]

#     s = reduce(lambda x, y: x * y,
#                map(lambda x: x[0].T.dot(x[1]),
#                    zip(factors_top, factors_bottom)))

#     return s.trace()


def cpd_initialize(ext_sizes, rank):
    """
    Initialize a CPD decomposition.
    :param ext_sizes:  sizes of external indices
    :param rank:  rank of the CPD decomposition

    >>> np.random.seed(0)
    >>> us = cpd_initialize((2,3,1), 3)
    """

    return (np.random.rand(size, rank)
            for size in ext_sizes)

# def cpd_normalize(factors, sort=False):
#     """
#     Normalize the columns of factors to unit. The norms
#     are returned in as ndarray

#     :param factors: iterable with factors
#     :param sort: 
#     """
