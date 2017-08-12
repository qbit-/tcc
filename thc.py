from tcc.utils import khatrirao
import numpy as np

def thc_rebuild(factors):
    """
    Rebuild full tensor from it's THC decomposition
    :param factors: iterable with factor matrices,
    the order of THC factors is given by:
    v_{pqrs} = \sum_{m, n} x1_{p,m} x2_{q,m} x5_{m, n} x3_{r,n} x4_{p,n}

    >>> raise
    """

    if len(factors) != 5:
        raise ValueError('THC decomposition has only 5 factors')

    tensor_shape = tuple(factor.shape[0] for factor in factors[:4])
    t = khatrirao(factors[:2], reverse=True).dot(factors[4]).dot(
        khatrirao(factors[2:4], reverse=True).transpose())

    return t.reshape(tensor_shape)


def thc_contract_thc(factors_top, factors_bottom, conjugate=False):
    """
    Fully contract two THC decomposed tensors
    :param factors_top: iterable with THC decomposition
                        of the top (left) tensor
    :param factors_bottom: iterable with THC decomposition
                        of the bottom (right) tensor

    the order of THC factors is given by:
    v_{pqrs} = \sum_{m, n} x1_{p,m} x2_{q,m} x5_{m, n} x3_{r,n} x4_{p,n}
    :param conjugate: conjugate the top (left) factor (default: False)

    >>> raise
    """

    if len(factors_top) != 5 or len(factors_bottom) != 5:
        raise ValueError('THC decomposition has only 5 factors')

    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    s = ((factors_top[0].T.dot(factors_bottom[0]))
         * (factors_top[1].T.dot(factors_bottom[1]))).dot(
             factors_bottom[4]
         ).dot(
             (factors_bottom[2].T.dot(factors_top[2]))
             * (factors_bottom[3].T.dot(factors_top[3]))
         ).dot(
             factors_top[4].T
         )
    return s.trace()


def thc_initialize(ext_sizes, rank, scale_to_norm=None):
    """
    Initialize a THC decomposition.
    :param ext_sizes:  sizes of external indices
    :param rank:  rank of the THC decomposition
    :param scale_to_norm: the desired norm of the final tensor

    >>> raise
    """

    if len(ext_sizes) != 4:
        raise ValueError('THC is defined only for order 4 tensors')

    factors = [np.random.rand(size, rank)
               for size in ext_sizes]
    factors.append(np.random.rand(rank, rank))

    if scale_to_norm is not None:
        norm = thc_contract_thc(factors, factors)
        factors = [scale_to_norm / np.sqrt(np.power(norm, 1./8)) * factor
                   for factor in factors[:4]] + [scale_to_norm /
                                                 np.power(norm, 1./4)
                                                 * factors[4], ]

    return factors
