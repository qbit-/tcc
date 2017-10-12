import numpy as np
from tcc.utils import khatrirao


def cpd_initialize(ext_sizes, rank):
    """
    Initialize a CPD decomposition.
    :param ext_sizes:  ndarray
        sizes of external indices
    :param rank:  int
        rank of the CPD decomposition

    Returns
    -------
    lam, factors: vector of factor norms and a tuple with factors  
    """

    return [np.random.rand(size, rank)
            for size in ext_sizes]


def cpd_normalize(factors, sort=True, merge_lam=False):
    """
    Normalize the columns of factors to unit. The norms
    are returned in as ndarray

    :param factors: ndarray iterable
        factor matrices
    :param sort: bool, optional
        default True
    :param merge_lam: bool, optional, default False
        merge lam and normalized factors into
        a single tuple, as in nCPD format

    Returns
    -------
    lam: ndarray
    factors: ndarray tuple  

    >>> import numpy as np;
    >>> k = cpd_initialize([2,2],3)
    >>> l, kn = cpd_normalize(k)
    >>> np.allclose(cpd_rebuild(k), ncpd_rebuild((l,) + kn))
    True

    """

    lam = np.ones(factors[0].shape[1])
    new_factors = []

    for factor in factors:
        lam_factor = np.linalg.norm(factor, axis=0)
        new_factors.append(factor / lam_factor)
        lam = lam * lam_factor

    if sort:
        order = np.argsort(lam)[::-1]
        lam = lam[order]

        for idx, factor in enumerate(new_factors):
            new_factors[idx] = factor[:, order]

    if merge_lam:
        return (lam, ) + tuple(new_factors)
    else:
        return lam, tuple(new_factors)


def ncpd_denormalize(factors, sort=True):
    """
    Normalize the columns of factors to unit. The norms
    are returned in as ndarray

    :param factors: ndarray iterable
        factor matrices in ncpd format
    :param sort: bool, if the factors are sorted by norm
        default True

    Returns
    -------
    lam: ndarray
    factors: ndarray tuple  

    >>> import numpy as np;
    >>> k = cpd_initialize([2,2],3)
    >>> kn = cpd_normalize(k, sort=False, merge_lam=True)
    >>> kk = ncpd_denormalize(kn, sort=True)
    >>> np.allclose(cpd_rebuild(k), cpd_rebuild(kk))
    True

    """

    lam, new_factors = factors

    if sort:
        order = np.argsort(lam)[::-1]
        lam = lam[order]

        for idx, factor in enumerate(new_factors):
            new_factors[idx] = factor[:, order]

    lam_factor = np.power(lam, 1. / len(lam))
    for idx, factor in enumerate(new_factors):
        new_factors[idx] = np.dot(factor, np.diag(lam_factor))

    return new_factors


def ncpd_initialize(ext_sizes, rank):
    """
    Initialize a normalized CPD decomposition.
    :param ext_sizes:  ndarray
        sizes of external indices
    :param rank: int
        rank of the CPD decomposition

    Returns 
    -------
    norm_and_factors: ndarray tuple

    """

    lam, factors = cpd_normalize(
        cpd_initialize(ext_sizes, rank),
        sort=True)

    return (lam, ) + factors


def cpd_rebuild(factors):
    """
    Rebuild full tensor from it's CPD decomposition
    :param factors: iterable with factor matrices

    Returns
    -------
    tensor: ndarray

    >>> a = np.array([[1,3],[2,4]])
    >>> b = np.array([[5,6],[7,8],[9,10]])
    >>> c = np.array([[11,12],[13,14]])
    >>> ref = np.array([  784.,  1058.,  1332.])
    >>> np.allclose(sum(cpd_rebuild((a,b,c))[:,:,1], 1), ref)
    True
    """

    N = len(factors)
    tensor_shape = tuple(factor.shape[0] for factor in factors)
    tensor = khatrirao(tuple(factors[ii] for ii in range(
        N - 1)), reverse=False).dot(factors[N - 1].transpose())
    return tensor.reshape(tensor_shape)


def ncpd_rebuild(norm_factors):
    """
    Rebuild full tensor from it's normalized CPD decomposition
    :param norm_factors: iterable with norm and factors. First
                         iterate is the normalization vector lambda,
                         the rest are factor matrices

    
    >>> import numpy as np; np.random.seed(0)
    >>> k = cpd_initialize([2,2],3)
    >>> np.random.seed(0)
    >>> kn = ncpd_initialize([2,2],3)
    >>> np.allclose(cpd_rebuild(k), ncpd_rebuild(kn))
    True

    """

    lam = norm_factors[0]
    factors = norm_factors[1:]

    N = len(factors)
    tensor_shape = tuple(factor.shape[0] for factor in factors)
    t = khatrirao(tuple(factors[ii] for ii in range(
        N - 1)), reverse=False).dot((lam*factors[N - 1]).transpose())
    return t.reshape(tensor_shape)


def cpd_contract_free_cpd(factors_top, factors_bottom,
                          conjugate=False, skip_factor=None):
      """
      Contract "external" indices of two CPD decomposed tensors
      :param factors_top: iterable with CPD decomposition
                          of the top (left) tensor
      :param factors_bottom: iterable with CPD decomposition
                          of the bottom (right) tensor
      :param conjugate: conjugate the top (left) factor (default: False)
      :param skip_factor: int, default None
                skip the factor number skip_factor 
      
      >>> import numpy as np
      >>> k1 = cpd_initialize([3,3,4], 3)
      >>> k2 = cpd_initialize([3,3,4], 3)
      >>> t1 = cpd_rebuild(k1)
      >>> t2 = cpd_rebuild(k2)
      >>> s1 = t1.conj().flatten().dot(t2.flatten()[None].T)
      >>> s2 = np.sum(cpd_contract_free_cpd(k1, k2, conjugate=True))
      >>> np.allclose(s1, s2)
      True

      """

      from functools import reduce

      if skip_factor is not None:
          factors_top = [factors_top[ii]
                         for ii in range(len(factors_top))
                         if ii != skip_factor]
          factors_bottom = [factors_bottom[i]
                            for ii in range(len(factors_bottom))
                            if ii != skip_factor]

      if conjugate:
          factors_top = [factor.conjugate() for factor in factors_top]
          
      s = reduce(lambda x, y: x * y,
                 map(lambda x: x[0].T.dot(x[1]),
                     zip(factors_top, factors_bottom)))

      return s


def ncpd_contract_free_ncpd(factors_top, factors_bottom,
                            conjugate=False, skip_factor=None):
      """
      Contract "external" indices of two nCPD decomposed tensors
      :param factors_top: iterable with nCPD decomposition
                          of the top (left) tensor
      :param factors_bottom: iterable with nCPD decomposition
                          of the bottom (right) tensor
      :param conjugate: conjugate the top (left) factor (default: False)
      :param skip_factor: int, default None
                skip the factor number skip_factor 

      >>> import numpy as np
      >>> k1 = cpd_initialize([3,3,4], 3)
      >>> kn1 = cpd_normalize(k1, sort=False, mergeout=True)
      >>> k2 = cpd_initialize([3,3,4], 3)
      >>> kn2 = cpd_normalize(k2, sort=False, mergeout=True)
      >>> s1 = cpd_contract_free_cpd(k1, k2)
      >>> s2 = ncpd_contract_free_ncpd(kn1, kn2)
      >>> np.allclose(s1, s2)
      True

      """

      s = cpd_contract_free_cpd(factors_top[1:],
                                factors_bottom[1:],
                                conjugate, skip_factor)

      lam_top = factors_top[0]
      lam_bottom = factors_bottom[0]
      
      return s * (lam_top[np.newaxis].T * lam_bottom)


def cpd_symmetrize(factors, permdict, adjust_scale=True):
    """
    Produce a symmetric CPD decomposition.
    :param factors: CPD factors
    :param permdict: dictionary of tuple : tuple pairs.
                     keys are tuples containing the permutation
                     values are tuples of any of ('ident', 'neg', 'conj')
                     Identity permutation has to be excluded (added internally)
    :param adjust_scale: bool, default True
              If factors have to be scaled by the order of the permutation group

    Returns
    -------
    symm_factos: ndarray list, symmetrized CPD factors

    """

    if adjust_scale:
        scaling_factor = 1 / (len(permdict) + 1)
    else:
        scaling_factor = 1

    new_factors = []

    def ident(x):
        return x
    def neg(x):
        return -1 * x
    def conj(x):
        return np.conj(x)
    from functools import reduce

    new_factors = []
    for idx, factor in enumerate(factors):
        new_factor = factor * scaling_factor
        for perm, operations in permdict.items():
            transforms = []
            for operation in operations:
                if operation == 'ident':
                    transforms.append(ident)
                elif operation == 'neg':
                    transforms.append(neg)
                elif operation == 'conj':
                    transforms.append(conj)
                else:
                    raise ValueError(
                        'Unknown operation: {}'.format(
                            operation))
            new_factor = np.hstack(
                (new_factor,
                 scaling_factor *
                 reduce((lambda x, y: y(x)), transforms, factors[perm[idx]])))
        new_factors.append(new_factor)

    return new_factors 


def ncpd_symmetrize(norm_factors, permdict):
    """
    Produce a symmetric nCPD decomposition.
    :param norm_factors: norm and normalized CPD factors
    :param permdict: dictionary of tuple : tuple pairs.
                     keys are tuples containing the permutation
                     values are tuples of any of ('ident', 'neg', 'conj')
                     Identity permutation has to be excluded (added internally)
    Returns
    -------
    symm_norm_factos: ndarray list, symmetrized nCPD factors

    """

    lam = norm_factors[0]
    factors = norm_factors[1:]
    
    new_lam = 1 / (len(permdict) + 1) * np.hstack((lam, ) * (len(permdict) + 1))
    new_factors = cpd_symmetrize(factors, permdict, adjust_scale=False)
    
    return [new_lam, ] + new_factors




