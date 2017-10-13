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

    >>> k = cpd_initialize([2,2],3)
    >>> kn = cpd_normalize(k, sort=False, merge_lam=True)
    >>> kk = ncpd_denormalize(kn, sort=False)
    >>> np.allclose(cpd_rebuild(k), cpd_rebuild(kk))
    True

    """

    lam = factors[0]
    new_factors = list(factors[1:])

    if sort:
        order = np.argsort(lam)[::-1]
        lam = lam[order]

        for idx, factor in enumerate(new_factors):
            new_factors[idx] = factor[:, order]

    lam_factor = np.power(lam, 1. / len(new_factors))
    for idx, factor in enumerate(new_factors):
        new_factors[idx] = np.dot(factor, np.diag(lam_factor))

    return new_factors


def ncpd_renormalize(factors, sort=True):
    """
    Normalizes the columns of factors in nCPD format, in
    case the normalization was lost due to some operation on
    the original factors.

    :param factors: ndarray iterable
        factor matrices in ncpd format
    :param sort: bool, if the factors are sorted by norm
        default True

    Returns
    -------
    lam: ndarray
    factors: ndarray tuple

    >>> k = cpd_initialize([2,2],3)
    >>> kn = cpd_normalize(k, merge_lam=True)
    >>> kk = [np.ones(3), ] + k
    >>> kt = ncpd_renormalize(kk)
    >>> np.allclose(ncpd_rebuild(kt), ncpd_rebuild(kn))
    True

    """
    old_lam = factors[0]
    old_factors = list(factors[1:])

    lam, factors = cpd_normalize(old_factors, sort=sort, merge_lam=False)

    if sort:
        order = np.argsort(old_lam)[::-1]
        old_lam = old_lam[order]

    new_lam = old_lam * lam

    return (new_lam, ) + factors


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
        N - 1)), reverse=False).dot((lam * factors[N - 1]).transpose())
    return t.reshape(tensor_shape)


def cpd_contract_free_cpd(factors_top, factors_bottom,
                          skip_factor=None, conjugate=False):
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
        factors_bottom = [factors_bottom[ii]
                          for ii in range(len(factors_bottom))
                          if ii != skip_factor]

    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    s = reduce(lambda x, y: x * y,
               map(lambda x: x[0].T.dot(x[1]),
                   zip(factors_top, factors_bottom)))

    return s


def ncpd_contract_free_ncpd(factors_top, factors_bottom,
                            skip_factor=None, conjugate=False):
    """
    Contract "external" indices of two nCPD decomposed tensors
    :param factors_top: iterable with nCPD decomposition
                        of the top(left) tensor
    :param factors_bottom: iterable with nCPD decomposition
                        of the bottom(right) tensor
    :param conjugate: conjugate the top(left) factor(default: False)
    :param skip_factor: int, default None
              skip the factor number skip_factor

    >>> k1 = cpd_initialize([3, 3, 4], 3)
    >>> kn1 = cpd_normalize(k1, sort=False, merge_lam=True)
    >>> k2 = cpd_initialize([3, 3, 4], 3)
    >>> kn2 = cpd_normalize(k2, sort=False, merge_lam=True)
    >>> s1 = cpd_contract_free_cpd(k1, k2)
    >>> s2 = ncpd_contract_free_ncpd(kn1, kn2)
    >>> np.allclose(s1, s2)
    True
    >>> s3 = ncpd_contract_free_ncpd(kn1, kn2, skip_factor=0)
    >>> np.allclose(s2, np.diag(kn1[0]) @ s3 @ np.diag(kn2[0]))
    True
    """
    if skip_factor is not None:
        if skip_factor != 0:
            skip_cpd = skip_factor - 1
        else:
            skip_cpd = None
    else:
        skip_cpd = None
    s = cpd_contract_free_cpd(factors_top[1:],
                              factors_bottom[1:],
                              skip_factor=skip_cpd,
                              conjugate=conjugate)

    lam_top = factors_top[0]
    lam_bottom = factors_bottom[0]

    if skip_factor != 0:
        return s * (lam_top[np.newaxis].T * lam_bottom)
    else:
        return s


def cpd_symmetrize(factors, permdict, adjust_scale=True):
    """
    Produce a symmetric CPD decomposition.
    :param factors: CPD factors
    :param permdict: dictionary of tuple: tuple pairs.
                     keys are tuples containing the permutation
                     values are tuples of any of('ident', 'neg', 'conj')
                     Identity permutation has to be excluded(added internally)
    :param adjust_scale: bool, default True
    If factors have to be scaled by the order of the permutation group

    Returns
    -------
    symm_factos: ndarray list, symmetrized CPD factors

    >>> a = cpd_initialize([3, 3, 4, 4], 3)
    >>> t1 = cpd_rebuild(a)
    >>> ts = 1/2 * (t1 + t1.transpose([1, 0, 3, 2]))
    >>> k = cpd_symmetrize(a, {(1, 0, 3, 2): ('ident', )})
    >>> np.allclose(ts, cpd_rebuild(k))
    True
    """

    if adjust_scale:
        nsym = len(permdict) + 1
        scaling_factor = pow(1 / nsym, 1 / len(factors))
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
    :param permdict: dictionary of tuple: tuple pairs.
                     keys are tuples containing the permutation
                     values are tuples of any of('ident', 'neg', 'conj')
                     Identity permutation has to be excluded(added internally)
    Returns
    -------
    symm_norm_factos: ndarray list, symmetrized nCPD factors

    """

    lam = norm_factors[0]
    factors = norm_factors[1:]

    nsym = len(permdict) + 1
    scaling_factor = pow(1 / nsym, 1 / len(factors))

    new_lam = scaling_factor * np.hstack((lam, ) * nsym)
    new_factors = cpd_symmetrize(factors, permdict, adjust_scale=False)

    return [new_lam, ] + new_factors


def unfold(tensor, mode=0):
    """Returns the mode-`mode` unfolding of `tensor`
    with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in
           ``range(0, tensor.ndim)``. If -1 is passed, then
           `tensor` is flattened

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    if mode > -1:
        return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))
    elif mode == -1:
        return tensor.ravel()
    else:
        raise ValueError('Wrong mode: {}'.format(mode))


def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`

        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape
        ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    if mode > -1:
        full_shape = list(shape)
        mode_dim = full_shape.pop(mode)
        full_shape.insert(0, mode_dim)
        return np.moveaxis(unfolded_tensor.reshape(full_shape), 0, mode)
    elif mode == -1:
        return unfolded_tensor.reshape(full_shape)
    else:
        raise ValueError('Wrong mode: {}'.format(mode))


def als_cpd_contract_cpd(factors_top, tensor_cpd,
                         skip_factor, conjugate=False):
    """
    Performs the first part of the ALS step on an (already CPD decomposed)
    tensor, which is to contract "external" indices of the tensor with all
    CPD factors of the decomposition except of one, denoted by skip_factor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor_cpd: iterable with CPD decomposition
               of the bottom (right) tensor
    :param skip_factor: int
               skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    Returns
    -------
           matrix
    """
    s = cpd_contract_free_cpd(factors_top, tensor_cpd,
                              skip_factor=skip_factor, conjugate=conjugate)
    return np.dot(tensor_cpd[skip_factor], s.T)


def als_ncpd_contract_ncpd(factors_top, tensor_ncpd,
                           skip_factor, conjugate=False):
    """
    Performs the first part of the ALS step on an (already nCPD decomposed)
    tensor, which is to contract "external" indices of the tensor with all
    nCPD factors of the decomposition except of one, denoted by skip_factor
    :param factors_top: iterable with nCPD decomposition
               of the top (left) tensor
    :param tensor_cpd: iterable with nCPD decomposition
               of the bottom (right) tensor
    :param skip_factor: int
               skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    Returns
    -------
           matrix
    """
    s = ncpd_contract_free_ncpd(factors_top, tensor_ncpd,
                                skip_factor=skip_factor, conjugate=conjugate)

    return np.dot(tensor_ncpd[skip_factor], s.T)


def als_ncpd_contract_dense(factors_top, tensor,
                            skip_factor, conjugate=False):
    """
    Performs the first part of the ALS step on a dense
    tensor, which is to contract "external" indices of the tensor with all
    nCPD factors of the decomposition except of one, denoted by skip_factor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor: tensor to contract with
    :param skip_factor: int
              skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    Returns
    -------
          matrix

    >>> kn = ncpd_initialize([3, 3, 4], 4)
    >>> k = kn[1:]
    >>> t = cpd_rebuild(k)
    >>> s1 = als_ncpd_contract_dense(kn, t, skip_factor=0)
    >>> s2 = np.dot(t.ravel(), khatrirao(k))
    >>> np.allclose(s1, s2)
    True
    """
    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    return np.dot(unfold(tensor, mode=skip_factor - 1),
                  khatrirao(factors_top, skip_matrix=skip_factor))


def als_cpd_contract_dense(factors_top, tensor,
                           skip_factor, conjugate=False):
    """
    Performs the first part of the ALS step on a dense
    tensor, which is to contract "external" indices of the tensor with all
    CPD factors of the decomposition except of one, denoted by skip_factor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor: tensor to contract with
    :param skip_factor: int
              skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    Returns
    -------
          matrix
    """
    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    return np.dot(unfold(tensor, skip_factor),
                  khatrirao(factors_top, skip_matrix=skip_factor))


def als_ncpd_pseudo_inverse(factors_top, factors_bottom,
                            skip_factor, conjugate=False, thresh=1e-10):
    """
    Calculates the pseudo inverse needed in the ALS algorithm.

    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param factors_bottom: iterable with CPD decomposition
               of the bottom (right) tensor
    :param skip_factor: int
              skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    :param thresh: float, default: 1e-10
               threshold used to calculate pseudo inverse
    Returns
    -------
          matrix

    >>> a = ncpd_initialize([3, 3, 4], 3)
    >>> b = ncpd_initialize([3, 3, 4], 4)
    >>> r = ncpd_contract_free_ncpd(a, b, skip_factor=2)
    >>> s = als_ncpd_pseudo_inverse(a, b, skip_factor=2)
    >>> np.allclose(np.linalg.pinv(r), s)
    True
    """

    factors_top = list(factors_top)
    factors_bottom = list(factors_bottom)
    if factors_top[0].ndim == 1:
        factors_top[0] = factors_top[0].reshape([1, -1])
    if factors_bottom[0].ndim == 1:
        factors_bottom[0] = factors_bottom[0].reshape([1, -1])

    rank1 = factors_top[0].shape[1]
    rank2 = factors_bottom[0].shape[1]

    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    pseudo_inverse = np.ones((rank1, rank2))
    for ii, (factor1, factor2) in enumerate(zip(factors_top, factors_bottom)):
        if ii != skip_factor:
            pseudo_inverse *= np.dot(factor1.T, factor2)

    return np.linalg.pinv(pseudo_inverse, thresh)


def als_cpd_pseudo_inverse(factors_top, factors_bottom,
                           skip_factor, conjugate=False, thresh=1e-10):
    """
    Calculates the pseudo inverse needed in the ALS algorithm.

    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param factors_bottom: iterable with CPD decomposition
               of the bottom (right) tensor
    :param skip_factor: int
              skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    :param thresh: float, default: 1e-10
               threshold used to calculate pseudo inverse
    Returns
    -------
          matrix

    >>> a = cpd_initialize([3, 3, 4], 3)
    >>> b = cpd_initialize([3, 3, 4], 4)
    >>> r = cpd_contract_free_cpd(a, b, skip_factor=2)
    >>> s = als_cpd_pseudo_inverse(a, b, skip_factor=2)
    >>> np.allclose(np.linalg.pinv(r), s)
    True
    """

    rank1 = factors_top[0].shape[1]
    rank2 = factors_bottom[0].shape[1]

    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    pseudo_inverse = np.ones((rank1, rank2))
    for ii, (factor1, factor2) in enumerate(zip(factors_top, factors_bottom)):
        if ii != skip_factor:
            pseudo_inverse *= np.dot(factor1.T, factor2)

    return np.linalg.pinv(pseudo_inverse, thresh)


def als_cpd_step_cpd(factors_top, tensor_cpd,
                     skip_factor, conjugate=False):
    """
    Performs one ALS update of the factor skip_factor
    for a CPD decomposed tensor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor_cpd: iterable with CPD decomposition
               of the bottom (right) tensor
    :param skip_factor: int
               skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    Returns
    -------
          matrix - updated factor skip_factor
    """

    r = als_cpd_contract_cpd(factors_top, tensor_cpd,
                             skip_factor, conjugate=conjugate)
    s = als_cpd_pseudo_inverse(factors_top, factors_top,
                               skip_factor, conjugate=conjugate,
                               thresh=1e-10)
    return np.dot(r, s)


def als_cpd_step_dense(factors_top, tensor,
                       skip_factor, conjugate=False):
    """
    Performs one ALS update of the factor skip_factor
    for a full tensor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor: ndarray, tensor to decompose
    :param skip_factor: int
               skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate CPD factors (needed for complex CPD)
    Returns
    -------
          matrix - updated factor skip_factor
    """

    r = als_cpd_contract_dense(factors_top, tensor,
                               skip_factor, conjugate=conjugate)
    s = als_cpd_pseudo_inverse(factors_top, factors_top,
                               skip_factor, conjugate=conjugate,
                               thresh=1e-10)
    return np.dot(r, s)


def als_cpd(guess, tensor_cpd, complex_cpd=False, max_cycle=100):
    """
    Run an ALS algorithm on the CPD decomposed tensor
    :param guess: iterable with initial guess for CPD decomposition
    :param tensor_cpd: iterable with CPD decomposition
                       of the target tensor
    :param complex_cpd: bool, default: False
               if complex decomposition is done
               (guess should also be complex, or this will not be enforced)
    :param max_cycle: maximal number of iterations

    Returns
    -------
    Iterable with CPD decomposition

    >>> a = cpd_initialize([3, 3, 4], 3)
    >>> b = cpd_initialize([3, 3, 4], 10)
    >>> k = als_cpd(b, a, max_cycle=100)
    >>> np.allclose(cpd_rebuild(a), cpd_rebuild(k), 1e-10)
    True
    """
    factors = [factor for factor in guess]  # copy the guess

    for iteration in range(max_cycle):
        for mode in range(len(tensor_cpd)):
            factor = als_cpd_step_cpd(factors, tensor_cpd,
                                      skip_factor=mode,
                                      conjugate=complex_cpd)
            factors[mode] = factor

    return factors


def als_dense(guess, tensor, complex_cpd=False, max_cycle=100):
    """
    Run an ALS algorithm on a dense tensor

    :param guess: iterable with initial guess for CPD decomposition
    :param tensor: ndarray, target tensor
    :param complex_cpd: bool, default: False
               if complex decomposition is done
               (guess should also be complex, or this will not be enforced)
    :param max_cycle: maximal number of iterations

    Returns
    -------
    Iterable with CPD decomposition

    >>> a = cpd_rebuild(cpd_initialize([3, 3, 4], 3))
    >>> b = cpd_initialize([3, 3, 4], 10)
    >>> k = als_dense(b, a, max_cycle=100)
    >>> np.allclose(a, cpd_rebuild(k), 1e-10)
    True
    """
    factors = [factor for factor in guess]  # copy the guess

    for iteration in range(max_cycle):
        for mode in range(tensor.ndim):
            factor = als_cpd_step_dense(factors, tensor,
                                        skip_factor=mode,
                                        conjugate=complex_cpd)
            factors[mode] = factor

    return factors


def _demonstration_symmetry_rank():
    """
    This function demonstrates that a symmetrized
    tensor has a rank which is much larger than the
    initial rank. Errors in the CPD decomposition
    of the symmetrized tensor with a rank equal
    to the rank of its unsymmetric part are not small,
    and factors of the unsymmetric part are a bad
    initial guess for the CPD of the symmetrized tensor
    """
    a = cpd_initialize([3, 3, 4, 4], 3)
    t1 = cpd_rebuild(a)
    ts = 1 / 2 * (t1 + t1.transpose([1, 0, 3, 2]))
    k = cpd_symmetrize(a, {(1, 0, 3, 2): ('ident', )})
    z0 = ts - cpd_rebuild(k)
    n = als_cpd(a, k, max_cycle=400)
    m = als_dense(a, ts, max_cycle=400)
    z1 = ts - cpd_rebuild(n)
    z2 = ts - cpd_rebuild(m)

    print(np.linalg.norm(z0.ravel))
    print(np.linalg.norm(z1.ravel))
    print(np.linalg.norm(z2.ravel))
