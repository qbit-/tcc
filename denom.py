def cc_denom(fock, ndim, ordering='mul', kind='full'):
    """
    Builds a cc denominator. Kind may be either full or
    cpd.
    """
    e_i = fock.oo.diagonal()
    e_a = fock.vv.diagonal()

    if kind == 'full':
        return _construct_cc_denom_full(e_a, e_i, ndim, ordering)
    elif kind == 'cpd':
        return _construct_cc_denom_cpd(e_a, e_i, ndim, ordering)

    
def _construct_cc_denom_full(fv, fo, ndim, ordering):
    """
    Builds an energy denominator
    from diagonals of occupied and virtual part of the
    Fock matrix. Denominator is defined as
    ordering == 'dir'
    d_{ab..ij..} = (fv_a + fv_b + .. - fo_i - fo_j - ..)
    ordering == 'mul'
    d_{aibj..} = (fv_a - fo_i + fv_b - fo_j + ..)

    >>> import numpy as np
    >>> a = np.array([1, 2])
    >>> b = np.array([3, 4])
    >>> c = _construct_cc_denom_full(fv, fo, 4, 'mul')
    >>> d = _construct_cc_denom_full(fv, fo, 4, 'dir')
    >>> np.allclose(c, d.transpose([0,2,1,3])
    True
    >>> c = _construct_cc_denom_full(fv, fo, 2, 'mul')
    >>> c[1,0]
    3
    """
    if ndim % 2 != 0:
        raise ValueError('Ndim is not an even integer: {}'.format(ndim))

    npair = ndim // 2

    if ordering == 'dir':
        vecs = [+fv.reshape(_get_expanded_shape_tuple(ii, ndim))
                for ii in range(npair)]
        vecs.extend([-fo.reshape(_get_expanded_shape_tuple(ii + npair, ndim))
                     for ii in range(npair)])

    elif ordering == 'mul':
        vecs = (+fv.reshape(_get_expanded_shape_tuple(2 * ii, ndim))
                - fo.reshape(_get_expanded_shape_tuple(2 * ii + 1, ndim))
                for ii in range(npair))
    else:
        raise ValueError('Unknown ordering: {}'.format(ordering))

    return sum(vecs)

def _get_expanded_shape_tuple(dim, ndim):
    """
    Returns a tuple for reshaping a vector into
    ndim - array
    >>> v = _get_expanded_shape_tuple(1,2)
    >>> v
    (1,-1)
    """

    if (dim >= 0) and (dim < ndim):
        return (1,) * (dim) + (-1,) + (1,) * (ndim - dim - 1)
    else:
        raise ValueError(
            'Invalid dimension: {}, not in 0..{}'.format(dim, ndim - 1))

