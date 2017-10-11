from itertools import chain
import numpy as np


def cc_denom(fock, ndim, ordering='mul', kind='full'):
    """
    Builds a cc denominator. Kind may be either full or
    cpd.
    :param focka: fock matrix for alpha spins
    :param fockb: fock matrix for beta spins
    :param nalpha: number of alpha fock matrices in the denominator
    :param ndim: dimension
    :param ordering: 'mul' or 'dir' - Dirac or Mulliken ordering
    :param kind: 'full' or 'cpd' - full tensor or CPD factors
    """
    e_i = fock.oo.diagonal()
    e_a = fock.vv.diagonal()
    npair = ndim // 2

    if kind == 'full':
        return 1 / _construct_cc_denom_full(
            (e_a,) * npair + (e_i,) * npair, ordering)
    elif kind == 'cpd':
        return _construct_cc_denom_cpd(
            (e_a,) * npair + (e_i,) * npair, ordering)


def cc_denom_spin(focka, fockb, nalpha, ndim, ordering='mul', kind='full'):
    """
    Builds a cc denominator for UHF theories, e.g. with distinct
    alpha and beta parts of the fock matrix.
    :param focka: fock matrix for alpha spins
    :param fockb: fock matrix for beta spins
    :param nalpha: number of alpha fock matrices in the denominator
    :param ndim: dimension
    :param ordering: 'mul' or 'dir' - Dirac or Mulliken ordering
    :param kind: 'full' or 'cpd' - full tensor or CPD factors
    """
    e_a_i = focka.oo.diagonal()
    e_b_i = fockb.oo.diagonal()
    e_a_a = focka.vv.diagonal()
    e_b_a = fockb.vv.diagonal()

    npair = ndim // 2
    if nalpha > npair:
        raise ValueError('nalpha > ndim/2')

    if kind == 'full':
        return 1 / _construct_cc_denom_full(
            (e_a_a,) * nalpha
            + (e_b_a,) * (npair - nalpha)
            + (e_a_i,) * nalpha
            + (e_b_i,) * (npair - nalpha), ordering)
    elif kind == 'cpd':
        return _construct_cc_denom_cpd(
            (e_a_a,) * nalpha
            + (e_b_a,) * (npair - nalpha)
            + (e_a_i,) * nalpha
            + (e_b_i,) * (npair - nalpha), ordering)


def _construct_cc_denom_full(fock_diags, ordering):
    """
    Builds an energy denominator
    from diagonals of occupied and virtual part of the
    Fock matrix. Denominator is defined as
    ordering == 'dir'
    d_{ab..ij..} = (fv_a + fv_b + .. - fo_i - fo_j - ..)
    ordering == 'mul'
    d_{aibj..} = (fv_a - fo_i + fv_b - fo_j + ..)

    :param fock_diags:  an iterable of diagonal parts of Fock matrices
    :param ordering: ordering of elements
    :rtype: CC denominator tensor

    >>> a = np.array([1, 2])
    >>> b = np.array([3, 4])
    >>> c = _construct_cc_denom_full((a,)*2 +(b,)*2, 'mul')
    >>> d = _construct_cc_denom_full((a,)*2 +(b,)*2, 'dir')
    >>> np.allclose(c, d.transpose([0,2,1,3]))
    True
    >>> c = _construct_cc_denom_full((a,) +(b,), 'mul')
    >>> c[1,0]
    -1
    """
    ndim = len(fock_diags)
    if ndim % 2 != 0:
        raise ValueError(
            'Odd number of vectors in fock_diags: {}'.format(ndim))

    npair = ndim // 2

    if ordering == 'dir':
        vecs = tuple(
            chain(
                (+fock_diags[ii].reshape(_get_expanded_shape_tuple(ii, ndim))
                 for ii in range(npair)),
                (-fock_diags[ii + npair].reshape(_get_expanded_shape_tuple(ii + npair, ndim))
                 for ii in range(npair))
            )
        )
    elif ordering == 'mul':
        vecs = tuple(
            chain.from_iterable(
                (+fock_diags[ii].reshape(_get_expanded_shape_tuple(2 * ii, ndim)),
                 - fock_diags[ii + npair].reshape(_get_expanded_shape_tuple(2 * ii + 1, ndim))
                 ) for ii in range(npair))
        )
    else:
        raise ValueError('Unknown ordering: {}'.format(ordering))

    return sum(vecs)


def _get_expanded_shape_tuple(dim, ndim):
    """
    Returns a tuple for reshaping a vector into
    ndim - array
    >>> v = _get_expanded_shape_tuple(1,2)
    >>> v
    (1, -1)
    """

    if (dim >= 0) and (dim < ndim):
        return (1,) * (dim) + (-1,) + (1,) * (ndim - dim - 1)
    else:
        raise ValueError(
            'Invalid dimension: {}, not in 0..{}'.format(dim, ndim - 1))


def _construct_cc_denom_cpd(fock_diags, ordering, epsilon=1e-12):
    """
    Builds CC energy denominator in CPD forma
    from diagonals of occupied and virtual part of the
    Fock matrix. Denominator is defined as
    ordering == 'dir'
    d_{ab..ij..} = 1 / (fv_a + fv_b + .. - fo_i - fo_j - ..) =
    sum_{p} dv_a^p * dv_b^p * .. * do_i^p * do_j^p * ..)  

    ordering == 'mul'
    d_{aibj..} = (fv_a - fo_i + fv_b - fo_j + ..)
    sum_{p} dv_a^p * do_i^p * .. * dv_b^p * do_j^p * ..)  

    :param fock_diags:  an iterable of diagonal parts of Fock matrices
    :param ordering: ordering of elements
    :param epsilon: accuracy of quadrature (default: 1e-10)
    :rtype: tuple of CPD factors of CC denominator tensor

    >>> from tcc.cpd import cpd_rebuild
    >>> a, b = np.array([3, 10]), np.array([-1, -2])
    >>> f = (a,)*2 +(b,)*2
    >>> c = _construct_cc_denom_full(f, 'mul')
    >>> d = _construct_cc_denom_cpd(f, 'mul', 1e-14)
    >>> np.allclose(1/c - cpd_rebuild(d), 1e-10)
    True
    >>> e = _construct_cc_denom_cpd(f, 'dir', 1e-14)
    >>> np.allclose(cpd_rebuild(d) - cpd_rebuild((e[0],e[2],e[1],e[3])), 1e-10)
    True
    """

    ndim = len(fock_diags)
    if ndim % 2 != 0:
        raise ValueError(
            'Odd number of vectors in fock_diags: {}'.format(ndim))

    npair = ndim // 2

    sorted_vecs = tuple(sorted(vec) for vec in fock_diags)

    # This check is not perfect, as accidential degeneracy may still happen?
    min_spreads = [sorted_vecs[ii][0] - sorted_vecs[ii + npair][-1] for ii in
                   range(npair)]
    max_spreads = [sorted_vecs[ii][-1] - sorted_vecs[ii + npair][0] for ii in
                   range(npair)]
    A = min(min_spreads)
    B = max(max_spreads)

    R = B / A
    # get quadrature coefficients and exponents
    c, a = _load_1_x_quadrature(R, epsilon)

    # properly normalize for the size of tensors we have
    cc = (c / A)**(1 / ndim)
    aa = a.reshape(-1, 1) / A

    if ordering == 'dir':
        vecs = tuple(
            chain(
                (np.diagflat(cc).dot(np.exp(np.kron(-aa, fock_diags[ii])))
                 for ii in range(npair)),
                (np.diagflat(cc).dot(np.exp(np.kron(-aa, -fock_diags[ii + npair])))
                 for ii in range(npair))
            )
        )

    elif ordering == 'mul':
        vecs = tuple(
            chain.from_iterable(
                (np.diagflat(cc).dot(np.exp(np.kron(-aa, fock_diags[ii]))),
                 np.diagflat(cc).dot(np.exp(np.kron(-aa, -fock_diags[ii + npair]))))
                for ii in range(npair)
            )
        )
    else:
        raise ValueError('Unknown ordering: {}'.format(ordering))

    return tuple(vec.transpose() for vec in vecs)


def _load_1_x_quadrature(R, epsilon, prefix=None):
    """
    Loads appropriate quadrature parameters for 1/x function based on the
    value of range metric R and the required accuracy epsilon.
    :param R: range metric, as defined in :func:`_construct_cc_denom_cpd`
    :param epsilon: accuracy of the quadrature
    :param prefix: path to directory containing tcc package
    :rtype: quadrature coefficients, quadrature exponents (as np vectors)
    """
    import os
    import h5py
    if prefix is None:
        prefix = os.path.dirname(os.path.abspath(__file__))
    basepath = prefix + '/data/quadratures/1_x'

    f = h5py.File(basepath + '/quad_idx_table.h5', 'r')
    quad_file_name, num_k, max_err = _find_1_x_quadname(
        R, epsilon, f['quad_idx_table']
    )
    f.close()
    c, a = _read_quadrature_from_file(basepath + '/' + quad_file_name)

    return c, a


def _read_quadrature_from_file(filename):
    """
    This is a special function to read quadrature parameters from
    formatted text files
    :param filename: full path to the quadrature file
    :rtype: omega, alpha - factors and their exponents from quadrature file (as
    np arrays)
    """
    import re
    import mmap
    patt_omega = b'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*{omega\[(\d+)'
    patt_alpha = b'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*{alpha\[(\d+)'

    with open(filename, 'r') as fp:
        data = mmap.mmap(fp.fileno(), 0, prot=mmap.PROT_READ)
        omega_indexed = np.asarray(re.findall(
            patt_omega, data), dtype='float64')
        data.seek(0)
        alpha_indexed = np.asarray(re.findall(
            patt_alpha, data), dtype='float64')
        data.close()

    # Possibly permute vector entries according to the index we have read
    # (this is never encountered in the datafiles, but we do it for "just in case")

    omega = omega_indexed[:, 0][omega_indexed[:, 1].astype(int) - 1]
    alpha = alpha_indexed[:, 0][alpha_indexed[:, 1].astype(int) - 1]
    return omega, alpha


def _find_1_x_quadname(R, epsilon, quad_idx_table):
    """
    Finds appropriate quadrature file name based on the
    quad_idx_table
    :param R: range metric, as defined in :func:`_construct_cc_denom_cpd`
    :param epsilon: accuracy of the quadrature
    :param quad_idx_table: index table for available quadratures
    :rtype: quadrature file name, length of quadrature, maximal error 

    >>> import h5py, os
    >>> basepath = os.path.dirname(os.path.abspath(__file__))
    >>> f = h5py.File(basepath+'/data/quadratures/1_x/quad_idx_table.h5', 'r')
    >>> d = f['quad_idx_table']
    >>> q, k, e = _find_1_x_quadname(4000, 1e-5, d)
    >>> f.close()
    >>> print('{}, {:d}, {:e}'.format(q, k, e))
    1_xk10_4E3, 10, 6.284000e-06
    """

    err_table = np.array(quad_idx_table[:, 1:])

    # Find index for entries with r >= R
    rs = np.array(quad_idx_table[:, 0])
    idxR = np.argmin(rs < R)

    # Extract table with possible quadratures
    err_table[:idxR, :] = 0
    err_table[err_table > epsilon] = 0

    row, col = np.nonzero(err_table)

    if len(row) == 0 and len(col) == 0:
        raise ValueError('Quadrature not found')

    k_min_col = np.min(col)
    k_min = k_min_col + 1  # Since python indexing is 0-based, and k is at least 1
    col_k_min_idx = np.argmin(col)
    R_min_idx = row[col_k_min_idx]
    R_min = np.asscalar(rs[R_min_idx])

    mantissa, exponent = _mantissa_exponent10(R_min)

    quad_name = '1_xk{:02d}_{:d}E{:d}'.format(
        k_min, int(mantissa), int(exponent)
    )
    max_err = err_table[R_min_idx, k_min_col]
    # print('Minimal available R with requested epsilon:'
    #       + '{:e}\nk={:d}, maximal error={:e}\n'.format(
    #           R_min, k_min, err_table[R_min_idx,k_min_col]
    #       )
    # )

    return quad_name, k_min, max_err


def _mantissa_exponent10(val):
    """
    Returns the mantissa and exponent of a real base 10 argument

    >>> _mantissa_exponent10(-0.1)
    (-1.0, -1.0)
    >>> _mantissa_exponent10(1.)
    (1.0, 0.0)
    """
    sgn = np.sign(val)
    exponent = np.asscalar(np.fix(np.log10(abs(val))))
    mantissa = sgn * 10**(np.asscalar(np.log10(abs(val))) - exponent)
    if abs(mantissa) < 1:
        mantissa = mantissa * 10
        exponent = exponent - 1

    # print('\n\t{:6.4f} x E{:+04d}\n'.format(mant,expnt))
    return mantissa, exponent
