import numpy as np
import collections


def khatrirao(matrices, skip_matrix=None, reverse=False):
    """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.
        (see [1]_ for more details).

    Parameters
    ----------
    :param matrices: ndarray list
        list of matrices with the same number of columns, i.e.::

            for i in len(matrices):
                matrices[i].shape = (n_i, m)

    :param skip_matrix: None or int, optional, default is None
        if not None, index of a matrix to skip

    :param reverse: bool, optional
        if True, the order of the matrices is reversed

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the product.

    Notes
    -----
    Mathematically:

    .. math::
         \\text{If every matrix } U_k \\text{ is of size } (I_k \\times R),\\\\
         \\text{Then } \\left(U_1 \\bigodot \\cdots \\bigodot U_n \\right) \\text{ is of size } (\\prod_{k=1}^n I_k \\times R)


    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.

    >>> import numpy as np
    >>> a = np.array([[1,3],[2,4]])
    >>> b = np.array([[5,6],[7,8],[9,10]])
    >>> np.sum(khatrirao((a,b)),1)[2]
    31.0
    """
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    n_columns = matrices[0].shape[1]

    # Optional part, testing whether the matrices have the proper size
    for i, matrix in enumerate(matrices):
        if matrix.ndim != 2:
            raise ValueError('All the matrices must have exactly 2 dimensions!'
                             'Matrix {} has dimension {} != 2.'.format(
                                 i, matrix.ndim))
        if matrix.shape[1] != n_columns:
            raise ValueError('All matrices must have same number of columns!'
                             'Matrix {} has {} columns != {}.'.format(
                                 i, matrix.shape[1], n_columns))

    n_factors = len(matrices)

    if reverse:
        matrices = matrices[::-1]
        # Note: we do NOT use .reverse() which would reverse matrices even outside this function

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))

def unroll_iterable(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from unroll_iterable(el)
        else:
            yield el


def np_container_structure(l):
    return tuple(el.shape for el in l)


def merge_np_container(l):
    return np.hstack(el.flatten() for el in l)


def unmerge_np_tuple(shapes, t):
    offsets = _get_offsets(shapes)
    for ii in range(len(shapes)):
        start, end = offsets[ii]
        yield t[start:end].reshape(shapes[ii])


def _get_offsets(shapes):
    lengths = [0, ] + [np.prod(el) for el in shapes]
    return tuple(zip(np.cumsum(lengths)[:-1], np.cumsum(lengths)[1:]))


def unmerge_np_container(cont_type, s, t):
    return cont_type(*unmerge_np_tuple(s, t))
