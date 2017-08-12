import numpy as np
import collections


def khatrirao(matrices, reverse=False):
    """
    Compute the Khatri-Rao product of all matrices in list "matrices
    :param matrices: iterable with matrices
    :param reverse: if reversed order of multiplications is needed
    Reverse=True to be compliant with Matlab code

    >>> import numpy as np
    >>> a = np.array([[1,3],[2,4]])
    >>> b = np.array([[5,6],[7,8],[9,10]])
    >>> np.sum(khatrirao((a,b)),1)[2]
    31.0
    """
    # If reverse is true, does the product in reverse order.
    matorder = range(len(matrices)) if not reverse else list(
        reversed(range(len(matrices))))

    # Error checking on matrices; compute number of rows in result.
    # N = number of columns (must be same for each input)
    N = matrices[0].shape[1]
    # Compute number of rows in resulting matrix
    # After the loop, M = number of rows in result.
    M = 1
    for i in matorder:
        if matrices[i].ndim != 2:
            raise ValueError("Each argument must be a matrix.")
        if N != (matrices[i].shape)[1]:
            raise ValueError(
                "All matrices must have the same number of columns.")
        M *= (matrices[i].shape)[0]

    # Computation
    # Preallocate result.
    P = np.zeros((M, N))

    # n loops over all column indices
    for n in range(N):
        # ab = nth col of first matrix to consider
        ab = matrices[matorder[0]][:, n]
        # loop through matrices
        for i in matorder[1:]:
            # Compute outer product of nth columns
            ab = np.outer(matrices[i][:, n], ab[:])
        # Fill nth column of P with flattened result
        P[:, n] = ab.flatten()
    return P


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
