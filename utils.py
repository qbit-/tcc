import numpy as np

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
    t = khatrirao(tuple(factors[ii] for ii in range(N-1)), True).dot(factors[N-1].transpose())
    return t.reshape(tensor_shape)


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
    matorder = range(len(matrices)) if not reverse else list(reversed(range(len(matrices))))

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
            raise ValueError("All matrices must have the same number of columns.")
        M *= (matrices[i].shape)[0]

    # Computation
    # Preallocate result.
    P = np.zeros((M, N))
    
    # n loops over all column indices
    for n in range(N):
        # ab = nth col of first matrix to consider
        ab = matrices[matorder[0]][:,n]
        # loop through matrices
        for i in matorder[1:]:
            # Compute outer product of nth columns
            ab = np.outer(matrices[i][:,n], ab[:])
        # Fill nth column of P with flattened result
        P[:,n] = ab.flatten()
    return P
