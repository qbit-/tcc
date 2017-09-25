import numpy as np
from collections import deque
from functools import reduce
import pdb
"""
  Forms the DIIS guess for T, which we insert to calculate G.
  Remember that DIIS writes
  T* ~ c(i) T(i)
  We have the constraint
     Sum(c(i)) = 1
  and we get the c(i) by minimizing a residual, effectively:
     r(i) = H.T(i) - G[T(i)]
     R = c(i) r(i)
  then minimize
     R*R = Sum(c(i)* c(j) B(i,j))
     B(i,j) = r(i)*.r(j)

Try the real case:
     L = Sum[c(i) c(j) B(ij)] + 2 y (Sum[c(i)] - 1)
  Then
     dL/dy = 0    => Sum(c(i)) = 1
     dL/dc(i) = 0 => 2 Sum(B(ij) cj) + 2 y = 0.
  We can organize this as a matrix equation:
     [B(i,j)   1(j)] [c(j)] = [0(j)]
     [1(i)       0 ] [ y  ] = [ 1  ]
  which we define as M.C = V.

  Then consider the complex case (maybe? hopefully?):
      L = Sum[c(i)* c(j) B(ij)] + y (Sum[c(i)]-1)
                                + x (Sum[c(i)*]-1)
  Using that B is Hermitian, we find that
      L* = Sum[c(i)* c(j) B(ij)] + x* (Sum[c(i)]-1)
                                 + y* (Sum[c(i)*]-1)
  so, for L to be real, we must have x = y*.  Writing
      y = u + i v
      x = u - i v
  we would have
      L = Sum[c(i)* c(j) B(ij)] + u (Sum[c(i)+c(i)*]-2)
                                + i v Sum[c(i)-c(i)*]
  If I put this together, we get the following equations:
     Bij  cj  + u - i v = 0
     Bij* cj* + u + i v = 0
     Sum(ci + ci*)      = 2
     i Sum(ci - ci*)    = 0
  which I can group into the nice matrix equation
     [B   0   1  -i] [c]     [0]
     [0   B*  1   i] [c*]    [0]
     [1   1   0   0] [u]  =  [2]
     [i  -i   0   0] [v]     [0]
  Note that the matrix on the left-hand-side is Hermitian.
  Alternatively, we could try getting rid of the last row and
  column, which is just ignoring Sum(c-c*) = 0.
"""


class diis_single:
    """
    This class implements a DIIS extrapolation.
    """

    def __init__(self, ndiis, dtype='real'):
        """
        :param ndiis: number of diis vectors to use
        :param dtype: field to use. May be 'real' or 'complex'. Real will
        force coefficients to stay real
        """

        self._predictors = deque(maxlen=ndiis)
        self._variables = deque(maxlen=ndiis)
        self._ndiis = ndiis
        self._coefftable = np.zeros([ndiis, ndiis])
        if dtype in set(('real', 'complex')):
            self._dtype = dtype
        else:
            raise ValueError('Unsupported dtype: {}'.format(dtype))
        self._initialized = False

    @property
    def ready(self):
        """
        Returns true is DIIS is ready to predict next approximate
        """
        if len(self._predictors) == self._ndiis:
            return True
        else:
            return False

    def predictors(self):
        """
        Returns current predictors
        :rtype:
        """
        return self._predictors

    def variables(self):
        """
        Returns current (already predicted) variables
        :rtype:
        """
        return self._variables

    def push_predictor(self, predictor):
        """
        Push next pair of (predictor, variable) to the DIIS buffer
        :param predictor: current predictor
        """

        self._predictors.append(predictor)

        if self._initialized:
            self._update_coefftable(predictor)
        elif self.ready:
            self._build_coefftable()

    def _update_coefftable(self, predictor):
        """
        Updates the coefficient table
        """
        # calculate new row/column of the symmetric coefftable
        overlaps = np.array([np.inner(vec.flatten(),
                                      predictor.flatten())
                             for vec in self._predictors])

        # append last column/row to coefftable and drop the first row/col
        coefftable = np.vstack(
            (np.hstack((self._coefftable[1:, 1:],
                        np.reshape(overlaps[:-1], (-1, 1)))
                       ),
             overlaps)
        )

        self._coefftable = coefftable

    def _build_coefftable(self):
        """
        Builds the coefficient table for the first time
        """
        n = self._ndiis
        coefftable = np.zeros((n, n))
        for ii in range(n):
            for jj in range(n):
                coefftable[ii, jj] = np.inner(self._predictors[ii].flatten(),
                                              self._predictors[jj].flatten())
        self._coefftable = coefftable
        self._initialized = True


    def predict(self):
        """
        Returns next extrapolated variable
        :rtype: variable
        """
        n = self._ndiis
        if self._dtype == 'complex':
            A = np.bmat([[self._coefftable,
                          np.zeros((n, n), np.ones(n, 1), -1j * np.ones(n, 1))],
                         [np.zeros((n, n)), self._coefftable.T.conj(),
                          np.ones((n, 1)), 1j * np.ones((n, 1))],
                         [np.ones((1, n)), np.ones((1, n)),
                          np.zeros((1, 1)), np.zeros((1, 1))],
                         [np.ones((1, n)), -1j * np.ones((1, n)),
                          np.zeros((1, 1)), np.zeros((1, 1))]])
            b = np.zeros((2 * n + 1, 1))
            b[-2] = 2
        elif self._dtype == 'real':
            A = np.bmat([[self._coefftable, np.ones((n, 1))],
                         [np.ones((1, n)), np.zeros((1, 1))]])
            b = np.zeros((n + 1, 1))
            b[-1] = 1

        x = np.linalg.solve(A, b)
        c = x[:n]
        
        return reduce(lambda x, y: x + y, (self._variables[ii] * c[ii]
                                           for ii in range(n)))

    def push_variable(self, variable):
        """
        Takes current predictor and variable and returns
        next predicted variable
        :param variable: new variable
        """
        self._variables.append(variable)


class diis_multiple(diis_single):
    """
    This class implements diis for multiple variables stored
    in a Tensors dictionary. Each variable gets it's own coefficients from
    it's own predictors
    """

    def __init__(self, nvar, ndiis, dtype='real'):
        """
        Constructs multiple DIIS instances
        """
        self.v = []
        self._nvar = nvar
        for ii in range(nvar):
            self.v.extend([diis_single(ndiis, dtype), ])

    @property
    def ready(self):
        """
        Returns true is DIIS is ready to predict next approximate        
        """

        return all(self.v[ii].ready for ii in range(self._nvar))

    def predictors(self):
        """
        Returns an iterator for current predictors
        """
        return tuple(self.v[ii].predictors() for ii in range(self._nvar))

    def selected_predictors(self, select):
        """
        Returns predictors for a selected variable
        """
        return self.v[select].predictors()

    def variables(self):
        """
        Returns current (already predicted) variables
        :rtype:
        """
        return tuple(self.v[ii].variables() for ii in range(self._nvar))

    def selected_variables(self, select):
        """
        Returns predictors for a selected variable
        """
        return self.v[select].variables()

    def push_predictor(self, predictors):
        """
        Pushes predictors into deques
        :param predictor: is a dict-like structure (array of tuples)
                          which holds predictors
        """
        self._check_argument(predictors)

        for ii, key in enumerate(predictors):
            self.v[ii]._pred_key = key
            self.v[ii].push_predictor(predictors[key])

    def push_variable(self, variables):
        """
        Pushes variables into their deques
        :param variables: is a dict-like structure (array of tuples)
                          which holds variables
        """
        self._check_argument(variables)

        for ii, key in enumerate(variables):
            self.v[ii]._var_key = key
            self.v[ii].push_variable(variables[key])

    def _check_argument(self, x):
        """
        Simply checks the length of the supplied argumet
        """
        if len(x) != len(self.v):
            raise ValueError('The length of predictor\
            does not match the length of variable:\
            {} != {}'.format(len(x), len(self.v)))

    def predict(self):
        """
        Predict next set of variables. Returns a generator
        """
        return dict((self.v[ii]._var_key, self.v[ii].predict())
                    for ii in range(self._nvar))

    @property
    def _update_coefftable(self):
        raise AttributeError(
            "'diis_multiple' object has no attribute '_update_coefftable'")

