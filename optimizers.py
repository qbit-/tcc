"""
Here gradient descent step generators are defined
"""

import numpy as np
from tcc.tensors import Tensors


class MomentumOptimizer():
    """
    Momentum optimizer
    """

    def __init__(self, x, beta=0.9, correct_bias=False):
        """
        Initialize momentum optimizer
        :param beta: damping factor
        :param correct_bias: if bias has to be corrected.
                             If yes then p is required
        :param p: step number
        """
        self.v = x.map(np.zeros_like)
        self.beta = beta

        self._correct_bias = correct_bias
        self._step_number = 0

    def update(self, x):
        """
        Generate a momentum update
        """
        self._step_number = self._step_number + 1

        v = self.v * self.beta + x * (1 - self.beta)
        if self._correct_bias:
            v = v / (1 - self.beta ** (self._step_number))
        self.v = v

        return v


class RMSPropOptimizer(MomentumOptimizer):
    """
    RMSProp optimizer
    """

    def __init__(self, x, beta=0.9, epsilon=1e-10, correct_bias=False):
        """
        Initialize momentum optimizer
        :param beta: damping factor
        :param correct_bias: if bias has to be corrected.
                             If yes then p is required
        :param p: step number
        """
        self.v = x.map(np.zeros_like)
        self.beta = beta
        self._epsilon = epsilon

        self._correct_bias = correct_bias
        self._step_number = 0

    def update(self, x):
        """
        Generate RMSProp update
        """
        self._step_number = self._step_number + 1

        v = self.v * self.beta + x ** 2 * (1 - self.beta)
        if self._correct_bias:
            v = v / (1 - self.beta ** (self._step_number))
        self.v = v

        return x / ((v + self._epsilon).map(np.sqrt))


class AdamOptimizer():
    """
    Combined Momentum and RMSProp methods
    """

    def __init__(self, x, beta=0.9, gamma=0.999, epsilon=1e-10):
        """
        Initialize the optimizer
        :param beta: momentum coefficient
        :param gamma: rmsprop coefficient
        :param epsilon: threshold to avoid division by zero 
        """
        self.v = x.map(np.zeros_like)
        self.s = x.map(np.zeros_like)
        self.beta = beta
        self.gamma = gamma
        self._epsilon = epsilon

        self._step_number = 0

    def update(self, x):
        """
        Adam step
        """
        self._step_number = self._step_number + 1

        v = self.v * self.beta + x * (1 - self.beta)
        v_corr = v / (1 - self.beta ** (self._step_number))
        self.v = v_corr

        s = self.s * self.gamma + x ** 2 * (1 - self.gamma)
        s_corr = s / (1 - self.gamma ** (self._step_number))
        self.s = s_corr

        return v_corr / (s_corr.map(np.sqrt) + self._epsilon)


class AdaMaxOptimizer():
    """
    Generalizes over Adam by using infinity norm
    """

    def __init__(self, x, beta=0.9, gamma=0.999):
        """
        Initialize the optimizer
        :param beta: momentum coefficient
        :param gamma: rmsprop coefficient
        """
        self.v = x.map(np.zeros_like)
        self.s = x.map(np.zeros_like)
        self.beta = beta
        self.gamma = gamma

        self._step_number = 0

    def update(self, x):
        """
        Adam step
        """
        self._step_number = self._step_number + 1

        v = self.v * self.beta + x * (1 - self.beta)
        v_corr = v / (1 - self.beta ** (self._step_number))
        self.v = v_corr

        s = (self.s * self.gamma)._map_binary(
            x.map(np.abs), lambda x, y: np.maximum(x, y))
        self.s = s

        return v_corr / s


def initialize(use_optimizer, x, **optimizer_kwargs):
    """
    Initialize a proper optmizer object
    """
    dispatcher = {
        'momentum': MomentumOptimizer,
        'rmsprop': RMSPropOptimizer,
        'adam': AdamOptimizer,
        'adamax': AdaMaxOptimizer,
    }
    try:
        optimizer_cls = dispatcher[use_optimizer]
    except:
        raise ValueError('Unknown optimizer: {}'.format(use_optimizer))

    return optimizer_cls(x, **optimizer_kwargs)


if __name__ == '__main__':
    from tcc.tensors import Tensors
    import numpy as np
    import optimizers
    a = Tensors(s=np.random.rand(2, 3), d=Tensors(
        a=np.ones((2, 2)), b=np.eye(2, 2)))

    opt = optimizers.initialize('adam', a)
