""" """
from collections import OrderedDict
import numpy as np
from operator import add, sub, mul, truediv


class Tensors(dict):
    """A simple container for tensors supporting mapping and product.
    """

    __slots__ = ()

    def __getattr__(self, key):
        return self[key]

    def __repr__(self):
        """ Representation for Tensors
        """
        return '{0}({1})'.format(self.__class__.__name__, dict.__repr__(self))

    def map(self, func):
        """Map the function to the tensors with structure kept"""
        res = Tensors()
        for k, v in self.items():
            res[k] = func(v) if not isinstance(v, Tensors) else v.map(func)
        return res

    def zip(self, other):
        """Zip the two tensor containers.

        The two tensor containers are required to have exactly same structure.
        """

        res = Tensors()
        proced = set(other.keys())
        for k, v in self.items():
            err_msg = 'Unmatching Tensors key {}'.format(k)
            if k not in other:
                raise ValueError(err_msg)
            o = other[k]
            if isinstance(v, Tensors):
                if not isinstance(o, Tensors):
                    raise ValueError(err_msg)
                res[k] = v.zip(o)
            else:
                res[k] = (
                    v if isinstance(v, tuple) else (v, )
                ) + (
                    o if isinstance(o, tuple) else (o, )
                )

            proced.remove(k)

        if len(proced) != 0:
            raise ValueError('Unmatching Tensors keys {}'.format(proced))

        return res

    def _map_binary(self, other, func):
        """
        Implements a function of two arguments for Tensors
        """
        if isinstance(other, Tensors):
            return self.zip(other).map(lambda x: func(x[0], x[1]))
        else:
            return self.map(lambda x: func(x, other))

    # Specific math operations.

    def sqrt(self):
        """Take the square root of the elements."""
        return self.map(np.sqrt)

    def __add__(self, other):
        """Addition with another quantity."""
        return self._map_binary(other, add)

    def __sub__(self, other):
        """Substraction with another quantity."""
        return self._map_binary(other, sub)

    def __mul__(self, other):
        """Multiply with another quantity."""
        return self._map_binary(other, mul)

    def __truediv__(self, other):
        """Divide operation."""
        return self._map_binary(other, truediv)

    def __pow__(self, other, modulo=None):
        """Elementwise power"""
        if modulo is None:
            return self.map(lambda x: pow(x, other))
        else:
            return self.map(lambda x: pow(x, other, modulo))

    # Flattening and unflattening

    def struct(self):
        """Returns structure of the container"""
        return self.map(np.shape)

    def flatten(self):
        """Returns contents of the container reshaped to a vector"""
        return flatten(self)

    def update_from_vector(self, vec):
        """Updates the contents of the container from a vector"""
        struct = self.struct()
        offset = update_from_vector(self, vec, struct)
        if offset != len(vec):
            raise ValueError(
                'Vector length mismatch: expected {}, got {}'.format(
                    offset, len(vec)))

        
def from_vector(vec, struct):
    """Rebuilds a container from a vector and a container with shapes"""
    res = Tensors()
    offset = 0
    for k in sorted(struct.keys()):
        if not isinstance(struct[k], Tensors):
            size = np.prod(struct[k])
            res[k] = np.reshape(vec[offset:offset + size], struct[k])
            offset = offset + size
        else:
            res[k], offset_inside = from_vector(vec[offset:], struct[k])
            offset = offset + offset_inside
    return res, offset


def from_vec(vec, struct):
    """Rebuilds a container from a vector and a container with shapes"""
    res, offset = from_vector(vec, struct)
    if offset != len(vec):
        raise ValueError(
            'Vector length mismatch: expected {}, got {}'.format(
                offset, len(vec)))
    return res


def update_from_vector(old, vec, struct):
    """Updates container contents from a vector and a container with shapes"""
    offset = 0
    for k in sorted(struct.keys()):
        err_msg = 'Unmatching Tensors key {}'.format(k)
        if k not in old:
            raise ValueError(err_msg)
        if not isinstance(struct[k], Tensors):
            if not isinstance(old[k], Tensors):
                size = np.prod(struct[k])
                old[k] = np.reshape(vec[offset:offset + size], struct[k])
                offset = offset + size
            else:
                raise ValueError(err_msg)
        else:
            if isinstance(old[k], Tensors):
                offset_inside = update_from_vector(old[k], vec[offset:], struct[k])
                offset = offset + offset_inside
            else:
                raise ValueError(err_msg)
    return offset
    

def flatten(tensors):
    """Returns a copy of a flattened container with all contents flattened"""
    flattened = np.hstack((
        np.reshape(tensors[k], -1)
        if not isinstance(tensors[k], Tensors)
        else flatten(tensors[k])
        for k in sorted(tensors.keys())
    ))
    return flattened


def from_dict(x, factory=Tensors):
    """ Recursively transforms a dictionary into Tensors
    """
    if isinstance(x, dict):
        return factory((k, from_dict(v, factory)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(from_dict(v, factory) for v in x)
    else:
        return x


def to_dict(x):
    """ Recursively converts Tensors into a dictionary.
    """
    if isinstance(x, dict):
        return dict((k, to_dict(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(to_dict(v) for v in x)
    else:
        return x


if __name__ == '__main__':
    a = Tensors({'s': np.random.rand(2,2), 'd': Tensors({'a': np.ones((2,3)), 'b': np.zeros((2,2))})})
    k = a.flatten()
    print(k)
    print(k.shape)
    st = a.struct()
    print(st)
    print(a)
    b = from_vec(k, st)
    print(b)
    update_from_vector(a, k, st)
    print(a)

    
