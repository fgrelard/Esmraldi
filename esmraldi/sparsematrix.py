from sparse import COO
import numpy as np

class SparseMatrix(COO):

    def __init__(self, coords,
        data=None,
        shape=None,
        has_duplicates=True,
        sorted=False,
        prune=False,
        cache=False,
        fill_value=None,
        idx_dtype=None):
        super().__init__(coords, data, shape, has_duplicates, sorted, prune, cache, fill_value, idx_dtype)

    def __add__(self, other):
        if np.isscalar(other) or self.data.shape != other.data.shape:
            return self.todense() + other
        return self.todense() + other.todense()

    def __sub__(self, other):
        if np.isscalar(other) or self.data.shape != other.data.shape:
            return self.todense() - other
        return self.todense() - other.todense()

    def __mul__(self, other):
        if np.isscalar(other) or self.data.shape != other.data.shape:
            return self.todense() * other
        return self.todense() * other.todense()

    def __div__(self, other):
        if np.isscalar(other) or self.data.shape != other.data.shape:
            return self.todense() / other
        return self.todense() / other.todense()

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        if np.isscalar(other) or self.data.shape != other.data.shape:
            return self.todense() // other
        return self.todense() // other.todense()

    def __mod__(self, other):
        if np.isscalar(other) or self.data.shape != other.data.shape:
            return self.todense() % other
        return self.todense() % other.todense()

    def __pow__(self, other):
        if np.isscalar(other) or self.data.shape != other.data.shape:
            return self.todense() ** other
        return self.todense() ** other.todense()


    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other

    def __itruediv__(self, other):
        return self / other

    def __ifloordiv__(self, other):
        return self // other

    def __imod__(self, other):
        return self % other

    def __ipow__(self, other):
        return self ** other

    def __getitem__(self, key):
        restricted_self = COO.__getitem__(self, key)
        try:
            value = restricted_self.maybe_densify(max_size=1e7)
        except ValueError as ve:
            value = SparseMatrix(restricted_self)
            # value = COO.__getitem__(array, key)
        return value

    def __setitem__(self, key, value):
        try:
            array = self.maybe_densify(max_size=1e7)
            array[key] = value
            newself = self.from_numpy(array)
            self.data = newself.data
            self.coords = newself.coords
        except ValueError as ve:
            array = self
            COO.__setitem__(array, key, value)

    def __delitem__(self, key):
        try:
            array = self.maybe_densify(max_size=1e7)
            del array[key]
            self.from_numpy(array)
        except ValueError as ve:
            array = self
            COO.__delitem__(array, key)

    def transpose(self, axes=None):
        cooT = super().transpose(axes)
        return SparseMatrix(cooT)

    def flatten(self, order="C"):
        cooF = super().flatten(order)
        return SparseMatrix(cooF)

    def reshape(self, shape, order="C"):
        cooR = super().reshape(shape, order)
        return SparseMatrix(cooR)
