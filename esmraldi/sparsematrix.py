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
        print(type(self))
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
        print("coucou")
        array = self.todense()
        return array[key]

    def __setitem__(self, key, value):
        array = self.todense()
        print(key, type(array), type(self.data), type(self.todense()))
        array[key] = value
        self.data = array

    def __delitem__(self, key):
        array = self.todense().data
        del array[key]
        self.from_numpy(array)
