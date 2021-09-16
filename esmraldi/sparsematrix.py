import operator
import traceback
import sys

import numpy as np

from sparse import COO, DOK

from collections.abc import Iterable, Iterator, Sized
from functools import reduce
from typing import Callable

def _find_start_end(mask):
    signed_mask = np.array(mask, dtype=int)
    signed_mask[signed_mask==False] = -1
    changes = ((np.roll(np.sign(signed_mask), 1) - np.sign(signed_mask)) != 0).astype(int)
    indices_changes = np.argwhere(changes==1).flatten()
    slices = []
    current_slice = []
    for i in indices_changes:
        if i == 0 and signed_mask[i]==-1:
            continue
        current_slice.append(i)
        if (len(current_slice) == 2):
            s = slice(current_slice[0], current_slice[1], 1)
            current_slice = []
            slices.append(s)
    if len(current_slice) == 1:
        s = slice(current_slice[0], len(mask), 1)
        slices.append(s)
    return tuple(slices)

def delete(sparse, indices, axis=0):
    N = sparse.shape[axis]
    mask = np.ones(N, dtype=bool)
    mask[indices] = False
    full_indices = np.arange(N)
    keep = full_indices[mask]
    slices = _find_start_end(mask)
    return np.concatenate([sparse[..., k] for k in slices], axis=-1)

def count_nonzero(arr):
    return arr.nnz


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
        self.__class__.__name__ = "coo"
        self.is_maybe_densify = True

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
        if self.is_maybe_densify:
            try:
                value = restricted_self.maybe_densify(max_size=1e7)
            except ValueError as ve:
                #If array
                value = SparseMatrix(restricted_self)
            except Exception as e:
                #If Number
                value = restricted_self
        else:
            value = SparseMatrix(restricted_self)
        return value

    def __setitem__(self, key, value):
        try:
            array = self.maybe_densify(max_size=1e7)
            array[key] = value
            newself = self.from_numpy(array)
        except ValueError as ve:
            array = self.asformat("dok")
            array[key] = value
            newself = SparseMatrix(array.asformat("coo"))
        self.data = newself.data
        self.coords = newself.coords

    def broadcast_to(self, shape):
        return SparseMatrix(super().broadcast_to(shape))

    def __array_function__(self, func, types, args, kwargs):
        try:
            sparse_func = getattr(sys.modules[__name__], func.__name__)
        except:
            pass
        else:
            return sparse_func(*args, **kwargs)


        array_func = super().__array_function__(func, types, args, kwargs)
        try:
            return SparseMatrix(array_func)
        except Exception as ve:
            return array_func

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        array_ufunc = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        try:
            return SparseMatrix(array_ufunc)
        except Exception as ve:
            return array_ufunc


    def view(self):
        return self

    def transpose(self, axes=None):
        cooT = super().transpose(axes)
        return SparseMatrix(cooT)

    def flatten(self, order="C"):
        cooF = super().flatten(order)
        return SparseMatrix(cooF)

    def reshape(self, shape, order="C"):
        if order == "F":
            if isinstance(shape, Iterable):
                shape = tuple(shape)
            else:
                shape = (shape,)

            if self.shape == shape:
                return self
            if any(d == -1 for d in shape):
                extra = int(self.size / np.prod([d for d in shape if d != -1]))
                shape = tuple([d if d != -1 else extra for d in shape])

            if self.size != reduce(operator.mul, shape, 1):
                raise ValueError(
                    "cannot reshape array of size {} into shape {}".format(self.size, shape)
                )

            if self._cache is not None:
                for sh, value in self._cache["reshape"]:
                    if sh == shape:
                        return value

            # TODO: this self.size enforces a 2**64 limit to array size
            linear_loc = np.ravel_multi_index(self.coords, self.shape, order="F")

            idx_dtype = self.coords.dtype
            coords = np.empty((len(shape), self.nnz), dtype=idx_dtype)
            strides = 1
            for i, d in enumerate(shape):
                coords[i, :] = (linear_loc // strides) % d
                strides *= d

            cooR = COO(
                coords,
                self.data,
                shape,
                has_duplicates=False,
                sorted=True,
                cache=self._cache is not None,
                fill_value=self.fill_value,
            )
        else:
            cooR = super().reshape(shape, order)
        return SparseMatrix(cooR)
