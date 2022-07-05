import operator
import traceback
import sys

import numpy as np

from sparse import COO, DOK
from sparse._slicing import normalize_index

from collections.abc import Iterable, Iterator, Sized
from functools import reduce
from typing import Callable

from itertools import product, repeat

def _find_start_end(mask):
    signed_mask = np.array(mask, dtype=int)
    signed_mask[signed_mask==False] = -1
    changes = ((np.roll(np.sign(signed_mask), 1) - np.sign(signed_mask)) != 0).astype(int)
    indices_changes = np.argwhere(changes==1).flatten()
    slices = []
    current_slice = []
    for i, index_change in enumerate(indices_changes):
        if index_change == 0 and signed_mask[index_change]==-1:
            continue
        if i == 0 and index_change != 0:
            current_slice.append(0)
        current_slice.append(index_change)
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
    L = []
    for k in slices:
        L.append(sparse[..., k])
    return np.concatenate(L, axis=-1)

def count_nonzero(arr):
    return arr.nnz

def zeros(shape, dtype=float, order='C', *, like=None):
    return SparseMatrix(coords=[], data=None, shape=shape)

def zeros_like(a, dtype=None, order='C', subok='True', shape=None):
    return zeros(a.shape, dtype, order)

def take(array, indices, axis=None, **kwargs):
    index = []
    for dim in range(array.ndim):
        if dim == axis or (axis==-1 and dim == array.ndim-1):
            index.append(indices)
        else:
            index.append(slice(None))
    return array[tuple(index)]


class SparseMatrix(COO):
    def __init__(self, coords,
                 data=None,
                 shape=None,
                 has_duplicates=True,
                 sorted=False,
                 prune=False,
                 cache=False,
                 fill_value=None,
                 idx_dtype=None,
                 is_maybe_densify=True):
        super().__init__(coords, data, shape, has_duplicates, sorted, prune, cache, fill_value, idx_dtype)
        self.__class__.__name__ = "coo"
        self.is_maybe_densify = is_maybe_densify

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


    def get_nd_iterable_indices(self, keys):
        try:
            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            idx = np.array(keys)
        except:
            return None
        if idx.ndim <= 2:
            return None

        x, y = keys
        final_shape = x.shape + (self.shape[-1],)
        final_array = SparseMatrix(coords=[], data=None, shape=final_shape)
        for ind in np.ndindex(x.shape[:-2]):
            curr_img = COO.__getitem__(self, (x[ind].flatten(), y[ind].flatten()))
            curr_img = curr_img.reshape((x.shape[-2], x.shape[-1], self.shape[-1]))
            final_array[ind] = curr_img.todense()
        return final_array

    def __getitem__(self, key):
        restricted_self = self.get_nd_iterable_indices(key)
        if restricted_self is None:
            restricted_self = COO.__getitem__(self, key)
        if self.is_maybe_densify:
            try:
                value = restricted_self.maybe_densify(max_size=int(1e7), min_density=1)
            except ValueError as ve:
                #If array
                value = SparseMatrix(restricted_self)
            except Exception as e:
                #If Number
                value = restricted_self
        else:
            try:
                value = SparseMatrix(restricted_self)
            except Exception as e:
                value = restricted_self
        return value

    def __setitem__(self, key, value):
        try:
            array = self.maybe_densify(max_size=1e7)
            array[key] = value
            newself = COO.from_numpy(array)
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
        try:
            sparse_func = getattr(sys.modules[__name__], ufunc.__name__)
        except:
            pass
        else:
            return sparse_func(*inputs, **kwargs)

        array_ufunc = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        try:
            return SparseMatrix(array_ufunc)
        except Exception as ve:
            return array_ufunc


    def view(self, dtype=None, type=None):
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
