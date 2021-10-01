import bisect
import numpy as np
import collections
import sys

import functools
import inspect
import numbers

import esmraldi.spectraprocessing as sp
import esmraldi.imzmlio as io

from esmraldi.sparsematrix import SparseMatrix
from abc import ABC, abstractmethod


class MSImage:
    def __new__(cls, spectra, image=None, mzs=None, coordinates=None, tolerance=0, mean_spectra=None, dtype=float, buffer=None, offset=0, strides=None, order=None):
        if image is None:
            if coordinates is None:
                raise ValueError("Coordinates not supplied and image missing.")
            max_x = max(coordinates, key=lambda item:item[0])[0]
            max_y = max(coordinates, key=lambda item:item[1])[1]
            max_z = max(coordinates, key=lambda item:item[2])[2]
            image = io.get_images_from_spectra(spectra, (max_x, max_y, max_z))
            tuple_transpose = [i-1 for i in range(len(image.shape)-1, 0, -1)] + [len(image.shape)-1]
            # image = image.transpose(tuple_transpose)
        return MSImageImplementation(spectra, image, mzs, tolerance, mean_spectra=mean_spectra)


def concatenate(arrays, axis=0):
    _self = arrays[0]
    concat = np.concatenate([a.image for a in arrays], axis=axis)
    spectra = np.concatenate([a.spectra for a in arrays], axis=axis)
    is_maybe_densify = _self.is_maybe_densify
    return MSImageImplementation(spectra, concat, None, _self.tolerance, is_maybe_densify=is_maybe_densify)

class MSImageImplementation:

    def __init__(self, spectra, image=None, mzs=None, tolerance=0, is_maybe_densify=False, spectral_axis=-1, mean_spectra=None):

        self._mean_spectra = None

        if image is None and isinstance(spectra, MSImageImplementation):
            image = spectra.copy()
            self.init_attributes(image.spectra, image.mzs, image.tolerance, image.spectral_axis, image.mean_spectra)
        else:
            self.init_attributes(spectra, mzs, tolerance, spectral_axis, mean_spectra)

        if image.shape:
            self.image = image
        else:
            raise AttributeError("Please a provide a valid image")

    def init_attributes(obj, spectra, mzs, tolerance, spectral_axis, mean_spectra):
        if mzs is None:
            all_mzs = spectra[:, 0, ...]
            obj.mzs = np.unique(all_mzs[np.nonzero(all_mzs)])
        else:
            obj.mzs = mzs
        obj.spectra = spectra
        obj.tolerance = tolerance
        obj.spectral_axis = spectral_axis
        obj._mean_spectra = mean_spectra


    @property
    def dtype(self):
        return self.image.dtype

    @property
    def nnz(self):
        return self.image.nnz

    @property
    def data(self):
        return self.image.data

    @property
    def coords(self):
        return self.image.coords

    @property
    def shape(self):
        return self.image.shape

    @property
    def ndim(self):
        return self.image.ndim

    @property
    def size(self):
        return self.image.size

    @property
    def is_maybe_densify(self):
        try:
            return self.image.is_maybe_densify
        except:
            return True

    @property
    def mean_spectra(self):
        if self._mean_spectra is None:
            self._mean_spectra = sp.spectra_mean(self.spectra)
        return self._mean_spectra

    @mean_spectra.setter
    def mean_spectra(self, value):
        self._mean_spectra = value

    @is_maybe_densify.setter
    def is_maybe_densify(self, value):
        try:
            self.image.is_maybe_densify = value
        except:
            pass
        try:
            self.spectra.is_maybe_densify = value
        except:
            pass

    def compute_mean_spectra(self):
        self.mean_spectra = sp.spectra_mean(self.spectra)

    def max(self, axis, out, keepdims):
        return self.image.max(axis, out, keepdims)

    def min(self):
        return self.image.min(axis, out, keepdims)

    def __array_function__(self, func, types, args, kwargs):
        try:
            sparse_func = getattr(sys.modules[__name__], func.__name__)
        except:
            pass
        else:
            return sparse_func(*args, **kwargs)

        L = [arg.image if isinstance(arg, MSImageImplementation) else arg for arg in args]
        t = [type(arg.image) if isinstance(arg, MSImageImplementation) else type(arg) for arg in args]
        array_func = self.image.__array_function__(func, t, tuple(L), kwargs)
        if isinstance(array_func, MSImageImplementation):
            return MSImageImplementation(array_func)
        try:
            return MSImageImplementation(self.spectra, array_func, self.mzs, self.tolerance, is_maybe_densify=self.is_maybe_densify, spectral_axis=self.spectral_axis, mean_spectra=self.mean_spectra)
        except Exception as ve:
            return array_func


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        try:
            sparse_func = getattr(sys.modules[__name__], ufunc.__name__)
        except:
            pass
        else:
            return sparse_func(*inputs, **kwargs)

        L = [arg.image if isinstance(arg, MSImageImplementation) else arg for arg in inputs]
        array_ufunc = self.image.__array_ufunc__(func, method, *L, **kwargs)
        if isinstance(array_ufunc, self.__class__):
            return MSImageImplementation(array_ufunc)
        try:
            return MSImageImplementation(self.spectra, array_ufunc, self.mzs, self.tolerance, is_maybe_densify=self.is_maybe_densify, spectral_axis=self.spectral_axis, mean_spectra=self.mean_spectra)
        except Exception as ve:
            return array_ufunc


    def __getitem__(self, key):
        is_array = isinstance(key, tuple) and any([isinstance(elem, (collections.abc.Iterable, slice)) for elem in key])
        is_still_image = isinstance(key, numbers.Number)
        if isinstance(key, tuple):
            L = [isinstance(k, numbers.Number) for k in key]
            i = iter(L)
            is_still_image = any(i) and not any(i)
        if self.tolerance > 0 and not is_array and is_still_image:
            value = self.get_ion_image_index(key)
        else:
            value =  self.image[key]
            if value.ndim == self.image.ndim:
                value = self.average_image(key)
        if is_still_image and not self.is_maybe_densify:
            spectra = self.spectra[key]
            return MSImageImplementation(spectra, value, None, self.tolerance, is_maybe_densify=self.is_maybe_densify, spectral_axis=self.spectral_axis)
        return value


    def get_ion_image_index(self, index):
        current_mz = self.mzs[index]
        return self.get_ion_image_mzs(current_mz)

    def get_ion_image_mzs(self, current_mz):
        min_mz = current_mz - self.tolerance
        max_mz = current_mz + self.tolerance
        mask = (self.mzs > min_mz) & (self.mzs < max_mz)
        no_intersection = not mask.any()
        if no_intersection:
            mask_index = min(bisect.bisect_left(self.mzs, current_mz), len(self.mzs)-1)
            if mask_index > 0 and \
               abs(self.mzs[mask_index-1]-current_mz) < \
               abs(self.mzs[mask_index]-current_mz):
                mask_index = mask_index-1
            if mask_index < len(self.mzs) - 1 and \
               abs(self.mzs[mask_index+1]-current_mz) < \
               abs(self.mzs[mask_index]-current_mz):
                mask_index = mask_index+1
            mask[mask_index] = True
        indices = np.argwhere(mask == True)
        return self.average_image(indices)

    def average_image(self, indices):
        average_image = np.mean(np.take(self.image, indices.flatten(), axis=self.spectral_axis), axis=self.spectral_axis)
        return average_image


    def astype(self, new_type, casting="unsafe",copy=True):
        ast = self.image.astype(new_type, casting=casting, copy=copy)
        return MSImageImplementation(self.spectra, ast, self.mzs, self.tolerance, self.is_maybe_densify, self.spectral_axis, self.mean_spectra)


    def transpose(self, axes=None):
        tr = self.image.transpose(axes)
        return MSImageImplementation(self.spectra, tr, self.mzs, self.tolerance, self.is_maybe_densify, self.spectral_axis, self.mean_spectra)


    def reshape(self, shape, order="C"):
        res = self.image.reshape(shape, order)
        return MSImageImplementation(self.spectra, res, self.mzs, self.tolerance, self.is_maybe_densify, self.spectral_axis, self.mean_spectra)


    def copy(self):
        copy = self.image.copy()
        return MSImageImplementation(self.spectra, copy, self.mzs, self.tolerance, self.is_maybe_densify, self.spectral_axis, self.mean_spectra)
