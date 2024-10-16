import bisect
import collections
import sys

import numbers

import numpy as np
import esmraldi.spectraprocessing as sp
import esmraldi.utils as utils
from esmraldi.msimagebase import MSImageBase

class MSImageImplementation(MSImageBase):

    def __init__(self, spectra, image=None, mzs=None, tolerance=14, is_maybe_densify=True, spectral_axis=-1, mean_spectra=None, peaks=None, is_ppm=True):

        print("MS image impl begin")
        if image is None and isinstance(spectra, MSImageImplementation):
            image = spectra.copy()
            super().__init__(image.spectra, image.mzs, image.tolerance, image.spectral_axis, image.mean_spectra, image.peaks, None, image.is_ppm)
        else:
            super().__init__(spectra, mzs, tolerance, spectral_axis, mean_spectra, peaks, None, is_ppm)

        print("MS image impl end", image.shape)
        if image.shape:
            self.image = image
        else:
            raise AttributeError("Please a provide a valid image")


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

    def __lt__(self, other):
        return self.image < other

    def __gt__(self, other):
        return self.image > other

    def __le__(self, other):
        return self.image <= other

    def __ge__(self, other):
        return self.image >= other

    def __eq__(self, other):
        return self.image == other

    def __ne__(self, other):
        return self.image != other

    def max(self, axis=None, out=None, keepdims=False):
        return self.image.max(axis, out, keepdims)

    def min(self, axis=None, out=None, keepdims=False):
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
            return MSImageImplementation(self.spectra, array_func, self.mzs, self.tolerance, is_maybe_densify=self.is_maybe_densify, spectral_axis=self.spectral_axis, mean_spectra=self.mean_spectra, peaks=self.peaks)
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
        array_ufunc = self.image.__array_ufunc__(ufunc, method, *L, **kwargs)
        if isinstance(array_ufunc, self.__class__):
            return MSImageImplementation(array_ufunc)
        try:
            return MSImageImplementation(self.spectra, array_ufunc, self.mzs, self.tolerance, is_maybe_densify=self.is_maybe_densify, spectral_axis=self.spectral_axis, mean_spectra=self.mean_spectra, peaks=self.peaks)
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
                try:
                    value = self.average_image(key)
                except Exception as e:
                    pass
        if is_still_image and not self.is_maybe_densify:
            spectra = self.spectra[key]
            return MSImageImplementation(spectra, value, None, self.tolerance, is_maybe_densify=self.is_maybe_densify, spectral_axis=self.spectral_axis)
        return value


    def get_ion_image_index(self, index):
        current_mz = self.mzs[index]
        return self.get_ion_image_mzs(current_mz)

    def get_ion_image_mzs(self, current_mz, tl=0, tr=0):
        tol = utils.tolerance(current_mz, self.tolerance, self.is_ppm)
        min_mz = current_mz - tol
        max_mz = current_mz + tol
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
        image_indices = tuple([slice(None) if i != self.spectral_axis else indices.flatten() for i in range(self.image.ndim)])
        average_image = np.mean(self.image[image_indices], axis=self.spectral_axis)
        try:
            average_image = average_image.todense()
        except:
            pass
        return average_image


    def astype(self, new_type, casting="unsafe",copy=True):
        ast = self.image.astype(new_type, casting=casting, copy=copy)
        return MSImageImplementation(self.spectra, ast, self.mzs, self.tolerance, self.is_maybe_densify, self.spectral_axis, self.mean_spectra, peaks=self.peaks)


    def transpose(self, axes=None):
        tr = self.image.transpose(axes)
        return MSImageImplementation(self.spectra, tr, self.mzs, self.tolerance, self.is_maybe_densify, self.spectral_axis, self.mean_spectra, peaks=self.peaks)


    def reshape(self, shape, order="C"):
        res = self.image.reshape(shape, order)
        return MSImageImplementation(self.spectra, res, self.mzs, self.tolerance, self.is_maybe_densify, self.spectral_axis, self.mean_spectra, peaks=self.peaks)


    def copy(self):
        copy = self.image.copy()
        return MSImageImplementation(self.spectra, copy, self.mzs, self.tolerance, self.is_maybe_densify, self.spectral_axis, self.mean_spectra, peaks=self.peaks)

    def view(self, dtype=np.float64):
        copy = self.copy()
        copy.is_maybe_densify = True
        return copy
