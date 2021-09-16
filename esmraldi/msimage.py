import bisect
import numpy as np
import esmraldi.imzmlio as io
import collections
import traceback

from esmraldi.sparsematrix import SparseMatrix
from abc import ABC, abstractmethod


class MSImage:
    def __new__(cls, spectra, image=None, mzs=None, coordinates=None, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        if image is None:
            if coordinates is None:
                raise ValueError("Coordinates not supplied and image missing.")
            max_x = max(coordinates, key=lambda item:item[0])[0]
            max_y = max(coordinates, key=lambda item:item[1])[1]
            max_z = max(coordinates, key=lambda item:item[2])[2]
            image = io.get_images_from_spectra(spectra, (max_x, max_y, max_z))
            tuple_transpose = [i-1 for i in range(len(image.shape)-1, 0, -1)] + [len(image.shape)-1]
            image = image.transpose(tuple_transpose)
        try:
            return MSImageNPY(spectra, image, mzs, tolerance, dtype, buffer, offset, strides, order)
        except Exception as e:
            return MSImageSparse(spectra, image, mzs, tolerance, dtype, buffer, offset, strides, order)


class MSImageInterface(ABC):
    def __init__(self, spectra, image=None, mzs=None, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        # cond2 = (spectra.shape[-1] <= 1 and len(spectra.shape) == len(image.shape)+1)
        # cond3 = (image.shape[-1] <= 1 and len(image.shape) == len(spectra.shape)+1)
        # print(image.shape, spectra.shape)
        # assert(image.shape[-1] == spectra.shape[-1] or cond2 or cond3)
        self.init_attributes(spectra, mzs, tolerance)

    def init_attributes(obj, spectra, mzs, tolerance):
        if mzs is None:
            all_mzs = spectra[:, 0, ...]
            obj.mzs = np.unique(all_mzs[np.nonzero(all_mzs)])
        else:
            obj.mzs = mzs
        obj.spectra = spectra
        obj.tolerance = tolerance
        obj.is_still_image = True

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
        average_image = np.mean(self[..., indices.flatten()], axis=-1)
        return average_image


    def __getitem__(self, key):
        is_array = isinstance(key, tuple) and any([isinstance(elem, (collections.abc.Iterable, slice)) for elem in key])
        self.is_still_image = isinstance(key, tuple) and all([(k==Ellipsis or k==slice(None)) for i, k in enumerate(key) if i < len(key)-1])
        if self.tolerance > 0 and not is_array and self.is_still_image:
            return self.get_ion_image_index(key)
        else:
            return super().__getitem__(key)



class MSImageNPY(MSImageInterface, np.ndarray):
    def __new__(cls, spectra, image, mzs=None, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        obj = np.asarray(image).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.spectra = getattr(obj, "spectra", None)
        self.mzs = getattr(obj, "mzs", None)
        self.tolerance = getattr(obj, "tolerance", None)

    def __array_function__(self, func, types, args, kwargs):
        array_func = super().__array_function__(func, types, args, kwargs)
        try:
            return MSImageNPY(self.spectra, array_func, self.mzs, self.tolerance, self.dtype)
        except Exception as ve:
            return array_func

    def __getitem__(self, key):
        array = super().__getitem__(key)
        if self.is_still_image:
            spectra = self.spectra[key]
            return MSImageNPY(spectra, array, None, self.tolerance, self.dtype)
        return array

    def copy(self):
        copy = super().copy()
        copy.init_attributes(self.spectra, self.mzs, self.tolerance)
        return copy


class MSImageSparse(MSImageInterface, SparseMatrix):
    def __init__(self, spectra, image, mzs=None, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        super(SparseMatrix, self).__init__(image)
        super().__init__(spectra, image, mzs, tolerance, dtype, buffer, offset, strides, order)


    def __array_function__(self, func, types, args, kwargs):
        array_func = super().__array_function__(func, types, args, kwargs)
        try:
            return MSImageSparse(self.spectra, array_func, self.mzs, self.tolerance, self.dtype)
        except Exception as ve:
            return array_func


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        array_ufunc = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        try:
            return MSImageSparse(self.spectra, array_ufunc, self.mzs, self.tolerance, self.dtype)
        except Exception as ve:
            return array_ufunc

    def __getitem__(self, key):
        array = super().__getitem__(key)
        if self.is_still_image:
            spectra = self.spectra[key]
            return MSImageSparse(spectra, array, None, self.tolerance, self.dtype)
        return array

    def copy(self):
        copy = super().copy()
        copy.init_attributes(self.spectra, self.mzs, self.tolerance)
        return copy
