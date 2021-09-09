import bisect
import numpy as np
import esmraldi.imzmlio as io
import collections

from esmraldi.sparsematrix import SparseMatrix
from abc import ABC, abstractmethod


class MSImage:
    def __new__(cls, spectra, image=None, coordinates=None, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        if image is None:
            if coordinates is None:
                raise ValueError("Coordinates not supplied and image missing.")
            max_x = max(coordinates, key=lambda item:item[0])[0]
            max_y = max(coordinates, key=lambda item:item[1])[1]
            max_z = max(coordinates, key=lambda item:item[2])[2]
            image = io.get_images_from_spectra(spectra, (max_x, max_y, max_z))
            tuple_transpose = [i-1 for i in range(len(image.shape)-1, 0, -1)] + [len(image.shape)-1]
            print(tuple_transpose)
            image = image.transpose(tuple_transpose)
        try:
            return MSImageNPY(spectra, image, tolerance, dtype, buffer, offset, strides, order)
        except Exception as e:
            return MSImageSparse(spectra, image, tolerance, dtype, buffer, offset, strides, order)


class MSImageInterface(ABC):
    @abstractmethod
    def __init__(self, spectra, image=None, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        pass

    def init_attributes(obj, spectra, image, tolerance):
        all_mzs = spectra[:, 0, ...]
        obj.mzs = np.unique(all_mzs[np.nonzero(all_mzs)])
        obj.spectra = spectra
        obj.tolerance = tolerance

    def get_ion_image_index(self, index):
        print("index", index)
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
        is_array = isinstance(key, collections.abc.Iterable) and any([isinstance(elem, collections.abc.Iterable) for elem in key])
        if self.tolerance > 0 and not is_array:
            return self.get_ion_image_index(key)
        else:
            return super().__getitem__(key)


class MSImageNPY(MSImageInterface, np.ndarray):
    def __new__(cls, spectra, image, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        assert(image.shape[-1] == spectra.shape[-1])
        obj = np.asarray(image).view(cls)
        return obj

    def __init__(self, spectra, image, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        self.init_attributes(spectra, image, tolerance)


    def __array_finalize__(self, obj):
        if obj is None: return
        self.mzs = getattr(obj, "mzs", None)
        self.tolerance = getattr(obj, "tolerance", None)


class MSImageSparse(MSImageInterface, SparseMatrix):
    def __init__(self, spectra, image, tolerance=0, dtype=float, buffer=None, offset=0, strides=None, order=None):
        assert(image.shape[-1] == spectra.shape[-1])
        super(SparseMatrix, self).__init__(image)
        self.init_attributes(spectra, image, tolerance)
