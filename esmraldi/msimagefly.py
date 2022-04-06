import collections
import numbers
import numpy as np
from bisect import bisect_left, bisect_right

import esmraldi.spectraprocessing as sp
from esmraldi.msimagebase import MSImageBase

class MSImageOnTheFly(MSImageBase):
    def __init__(self, spectra, coords=None, mzs=None, tolerance=0, spectral_axis=-1, mean_spectra=None, peaks=None):
        super().__init__(spectra, mzs, tolerance, spectral_axis, mean_spectra, peaks)

        self.coords = coords
        max_x = max(self.coords, key=lambda item:item[0])[0]
        max_y = max(self.coords, key=lambda item:item[1])[1]
        max_z = max(self.coords, key=lambda item:item[2])[2]
        coords = (max_x, max_y, max_z)

        if coords[-1] == 1:
            coords = coords[:-1]

        self.shape = coords + (len(self.mzs), )

        self.image = np.zeros(self.shape[:-1])

    @property
    def dtype(self):
        return self.spectra.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)


    def max(self, axis=None, out=None):
        return np.hstack(self.spectra[:, 1]).flatten().max()

    def min(self, axis=None, out=None):
        return np.hstack(self.spectra[:, 1]).flatten().min()

    def bisect_spectrum(self, mzs, mz_value, tol_left, tol_right):
        ix_l, ix_u = bisect_left(mzs, mz_value - tol_left), bisect_right(mzs, mz_value + tol_right) - 1
        if ix_l == len(mzs):
            return len(mzs), len(mzs)
        if ix_u < 1:
            return 0, 0
        if ix_u == len(mzs):
            ix_u -= 1
        if mzs[ix_l] < (mz_value - tol_left):
            ix_l += 1
        if mzs[ix_u] > (mz_value + tol_right):
            ix_u -= 1
        return ix_l, ix_u

    def __getitem__(self, key):
        iterfunc = lambda x: isinstance(x, (collections.abc.Iterable, slice))
        is_array = (isinstance(key, tuple) and any([iterfunc(elem) for elem in key])) or iterfunc(key)

        mz_value = self.mzs[key]
        tolerance_left, tolerance_right = self.tolerance, self.tolerance
        if is_array:
            tolerance_left = np.abs(np.median(mz_value) - np.amin(mz_value))
            tolerance_right = np.abs(np.median(mz_value) - np.amax(mz_value))
        im = self.get_ion_image_mzs(mz_value, tolerance_left, tolerance_right)

        return im


    def get_ion_image_index(self, index):
        current_mz = self.mzs[index]
        return self.get_ion_image_mzs(current_mz)

    def get_ion_image_mzs(self, mz_value, tl=0, tr=0):
        im = np.zeros(self.shape[self.spectral_axis+1:])
        for i, (x, y, z_) in enumerate(self.coords):
            mzs, ints = self.spectra[i, 0], self.spectra[i, 1]
            min_i, max_i = self.bisect_spectrum(mzs, np.median(mz_value), tl, tr)
            im[y-1, x-1] = sum(ints[min_i:max_i+1])

        return im

    def astype(self, new_type, casting="unsafe", copy=True):
        return self

    def reshape(self, shape, order="C"):
        self.shape = shape
        return self

    def transpose(self, axes=None):
        self.shape = tuple(np.array(self.shape)[list(axes)])
        return self

    def copy(self):
        return self

    def view(self, dtype=np.float64):
        return self
