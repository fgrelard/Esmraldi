import collections
import numbers
import numpy as np
from bisect import bisect_left, bisect_right

import esmraldi.spectraprocessing as sp
import esmraldi.utils as utils
from esmraldi.msimagebase import MSImageBase

class MSImageOnTheFly(MSImageBase):
    def __init__(self, spectra, coords=None, mzs=None, tolerance=14, spectral_axis=-1, mean_spectra=None, peaks=None, indexing=None, is_ppm=True):
        super().__init__(spectra, mzs, tolerance, spectral_axis, mean_spectra, peaks, None, is_ppm)
        all_mzs = np.hstack(spectra[:, 0]).flatten()
        if indexing is None:
            self.indexing = utils.indices_search_sorted(all_mzs, self.mzs)
        else:
            self.indexing = indexing
        all_len = [len(g) for g in spectra[:, 0]]
        self.ind_len = np.hstack([np.arange(l) for l in all_len])
        self.cumlen = np.cumsum(all_len)-1
        self.coords = coords
        max_x = max(self.coords, key=lambda item: item[0])[0]
        max_y = max(self.coords, key=lambda item: item[1])[1]
        max_z = max(self.coords, key=lambda item: item[2])[2]
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
        intensities = np.hstack(self.spectra[:, 1]).flatten()
        return intensities.max()

    def min(self, axis=None, out=None):
        intensities = np.hstack(self.spectra[:, 1]).flatten()
        return intensities.min()

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
        if is_array:
            tolerance_left = np.abs(np.median(mz_value) - np.amin(mz_value))
            tolerance_right = np.abs(np.median(mz_value) - np.amax(mz_value))
        else:
            tolerance_left = utils.tolerance(mz_value, self.tolerance, is_ppm=self.is_ppm)
            tolerance_right = tolerance_left
        im = self.get_ion_image_mzs(mz_value, tolerance_left, tolerance_right)

        return im


    def get_ion_image_index(self, index):
        current_mz = self.mzs[index]
        return self.get_ion_image_mzs(current_mz)

    def get_ion_image_mzs(self, mz_value, tl=0, tr=0):
        min_i, max_i = utils.indices_search_sorted([mz_value-tl, mz_value+tr], self.mzs)
        ind = np.where((self.indexing >= min_i) & (self.indexing < max_i))[0]
        ind_value = self.ind_len[ind]
        ind_spectra = np.searchsorted(self.cumlen, ind)
        ind_value = np.split(ind_value, np.where(np.diff(ind_spectra) != 0)[0]+1)
        ind_spectra = np.unique(ind_spectra)
        mask = np.ones(len(self.shape),dtype=bool)
        mask[self.spectral_axis] = False
        shape = np.array(self.shape)[mask]

        im = np.zeros(shape)
        x, y = np.unravel_index(ind_spectra, shape, order="C")
        intensities = self.spectra[:, 1][ind_spectra]
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            curr_i = intensities[i]
            curr_i = curr_i[ind_value[i]]
            im[x_i, y_i] = sum(curr_i)

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
