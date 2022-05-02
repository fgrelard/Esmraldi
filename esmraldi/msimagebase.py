import numpy as np
import esmraldi.spectraprocessing as sp

class MSImageBase:
    def __init__(self, spectra, mzs=None, tolerance=0, spectral_axis=-1, mean_spectra=None, peaks=None):
        if mzs is None:
            all_mzs = spectra[:, 0, ...]
            if len(spectra.shape) == 3:
                mzs = all_mzs[np.nonzero(all_mzs)]
            else:
                mzs = np.hstack(all_mzs).flatten()
            self.mzs = np.unique(mzs)
        else:
            self.mzs = mzs
        self.spectra = spectra
        self.tolerance = tolerance
        self.spectral_axis = spectral_axis
        self._mean_spectra = mean_spectra
        self._peaks = peaks
        self.is_maybe_densify = True
        self._normalization_image = None
        self._tic = None

    @property
    def normalization_image(self):
        return self._normalization_image

    @normalization_image.setter
    def normalization_image(self, value):
        self._normalization_image = value

    @property
    def tic(self):
        if self._tic is None:
            self._tic = sp.tic_values(self.spectra)
        return self._tic

    @tic.setter
    def tic(self, value):
        self._tic = value

    @property
    def mean_spectra(self):
        if self._mean_spectra is None:
            self._mean_spectra = self.compute_mean_spectra()
        return self._mean_spectra


    @mean_spectra.setter
    def mean_spectra(self, value):
        self._mean_spectra = value

    @property
    def peaks(self):
        return self._peaks

    @peaks.setter
    def peaks(self, peaks):
        self._peaks = peaks


    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    def compute_mean_spectra(self, spectra=None, norm_img=None):
        if spectra is None:
            spectra = self.spectra
        if norm_img is None:
            norm_img = self.normalization_image

        if norm_img is not None:
            norm_factor = norm_img.flatten()
            num = spectra[:, 1, :]
            denom = norm_factor[:, np.newaxis]
            np.divide(num, denom, out=num, where=denom!=0)
        if len(spectra.shape) >= 3:
            mean_spectra = sp.spectra_mean(spectra)
        else:
            mean_spectra = sp.spectra_mean_centroided(spectra)
        return mean_spectra

    def max(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def min(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def get_ion_image_index(self, index):
        raise NotImplementedError

    def get_ion_image_mzs(self, current_mz):
        raise NotImplementedError

    def astype(self, new_type, casting="unsafe", copy=True):
        raise NotImplementedError

    def reshape(self, shape, order="C"):
        raise NotImplementedError

    def transpose(self, axes=None):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def view(self, dtype=np.float64):
        raise NotImplementedError
