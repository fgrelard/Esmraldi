import numpy as np
import esmraldi.imzmlio as io

from esmraldi.msimageimpl import MSImageImplementation

class MSImage:
    def __new__(cls, spectra, image=None, mzs=None, shape=None, coordinates=None, tolerance=14, mean_spectra=None, dtype=float, buffer=None, offset=0, strides=None, order=None):
        if image is None:
            if shape is None:
                shape = cls.determine_shape(cls, coordinates)
            image = io.get_images_from_spectra(spectra, shape)
            tuple_transpose = [i-1 for i in range(len(image.shape)-1, 0, -1)] + [len(image.shape)-1]
            # image = image.transpose(tuple_transpose)
        return MSImageImplementation(spectra, image, mzs, tolerance, mean_spectra=mean_spectra)

    def determine_shape(cls, coordinates):
        if coordinates is None:
            raise ValueError("Shape not supplied and image missing.")
        max_x = max(coordinates, key=lambda item:item[0])[0]
        max_y = max(coordinates, key=lambda item:item[1])[1]
        max_z = max(coordinates, key=lambda item:item[2])[2]
        return (max_x, max_y, max_z)

def concatenate(arrays, axis=0):
    _self = arrays[0]
    concat = np.concatenate([a.image for a in arrays], axis=axis)
    spectra = np.concatenate([a.spectra for a in arrays], axis=axis)
    is_maybe_densify = _self.is_maybe_densify
    return MSImageImplementation(spectra, concat, None, _self.tolerance, is_maybe_densify=is_maybe_densify)
