import pyimzml.ImzMLWriter as imzmlwriter
import pyimzml.ImzMLParser as imzmlparser
import numpy as np
import nibabel as nib
import os

def open_imzml(filename):
    return imzmlparser.ImzMLParser(filename)

def write_imzml(mzs, intensities, coordinates, filename):
    with imzmlwriter.ImzMLWriter(filename) as writer:
        for i in range(len(coordinates)):
            writer.addSpectrum(mzs[i], intensities[i], coordinates[i])

def get_spectra(imzml):
    spectra = []
    for i, (x, y, z) in enumerate(imzml.coordinates):
        mz, ints = imzml.getspectrum(i)
        spectra.append([mz, ints])
    return np.asarray(spectra)

def get_spectra_from_images(images):
    shape = images.shape
    coordinates = []
    intensities = []
    index_max = shape[:-1] if len(shape)==4 else shape[:-1] + (1,)
    intensities.append([0] * shape[-1])
    coordinates.append(index_max)
    for index in np.ndindex(shape[:-1]):
        xy_index = index + (slice(None),)
        I = images[xy_index]
        if I.any():
            index_3D = index if len(index) == 3 else index + (0, )
            add_tuple = (1, 1, 1)
            imzml_index = tuple(map(sum, zip(index_3D, add_tuple)))
            intensities.append(I)
            coordinates.append(imzml_index)
    return intensities, coordinates


def to_image_array(image):
    x, y = image.getspectrum(0)
    image_list = []
    for mz in x:
        im = imzmlparser.getionimage(image, mz, tol=0.1)
        image_list.append(im)
    img_array = np.transpose(np.asarray(image_list))
    return img_array

def get_all_array_images(image):
    mz, ints = image.getspectrum(0)
    im = np.zeros((image.imzmldict["max count of pixels x"], image.imzmldict["max count of pixels y"], len(mz)))
    for i, (x, y, z_) in enumerate(image.coordinates):
        mzs, ints = map(lambda x: np.asarray(x), image.getspectrum(i))
        for i in range(len(mzs)):
            min_i, max_i = imzmlparser._bisect_spectrum(mzs, mzs[i], tol=0.1)
            im[x - 1, y - 1, i] = sum(ints[min_i:max_i+1])
    return im

def to_nifti(image, filename):
    nibimg = nib.Nifti1Image(image, np.eye(4))
    nibimg.to_filename(filename)
