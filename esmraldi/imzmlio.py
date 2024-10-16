"""
Module for the input/output
of imzML files
Conversion to numpy array
"""
import pyimzml.ImzMLWriter as imzmlwriter
import pyimzml.ImzMLParser as imzmlparser
# import esmraldi.imzmlparsermmapped as imzmlparser
import numpy as np
import nibabel as nib
import os
import cv2 as cv
import warnings
import bisect
import matplotlib.pyplot as plt
import h5py

from skimage import exposure, img_as_ubyte

from esmraldi.sparsematrix import SparseMatrix
from esmraldi.utils import progress, factors, attempt_reshape
import esmraldi.spectraprocessing as sp
import time
import math
import tifffile
import SimpleITK as sitk

MAX_MAGNITUDE_ORDER = 6
MAX_NUMBER = int(1e6)

def open_imzml(filename, only_metadata=False):
    """
    Opens an imzML file

    Parameters
    ----------
    filename: string
        input file

    Returns
    ----------
    imzmlparser.ImzMLParser
        parser of file

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ibd_file = None if only_metadata else imzmlparser.INFER_IBD_FROM_IMZML
        return imzmlparser.ImzMLParser(filename, ibd_file=ibd_file)




def write_imzml(mzs, intensities, coordinates, filename):
    """
    Writes a file to imzML

    Parameters
    ----------
    mzs: list
        list of list of m/z ratios
    intensities: list
        list of list of intensities
    coordinates: list
        pixel coordinates associated to each spectrum
    filename: string
        output file


    """
    with imzmlwriter.ImzMLWriter(filename) as writer:
        for i in range(len(coordinates)):
            writer.addSpectrum(mzs[i], intensities[i], coordinates[i])

def normalize(image):
    """
    Normalizes an image : 0 -- 255 scale

    Parameters
    ----------
    image: np.ndarray
        input image

    Returns
    ----------
    np.ndarray
        normalized image
    """
    def norm_img(image):
        min_value = image.min()
        max_value = np.percentile(image, 99)
        if min_value == max_value:
            max_value = image.max()
        if min_value == max_value:
            max_value = 1
            min_value = 0
        image_normalized = ((image - min_value) / (max_value - min_value))*255
        image_normalized = np.clip(image_normalized, 0, 255)
        return image_normalized

    image_normalized = np.zeros_like(image, dtype=np.uint8)
    if len(image.shape) <= 2:
        image_normalized = norm_img(image)
        # image_normalized = np.uint8(cv.normalize(image, None, 0, 255, cv.NORM_MINMAX))
    else:
        z = image.shape[-1]
        for k in range(z):
            slice2D = image[..., k]
            slice2DNorm = norm_img(slice2D)
            # slice2DNorm = np.uint8(cv.normalize(slice2D, None, 0, 255, cv.NORM_MINMAX))
            image_normalized[..., k] = slice2DNorm
    return image_normalized

def sparse_coordinates(spectra, imsize):
    mzs = spectra[:, 0]
    unique_mzs, indices_mzs = np.unique(np.hstack(mzs), return_inverse=True)
    number_points = len(unique_mzs)
    pixel_numbers = np.hstack([np.repeat(i, int(len(mzs[i]))) for i in range(len(mzs))]).astype(np.int32)
    shape = (imsize, 2, number_points)
    coordinates = np.zeros((2*len(pixel_numbers), 3), dtype=np.int32)
    for j in range(2):
        for i in range(len(pixel_numbers)):
            coord = (pixel_numbers[i], j, indices_mzs[i])
            coordinates[i+j*len(pixel_numbers)] = coord

    coordinates = coordinates.T
    return coordinates, shape

def get_full_spectra_sparse(spectra, imsize, sorted=False):
    coordinates, shape = sparse_coordinates(spectra, imsize)
    if spectra.ndim < 3:
        spectra = spectra.T.flatten()
    data = np.hstack(spectra).flatten().astype(np.float32)
    full_spectra_sparse = SparseMatrix(coordinates, data, shape, sorted=sorted, has_duplicates=False)
    return full_spectra_sparse

def get_full_spectra_dense(spectra, coordinates, shape):
    mzs, ints = spectra[0, ...]
    number_points = len(mzs)
    imsize = np.prod(shape)
    full_spectra = np.zeros((imsize, 2, number_points))
    full_spectra[:,0,:] = mzs
    for i, (x, y, z) in enumerate(coordinates):
        real_index = (x-1) + (y-1) * shape[0] + (z-1) * shape[0] * shape[1]
        mz, ints = spectra[i, ...]
        full_spectra[real_index, 0] = mz
        full_spectra[real_index, 1] = ints
    return full_spectra

def get_full_spectra(imzml, spectra=None):
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]

    if spectra is None:
        spectra = get_spectra(imzml)
    shape = (max_x, max_y, max_z)
    if len(spectra.shape) == 2:
        return get_full_spectra_sparse(spectra, np.prod(shape))

    return get_full_spectra_dense(spectra, imzml.coordinates, shape)

    return full_spectra


def get_spectra(imzml, pixel_numbers=[]):
    """
    Extracts spectra from imzML
    into numpy format

    Parameters
    ----------
    imzml: imzmlparser.ImzMLParser
        parser

    Returns
    ----------
    np.array
        collection of spectra
    """
    spectra = []
    coordinates = []
    for i in pixel_numbers:
        coordinates.append(imzml.coordinates[i])

    if len(pixel_numbers) == 0:
        coordinates = imzml.coordinates.copy()

    for i, (x, y, z) in enumerate(coordinates):
        mz, ints = imzml.getspectrum(i)
        spectra.append([mz, ints])
    if spectra and not all(len(l[0]) == len(spectra[0][0]) for l in spectra):
        return np.array(spectra, dtype=object)
    return np.array(spectra)


def get_spectra_intensities(imzml, pixel_numbers=[]):
    """
    Extracts spectra intensities from imzML
    into numpy format

    Parameters
    ----------
    imzml: imzmlparser.ImzMLParser
        parser

    Returns
    ----------
    np.array
        collection of spectra
    """
    spectra = np.zeros(shape=(len(imzml.coordinates), len(imzml.getspectrum(0)[1])), dtype="float32")
    for i, (x, y, z) in enumerate(imzml.coordinates):
        if (len(pixel_numbers) > 0 and i in pixel_numbers) or len(pixel_numbers) == 0:
            mz, ints = imzml.getspectrum(i)
            spectra[i] = ints.astype("float32")
    return spectra

def get_spectra_mzs(imzml, pixel_numbers=[]):
    """
    Extracts spectra mzs from imzML
    into numpy format

    Parameters
    ----------
    imzml: imzmlparser.ImzMLParser
        parser

    Returns
    ----------
    np.array
        collection of spectra
    """
    spectra = np.zeros(shape=(len(imzml.coordinates), len(imzml.getspectrum(0)[0])), dtype="float32")
    for i, (x, y, z) in enumerate(imzml.coordinates):
        if (len(pixel_numbers) > 0 and i in pixel_numbers) or len(pixel_numbers) == 0:
            mz, ints = imzml.getspectrum(i)
            spectra[i] = mz.astype("float32")
    return spectra


def get_spectra_from_images(images, full=False):
    """
    Extracts spectra intensities and coordinates
    from numpy array

    Parameters
    ----------
    images: np.ndarray
        images as numpy array

    Returns
    ----------
    list
        intensities
    list
        coordinates
    """
    shape = images.shape
    coordinates = []
    intensities = []
    index_max = shape[:-1] if len(shape)==4 else shape[:-1] + (1,)
    for index in np.ndindex(shape[:-1]):
        xy_index = index + (slice(None),)
        I = images[xy_index]
        if full or I.any():
            index_3D = index if len(index) == 3 else index + (0, )
            add_tuple = (1, 1, 1)
            imzml_index = tuple(map(sum, zip(index_3D, add_tuple)))
            intensities.append(I)
            coordinates.append(imzml_index)
    if index_max not in coordinates:
        intensities.append([0] * shape[-1])
        coordinates.append(index_max)
    return intensities, coordinates

def get_images_from_spectra(spectra, shape):
    """
    Extracts image as a numpy array from
    spectra intensities and the shape of the image,
    i.e the tuple (width, height)

    Parameters
    ----------
    spectra: np.ndarray
        spectra as numpy array [mz*I]
    shape: tuple
        shape of the image

    Returns
    ----------
    np.ndarray
        image

    """
    intensities = spectra[:, 1, :]
    new_shape = shape
    if shape[-1] == 1:
        new_shape = shape[:-1]
    image = np.reshape(intensities, new_shape + (intensities.shape[-1],), order='F')
    return image

def get_image(imzml, mz, tol=0.01):
    """
    Extracts an ion image at a given m/z value
    and with a tolerance

    Wrapper function for imzmlparser.getionimage

    Parameters
    ----------
    imzml: imzmlparser.ImzMLParser
        parser
    mz: float
        m/z ratio of desired image
    tol: float
        tolerance on accepted m/z

    """
    return imzmlparser.getionimage(imzml, mz, tol)

def to_image_array(image):
    """
    Extracts all existing images from imzML.

    Parameters
    ----------
    image: imzmlparser.ImzMLParser
        parser

    Returns
    ----------
    np.ndarray
        image array
    """
    x, y = image.getspectrum(0)
    image_list = []
    for mz in x:
        im = get_image(image, mz)
        image_list.append(im)
    img_array = np.transpose(np.asarray(image_list))
    return img_array

def to_image_array_3D(image):
    """
    Extracts all existing images from 3D imzML

    Parameters
    ----------
    image: imzmlparser.ImzMLParser
        parser

    Returns
    ----------
    np.ndarray
        image array
    """
    x, y = image.getspectrum(0)
    image_list = []
    min_z = min(image.coordinates, key=lambda item:item[2])[2]
    max_z = max(image.coordinates, key=lambda item:item[2])[2]
    for mz in x:
        images_along_z = []
        for i in range(min_z, max_z + 1):
            im = imzmlparser.getionimage(image, mz, tol=0.01, z=i)
            images_along_z.append(im)
        image_list.append(images_along_z)
    img_array = np.transpose(np.asarray(image_list))
    return img_array


def to_nifti(image, filename):
    """
    Converts to nifti

    Parameters
    ----------
    image: np.ndarray
        image
    filename: string
        output filename

    """
    nibimg = nib.Nifti1Image(image, np.eye(4))
    nibimg.to_filename(filename)

def to_csv(array, filename):
    """
    Converts a file to csv
    containing all m/z ratio

    Parameters
    ----------
    array: np.ndarray
        mz array
    filename: str
        output filename

    """
    np.savetxt(filename, array, delimiter=";", fmt='%1.4f')


def to_tif(array, mzs, filename):
    root, ext = os.path.splitext(filename)
    to_csv(mzs, root + ".csv")
    tifffile.imwrite(filename, array, imagej=True, metadata={"axes": "ZYX", "Labels": [str(mz) for mz in mzs]})

def open_tif(filename):
    mzs = []
    tif = tifffile.TiffFile(filename)
    if 50839 in tif.pages[0].tags.keys():
        try:
            mzs_str = tif.pages[0].tags[50839].value["Labels"]
            mzs = [float(mz) for mz in mzs_str]
        except:
            pass
    else:
        root, ext = os.path.splitext(filename)
        if os.path.exists(root + ".csv"):
            mzs = np.loadtxt(root + ".csv", delimiter=";")
    try:
        im_itk = sitk.ReadImage(filename)
    except:
        return cv.imread(filename), np.array(mzs)
    return sitk.GetArrayFromImage(im_itk), np.array(mzs)
