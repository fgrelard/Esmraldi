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

from mmappickle.dict import mmapdict
from mmappickle.stubs import EmptyNDArray

from esmraldi.sparsematrix import SparseMatrix
from sparse import COO
import time
import math
from functools import reduce


MAX_MAGNITUDE_ORDER = 6

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
    image_normalized = np.zeros_like(image, dtype=np.uint8)
    if len(image.shape) <= 2:
        image_normalized = np.uint8(cv.normalize(image, None, 0, 255, cv.NORM_MINMAX))
    else:
        z = image.shape[-1]
        for k in range(z):
            slice2D = image[..., k]
            slice2DNorm = np.uint8(cv.normalize(slice2D, None, 0, 255, cv.NORM_MINMAX))
            image_normalized[..., k] = slice2DNorm
    return image_normalized

def get_full_spectra(imzml):
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    mzs, ints = imzml.getspectrum(0)
    number_points = len(ints)
    zeros_ints = [0 for i in range(number_points)]

    full_spectra = np.zeros((max_x*max_y*max_z, 2, number_points))
    full_spectra[:,0,:] = mzs

    spectra = get_spectra(imzml)
    mzs = spectra[:, 0]
    if len(spectra.shape) == 2:
        #different dimensions
        unique_mzs, indices_mzs = np.unique(np.hstack(mzs), return_inverse=True)
        number_points = len(unique_mzs)
        pixel_numbers = np.hstack([np.repeat(i, len(mzs[i])) for i in range(len(mzs))])
        imsize = max_x*max_y*max_z
        shape = (imsize, 2, number_points)
        coordinates = np.array([(pixel_numbers[i], j, indices_mzs[i]) for j in range(2) for i in range(len(pixel_numbers))]).T
        full_spectra_sparse = SparseMatrix(coordinates, np.hstack(spectra.T.flatten()), shape)
        return full_spectra_sparse

    for i, (x, y, z) in enumerate(imzml.coordinates):
        real_index = (x-1) + (y-1) * max_x + (z-1) * max_x * max_y
        mz, ints = imzml.getspectrum(i)
        full_spectra[real_index, 0] = mz
        full_spectra[real_index, 1] = ints


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


def get_filename_mmap(imzml):
    filename = imzml.filename
    base = os.path.basename(filename)
    root, ext = os.path.splitext(base)
    mmap_name = "/tmp/" + root + ".mmap"
    return mmap_name

def get_filename_h5(imzml):
    filename = imzml.filename
    base = os.path.basename(filename)
    root, ext = os.path.splitext(base)
    h5_name = "/tmp/" + root + ".h5"
    return h5_name

def build_h5(imzml):
    h5_name = get_filename_h5(imzml)
    f = h5py.File(h5_name, "w")
    get_full_spectra_h5(imzml, f)
    return f

def load_h5(imzml):
    h5_name = get_filename_h5(imzml)
    f = h5py.File(h5_name, "r")
    return f

def load_mmap(imzml):
    mmap_name = get_filename_mmap(imzml)
    mdict = mmapdict(mmap_name)
    return mdict

def build_mmap(imzml):
    # mdict = load_mmap(imzml)
    # mdict.vacuum()
    mmap_name = get_filename_mmap(imzml)
    mdict = mmapdict(mmap_name, readonly=False)
    get_full_spectra_mmap(imzml, mdict)


def get_spectra_mmap(imzml, mdict, pixel_numbers=[]):
    coordinates = []
    for i in pixel_numbers:
        coordinates.append(imzml.coordinates[i])
    if len(pixel_numbers) == 0:
        coordinates = imzml.coordinates.copy()

    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]

    mdict["spectra"] = EmptyNDArray((max_x*max_y*max_z, 2), dtype=object)
    for i, (x, y, z) in enumerate(coordinates):
        mz, ints = imzml.getspectrum(i)
        mdict["spectra"][i, 0] = mz
        mdict["spectra"][i, 1] = ints


def chunk_process(array, fn, max_iter, chunk_size, **kwds):
    i = 0
    incr = chunk_size
    while i < max_iter:
        next_index = min(max_iter, i + chunk_size)
        res = fn(array, i, next_index, max_iter, **kwds)
        if len(res) == 3:
            index, out, incr = res
        else:
            index, out = res
        i += incr
        array[index] = out


def fn_coordinates(array, i, next_index, max_iter, **kwds):
    N = max_iter//2
    slice_index = (slice(None), slice(i, next_index, 1))
    pixel_numbers = kwds["pixel_numbers"]
    indices_mzs = kwds["indices_mzs"]
    out = np.array([(pixel_numbers[j-(j//N)*N], j//N, indices_mzs[j-(j//N)*N]) for j in range(i, next_index)]).T
    return slice_index, out

def fn_pixel_numbers(array, i, next_index, max_iter, **kwds):
    mzs = kwds["mzs"]
    sum_len = sum(len(l) for l in mzs[i:next_index])
    slice_index = slice(i, next_index*sum_len, 1)
    out = np.hstack([np.repeat(j, len(mz)) for j, mz in enumerate(mzs[i:next_index])])
    return slice_index, out

def fn_image(array, i, next_index, max_iter, **kwds):
    intensities = kwds["intensities"]
    new_shape = kwds["new_shape"]
    slice_index = (slice(None), slice(None), slice(i, next_index))
    out = np.reshape(intensities[..., i:next_index], new_shape + (next_index-i, ), order="F")
    return slice_index, out

def fn_unique(array, i, next_index, max_iter, **kwds):
    flatten = kwds["flatten"]
    out = np.unique(flatten[i:next_index])
    chunk_size = out.shape[0]
    slice_index = slice(i, i+chunk_size, 1)
    return slice_index, out, chunk_size

def fn_flatten(array, i, next_index, max_iter, **kwds):
    mzs = kwds["mzs"]
    previous_index = next_index
    next_index += len(mzs[i])
    slice_index = slice(previous_index, next_index, 1)
    out = np.array(mzs[i])
    return slice_index, out


def get_full_spectra_h5(imzml, f):
    def factors(n):
        return set(reduce(list.__add__,
            ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

    print("spectra h5")
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    mzs, ints = imzml.getspectrum(0)
    number_points = len(ints)

    prod_shape = max_x*max_y*max_z
    spectra = f.create_dataset("spectra", (prod_shape, 2, 0), maxshape=(prod_shape, 2, None), chunks=True)
    for i, (x, y, z) in enumerate(imzml.coordinates):
        mz, ints = imzml.getspectrum(i)
        spectra.resize(max(spectra.shape[-1], len(mz)), axis=2)
        spectra[i, 0, -len(mz):] = mz
        spectra[i, 1, -len(mz):] = ints

    mzs = spectra[:, 0]
    intensities = spectra[:, 1]
    len_full_mz = np.prod(mzs.shape)
    print(len_full_mz)

    magnitude_order = math.floor(math.log(spectra.shape[-1], 10))
    diff_magnitude = MAX_MAGNITUDE_ORDER - magnitude_order
    divisors = np.array(sorted(factors(spectra.shape[0])))
    index_max_divisor = np.where(divisors < 10**diff_magnitude)[0][-1]
    number_pixels = divisors[index_max_divisor]
    max_chunk_size = spectra.shape[-1] * number_pixels
    chunk_size = max(1, min(len_full_mz, max_chunk_size))
    data = f.create_dataset("data", (0,), maxshape=(np.prod(spectra.shape),), chunks=(chunk_size,))
    for i in range(0, data.maxshape[0], data.chunks[0]):
        mzi_index = i // len_full_mz
        i_parity = i-mzi_index*len_full_mz
        current_index = i_parity//(spectra.shape[-1])
        next_index = (i_parity+data.chunks[0])//(spectra.shape[-1])
        current_spectra = spectra[current_index:next_index, mzi_index, :].flatten()
        data.resize(i+current_spectra.shape[0], axis=0)
        data[i:i+current_spectra.shape[0]] = current_spectra




    unique = f.create_dataset("unique", (0,), maxshape=(len_full_mz,), chunks=(chunk_size,))
    for i in range(0, spectra.size//2, unique.chunks[0]):
        current_mzs = spectra[i//spectra.shape[-1]:(i+unique.chunks[0])//spectra.shape[-1], 0, :]
        current_mzs = np.hstack(current_mzs)
        current_unique = np.unique(current_mzs)

        current_index = unique.shape[0]
        next_index = current_index + current_unique.shape[0]
        unique.resize(next_index, axis=0)
        unique[current_index:next_index] = current_unique


    unique_mzs = np.unique(unique)
    indices_mzs = np.searchsorted(unique_mzs, mzs)

    number_points = len(unique_mzs)

    full_spectra = f.create_dataset("full_spectra", (prod_shape, 2, 0), maxshape=(prod_shape, 2, number_points), chunks=(prod_shape, 2, 1,))
    print(full_spectra.size, full_spectra.chunks)
    for i in range(1, full_spectra.maxshape[-1], full_spectra.chunks[-1]):
        if i%100000==0:
            print(i)
        current_mz = unique_mzs[i]
        mz_indices = np.argwhere(mzs == current_mz)
        mzs_array = np.repeat(current_mz, prod_shape)
        ints_array = intensities[mz_indices[:, 0], mz_indices[:, 1]]
        full_spectra.resize(i+1, axis=2)
        full_spectra[:, 0, i] = mzs_array
        full_spectra[:, 1, i] = np.zeros((prod_shape,))
        full_spectra[mz_indices[:, 0], 1, i] = ints_array


    print(unique_mzs[0], mzs.shape)
    mz = np.argwhere(mzs == unique_mzs[10000])
    print(mz)
    print()

    indices_mzs = indices_mzs.flatten()
    indices_2D = np.unravel_index(mz_indices.flatten(), (max_x, max_y))


    imsize = max_x*max_y*max_z
    shape = f.create_dataset("shape", (3,))
    shape[:] = (imsize, 2, number_points)

    pixel_numbers = f.create_dataset("pixel_numbers", (len_full_mz,), chunks=(chunk_size,))
    for i in range(0, pixel_numbers.size, pixel_numbers.chunks[0]):
        step = pixel_numbers.chunks[0]
        current_mzs = spectra[i//spectra.shape[-1]:(i+step)//spectra.shape[-1], 0, :]
        pixel_numbers[i:i+step] = np.hstack([np.repeat(j+i//spectra.shape[-1], len(mz)) for j, mz in enumerate(current_mzs)])

    image = f.create_dataset("image", (0,), maxshape=(max_x, max_y, number_points), chunks=(2,))


    coordinates = f.create_dataset("coordinates", (3, len_full_mz*2), chunks=(3, chunk_size,))
    N = len_full_mz
    step = coordinates.chunks[-1]
    for i in range(0, coordinates.shape[-1], step):
        next_ind = min(i+step, coordinates.shape[-1])
        test = np.array([(pixel_numbers[j-(j//N)*N], j//N, indices_mzs[j-(j//N)*N]) for j in range(i, next_ind)]).T
        print(test.shape, i, i+step)
        coordinates[:, i:i+step] = test

    print(pixel_numbers[:100], coordinates[:, :100])

    exit(0)

    # full_spectra = SparseMatrix(mdict["coordinates"], mdict["data"], mdict["shape"], sorted=True, has_duplicates=False)


    # full_spectra_sparse = SparseMatrix(coordinates, np.hstack(spectra.T.flatten()), shape)
    return full_spectra_sparse

def get_full_spectra_mmap(imzml, mdict):
    print("Get full")
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    mzs, ints = imzml.getspectrum(0)
    number_points = len(ints)

    print("Get spectra")
    get_spectra_mmap(imzml, mdict)
    print("End Spectra")
    spectra = mdict["spectra"]
    print("Get mzs")
    mzs = spectra[:, 0]

    print("compute flatten array")
    len_full_mz = sum(len(l) for l in mzs)
    mdict["flatten"] = EmptyNDArray((len_full_mz,))
    chunk_process(mdict["flatten"], fn_flatten, len(mzs), 1, mzs=mzs)

    mdict["unique"] = EmptyNDArray((len_full_mz,))
    print("compute unique")
    chunk_process(mdict["unique"], fn_unique, len_full_mz, MAX_NUMBER, flatten=mdict["flatten"])
    unique_mzs, indices_mzs = np.unique(mdict["unique"], return_inverse=True)
    print("unique", type(unique_mzs))
    number_points = len(unique_mzs)

    if len(spectra.shape) == 2:
        #different dimensions
        len_full_mz = sum(len(l) for l in mzs)
        mdict["pixel_numbers"] = EmptyNDArray((len_full_mz,), dtype=object)
        previous_index = 0
        pixel_numbers = mdict["pixel_numbers"]
        print("pixel numbers")
        chunk_process(pixel_numbers, fn_pixel_numbers, len(mzs), MAX_NUMBER, mzs=mzs)
        imsize = max_x*max_y*max_z
        shape = (imsize, 2, number_points)
        mdict["coordinates"] = EmptyNDArray((3, len_full_mz*2), dtype=np.int64)
        coordinates = mdict["coordinates"]
        print("coordinates")
        chunk_process(coordinates, fn_coordinates, len(pixel_numbers)*2, MAX_NUMBER, pixel_numbers=pixel_numbers, indices_mzs=indices_mzs)
        mdict["data"] = np.hstack(spectra.T.flatten())
        mdict["shape"] = shape
        print(type(mdict["spectra"]))

        full_spectra = SparseMatrix(mdict["coordinates"], mdict["data"], mdict["shape"], sorted=True, has_duplicates=False)
        get_images_from_spectra_mmap(full_spectra, (max_x, max_y, max_z), mdict)
        del mdict["spectra"]
        del mdict["pixel_numbers"]
        del mdict["unique"]
        del mdict["flatten"]
        return

    mdict["full_spectra"] = EmptyNDArray((max_x*max_y*max_z, 2, number_points))
    previous_index = 0
    for i, (x, y, z) in enumerate(imzml.coordinates):
        real_index = (x-1) + (y-1) * max_x + (z-1) * max_x * max_y
        mz, ints = imzml.getspectrum(i)
        new_index = previous_index + len(mz)
        indices = indices_mzs[previous_index:new_index]
        previous_index = new_index
        mdict["full_spectra"][real_index, 0, indices] = mz
        mdict["full_spectra"][real_index, 1, indices] = ints

    del mdict["spectra"]

def get_images_from_spectra_mmap(spectra, shape, mdict):
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
    mdict["image"] = EmptyNDArray(new_shape + (spectra.shape[-1], ))
    image = mdict["image"]
    step = max(1, MAX_NUMBER//np.prod(new_shape))
    chunk_process(image, fn_image, spectra.shape[-1], step, intensities=intensities, new_shape=new_shape)
    print("end images")
    return



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
    print(type(intensities))
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
