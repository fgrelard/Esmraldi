# import esmraldi.imzmlparsermmapped as imzmlparser
import numpy as np
import os
import bisect
import matplotlib.pyplot as plt
import h5py

from skimage import exposure, img_as_ubyte
from mmappickle.dict import mmapdict
import mmappickle.stubs as stubs

from esmraldi.sparsematrix import SparseMatrix
from esmraldi.utils import progress, factors, attempt_reshape
import esmraldi.spectraprocessing as sp
import math
import SimpleITK as sitk


def get_filename_mmap(filename):
    dirname = os.path.dirname(os.path.abspath(filename))
    base = os.path.basename(filename)
    root, ext = os.path.splitext(base)
    mmap_name = dirname + "/" + root + ".mmap"
    mmap_name = "/tmp/" + root + ".mmap"
    print(mmap_name)
    return mmap_name

def get_filename_h5(filename):
    base = os.path.basename(filename)
    root, ext = os.path.splitext(base)
    h5_name = "/tmp/" + root + ".h5"
    return h5_name

def build_h5(imzml):
    filename = imzml.filename
    h5_name = get_filename_h5(filename)
    f = h5py.File(h5_name, "w")
    get_full_spectra_h5(imzml, f)
    return f

def load_h5(imzml):
    filename = imzml.filename
    h5_name = get_filename_h5(filename)
    f = h5py.File(h5_name, "r")
    return f

def load_mmap(imzml):
    filename = imzml.filename
    mmap_name = get_filename_mmap(filename)
    mdict = mmapdict(mmap_name)
    return mdict

def build_mmap(imzml):
    # mdict = load_mmap(imzml)
    filename = imzml.filename
    mmap_name = get_filename_mmap(filename)
    print(mmap_name)
    mdict = mmapdict(mmap_name, readonly=False)
    get_full_spectra_mmap(imzml, mdict)



def chunk_process(array, fn, max_iter, chunk_size, **kwds):
    out_axis = None
    if "out_axis" in kwds:
        has_out_axis = True
        out_axis = kwds["out_axis"]
    in_ind_current = 0
    out_ind_current = 0
    incr = chunk_size
    while in_ind_current < max_iter:
        progress(in_ind_current, max_iter, fn.__name__)
        in_ind_next = min(max_iter, in_ind_current + chunk_size)
        res = fn(array, in_ind_current, in_ind_next, out_ind_current, max_iter, **kwds)
        index, out = res
        array[index] = out

        in_ind_current += chunk_size
        if out_axis is not None:
            index = index[out_axis]
        out_ind_current = index.stop


def fn_coordinates(array, i, next_index, out_i, max_iter, **kwds):
    N = max_iter//2
    slice_index = (slice(None), slice(i, next_index, 1))
    pixel_numbers = kwds["pixel_numbers"]
    indices_mzs = kwds["indices_mzs"]
    out = np.array([(pixel_numbers[j-(j//N)*N], j//N, indices_mzs[j-(j//N)*N]) for j in range(i, next_index)]).T
    return slice_index, out

def fn_pixel_numbers(array, i, next_index, out_i, max_iter, **kwds):
    spectra = kwds["spectra"]
    out = np.hstack([np.repeat(j+i, len(mz)) for j, mz in enumerate(spectra[i:next_index, 0])])
    slice_index = slice(out_i, out_i+out.shape[0], 1)
    return slice_index, out

def fn_image(array, i, next_index, out_i, max_iter, **kwds):
    intensities = kwds["intensities"]
    new_shape = kwds["new_shape"]
    slice_index = (slice(None), slice(None), slice(i, next_index))
    out = np.reshape(intensities[..., i:next_index], new_shape + (next_index-i, ), order="F")
    return slice_index, out

def fn_unique(array, i, next_index, out_i, max_iter, **kwds):
    flatten = kwds["flatten"]
    out = np.unique(flatten[i:next_index])
    slice_index = slice(out_i, out_i+out.shape[0], 1)
    return slice_index, out


def fn_flatten(array, i, next_index, out_i, max_iter, **kwds):
    mzs = kwds["mzs"]
    slice_index = slice(out_i, out_i + len(mzs[i]), 1)
    out = np.array(mzs[i])
    return slice_index, out

def get_spectra_mmap(imzml, mdict, pixel_numbers=[]):
    coordinates = []
    for i in pixel_numbers:
        coordinates.append(imzml.coordinates[i])
    if len(pixel_numbers) == 0:
        coordinates = imzml.coordinates.copy()

    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]

    max_len = max(imzml.intensityLengths)
    mdict["spectra"] = stubs.EmptyNDArray((max_x*max_y*max_z, 2), dtype=object)
    for i, (x, y, z) in enumerate(coordinates):
        progress(i, len(coordinates), "Spectra")
        mz, ints = imzml.getspectrum(i)
        mdict["spectra"][i, 0] = mz
        mdict["spectra"][i, 1] = ints


def get_spectra_reduced_mmap(imzml, unique, indices_mzs, mdict, pixel_numbers=[]):

    coordinates = []
    for i in pixel_numbers:
        coordinates.append(imzml.coordinates[i])
    if len(pixel_numbers) == 0:
        coordinates = imzml.coordinates.copy()

    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]

    npixels = max_x*max_y*max_z
    int_len = max(imzml.intensityLengths)
    max_len = 256e9 // (npixels * 8)
    step, npoints = sp.min_step(unique, max_len, 0)
    print(step, npoints)
    groups = sp.index_groups(unique, step)
    peaks = np.array(sp.peak_reference_indices_median(groups))
    cumlen_groups = np.cumsum([len(g) for g in groups])-1
    indices = np.searchsorted(cumlen_groups, indices_mzs)
    c, n = 0, 0
    mdict["spectra"] = stubs.EmptyNDArray((npixels, 2), dtype=object)
    for i, (x, y, z) in enumerate(coordinates):
        progress(i, len(coordinates), "Spectra red")
        mz, ints = imzml.getspectrum(i)
        n += len(mz)
        current_indices = indices[c:n]
        change = np.concatenate((np.where(np.roll(current_indices,1)!=current_indices)[0], [len(current_indices)]))
        mdict["spectra"][i, 0] = peaks[np.unique(current_indices)]
        new_ints = []
        for j in range(len(change)-1):
            curr_ints = np.mean(ints[change[j]:change[j+1]])
            new_ints.append(curr_ints)
        mdict["spectra"][i, 1] = new_ints
        c = n

def unique_mmap(spectra, mdict):
    mzs = spectra[:, 0]
    len_full_mz = sum(len(l) for l in mzs)
    mdict["flatten"] = stubs.EmptyNDArray((len_full_mz,))
    chunk_process(mdict["flatten"], fn_flatten, len(spectra[:, 0]), 1, mzs=mzs)

    mdict["flatten_intensities"] = stubs.EmptyNDArray((len_full_mz,))
    chunk_process(mdict["flatten_intensities"], fn_flatten, len(spectra[:, 1]), 1, mzs=spectra[:, 1])

    magnitude_order = math.floor(math.log(len_full_mz, 10))
    diff_magnitude = MAX_MAGNITUDE_ORDER - magnitude_order
    chunk_size = max(1, min(len_full_mz, int(len_full_mz * 10**diff_magnitude)))
    print(chunk_size)
    mdict["unique"] = stubs.EmptyNDArray((len_full_mz,))
    chunk_process(mdict["unique"], fn_unique, len_full_mz, chunk_size, flatten=mdict["flatten"])
    return chunk_size


def get_full_spectra_mmap(imzml, mdict):
    print("Get full")
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    mzs, ints = imzml.getspectrum(0)

    print("Get spectra")
    get_spectra_mmap(imzml, mdict)
    print("End Spectra")

    chunk_size = unique_mmap(mdict["spectra"], mdict)
    unique_mzs = np.unique(mdict["unique"])
    indices_mzs = np.searchsorted(unique_mzs, mdict["flatten"]).flatten()

    number_points = len(unique_mzs)
    flatten_intensities = mdict["flatten_intensities"]
    mdict["mean_spectra"] = stubs.EmptyNDArray((number_points,))
    mean_spectra = mdict["mean_spectra"]
    for i in range(len(flatten_intensities)):
        ind_mz = indices_mzs[i]
        mean_spectra[ind_mz] += flatten_intensities[i]

    mean_spectra /= number_points
    spectra = mdict["spectra"]
    len_full_mz = sum(len(l) for l in spectra[:, 0])

    # del mdict["spectra"]
    # del mdict["unique"]
    # del mdict["flatten"]
    # get_spectra_reduced_mmap(imzml, unique_mzs, indices_mzs, mdict)
    # spectra = mdict["spectra"]
    # len_full_mz = sum(len(l) for l in spectra[:, 0])

    # chunk_size = unique_mmap(spectra, mdict)
    # unique_mzs = np.unique(mdict["unique"])
    # indices_mzs = np.searchsorted(unique_mzs, mdict["flatten"]).flatten()

    number_points = len(unique_mzs)
    if len(spectra.shape) == 2:
        #different dimensions
        mdict["pixel_numbers"] = stubs.EmptyNDArray((len_full_mz,), dtype=object)
        previous_index = 0
        pixel_numbers = mdict["pixel_numbers"]
        print("pixel numbers")
        chunk_process(pixel_numbers, fn_pixel_numbers, spectra.shape[0], max(1, chunk_size//spectra.shape[0]), spectra=spectra)
        print(mdict["pixel_numbers"])
        imsize = max_x*max_y*max_z
        shape = (imsize, 2, number_points)
        mdict["coordinates"] = stubs.EmptyNDArray((3, len_full_mz*2), dtype=np.int64)
        coordinates = mdict["coordinates"]
        print("coordinates")
        chunk_process(coordinates, fn_coordinates, len_full_mz*2, chunk_size, pixel_numbers=pixel_numbers, indices_mzs=indices_mzs, out_axis=1)
        mdict["data"] = np.hstack(spectra.T.flatten())
        mdict["shape"] = shape
        print(type(mdict["spectra"]))
        del mdict["spectra"]
        del mdict["pixel_numbers"]
        return

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
    print(spectra.shape)
    mdict["image"] = stubs.EmptyNDArray(new_shape + (spectra.shape[-1], ))
    image = mdict["image"]
    step = max(1, MAX_NUMBER//np.prod(new_shape))
    chunk_process(image, fn_image, spectra.shape[-1], step, intensities=intensities, new_shape=new_shape, out_axis=-1)
    print("end images")
    return


def get_full_spectra_h5(imzml, f):
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

    # full_spectra = f.create_dataset("full_spectra", (prod_shape, 2, 0), maxshape=(prod_shape, 2, number_points), chunks=(prod_shape, 2, 1,))
    # print(full_spectra.size, full_spectra.chunks)
    # for i in range(1, full_spectra.maxshape[-1], full_spectra.chunks[-1]):
    #     if i%100000==0:
    #         print(i)
    #     current_mz = unique_mzs[i]
    #     mz_indices = np.argwhere(mzs == current_mz)
    #     mzs_array = np.repeat(current_mz, prod_shape)
    #     ints_array = intensities[mz_indices[:, 0], mz_indices[:, 1]]
    #     full_spectra.resize(i+1, axis=2)
    #     full_spectra[:, 0, i] = mzs_array
    #     full_spectra[:, 1, i] = np.zeros((prod_shape,))
    #     full_spectra[mz_indices[:, 0], 1, i] = ints_array




    imsize = max_x*max_y*max_z
    shape = f.create_dataset("shape", (3,))
    shape[:] = (imsize, 2, number_points)

    pixel_numbers = f.create_dataset("pixel_numbers", (len_full_mz,), chunks=(chunk_size,))
    for i in range(0, pixel_numbers.size, pixel_numbers.chunks[0]):
        step = pixel_numbers.chunks[0]
        current_mzs = spectra[i//spectra.shape[-1]:(i+step)//spectra.shape[-1], 0, :]
        pixel_numbers[i:i+step] = np.hstack([np.repeat(j+i//spectra.shape[-1], len(mz)) for j, mz in enumerate(current_mzs)])

    print(indices_mzs)

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
