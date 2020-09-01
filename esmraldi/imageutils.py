import numpy as np
import SimpleITK as sitk
import esmraldi.segmentation as seg
import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.transform as transform

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import gaussian_filter1d
from dtw import *


def center_images(images, size):

    shape_3D = size + (len(images),)
    image_3D = np.zeros(shape_3D)
    for i in range(len(images)):
        im = images[i]
        shape = im.shape
        start = tuple((size[i] - shape[i])//2 for i in range(len(size)))
        end = tuple(start[i] + shape[i] for i in range(len(shape)))
        index = tuple(slice(start[i], end[i]) for i in range(len(start)))
        index += (i,)
        image_3D[index] = im
    return image_3D

def resize(image, size):
    """
    Resize the image to a given size.

    Parameters
    ----------
    image: sitk.Image
        input image
    size: tuple
        new size of the image

    Returns
    ----------
    sitk.Image
        new resized image

    """
    image_array = sitk.GetArrayFromImage(image)
    reversed_size = np.array(size)[::-1]
    resized = transform.resize(image_array, reversed_size, order=0)
    resized_itk = sitk.GetImageFromArray(resized)
    return resized_itk

def max_area_slices(image):
    max_area = 0
    for i in range(image.shape[-1]):
        im = image[..., i]
        area = seg.properties_largest_area_cc(im).area
        if area > max_area:
            max_area  = area
    return max_area

def relative_area(image):
    max_area = max_area_slices(image)
    relative_area_image = []
    for i in range(image.shape[-1]):
        im = image[..., i]
        area = seg.properties_largest_area_cc(im).area
        relative_area = area / max_area
        relative_area_image.append(relative_area)
    return np.array(relative_area_image)

def enforce_continuity_values(sequence):
    continued_sequence = np.copy(sequence)
    for i in range(1,len(sequence)-1):
        previous = sequence[i-1]
        current = sequence[i]
        next = sequence[i+1]
        continuity = current + 1
        if previous == current and continuity < next:
            continued_sequence[i] = continuity
    return continued_sequence

def slice_correspondences(reference, target, sigma, is_reversed=False, is_continuity=True):
    relative_area_reference = relative_area(reference)
    relative_area_target = relative_area(target)

    if is_reversed:
        relative_area_target = relative_area_target[::-1]

    relative_area_target = gaussian_filter1d(np.copy(relative_area_target), sigma)
    alignment = dtw(relative_area_target, relative_area_reference, step_pattern=asymmetric, keep_internals=True, open_begin=True, open_end=True)
    correspondences = alignment.index2

    if is_continuity:
        correspondences = enforce_continuity_values(correspondences)

    if is_reversed:
        correspondences = correspondences[::-1]

    return correspondences

def slice_correspondences_manual(reference, target, resolution_reference, resolution_target, slices_reference, slices_target, is_reversed=False):
    correspondences = []

    physical_slice_reference = [slice*resolution_reference for slice in slices_reference]
    physical_slice_target = [slice*resolution_target for slice in slices_target]

    if is_reversed:
        physical_slice_reference = [max(physical_slice_reference) - elem for elem in physical_slice_reference]

    print(physical_slice_reference)
    print(physical_slice_target)

    correspondences = [(np.abs(np.array(physical_slice_reference) - i)).argmin() for i in physical_slice_target]
    return np.array(correspondences)

def compute_DT(image_itk):
    image_array = sitk.GetArrayFromImage(image_itk)
    image_bin = np.where(image_array > 0, 255, 0)
    image_dt = distance_transform_edt(image_bin)
    image_dt_itk = sitk.GetImageFromArray(image_dt.astype("float32"))
    return image_dt_itk

def estimate_noise(I):
    """
    Estimates the noise in an image
    by convolution with a kernel.

    See: Fast Noise Variance Estimation,
    Immerkaear et al.

    Parameters
    ----------
    I: np.ndarray
        image

    Returns
    ----------
    float
        the noise standard deviation
    """
    H, W = I.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    return sigma
