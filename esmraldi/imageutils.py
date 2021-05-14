import numpy as np
import SimpleITK as sitk
import esmraldi.segmentation as seg
import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.transform as transform

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import uniform_filter, gaussian_filter1d
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


def variance_image(image, size=3):
    image_mean = uniform_filter(image.astype(float), (size,size))
    image_mean_sq = uniform_filter(image.astype(float)**2, (size,size))
    image_var = image_mean_sq - image_mean**2
    image_var[image_var<0] = 0
    return image_var

def stddev_image(image, size=3):
    image_var = variance_image(image, size)
    return np.sqrt(image_var)

def mse_numpy(fixed_array, moving_array):
    diff_sq = (moving_array - fixed_array) ** 2
    return np.mean(diff_sq)

def mse(fixed, moving):
    fixed_array = sitk.GetArrayFromImage(fixed)
    moving_array = sitk.GetArrayFromImage(moving)
    return mse_numpy(fixed_array, moving_array)

def dt_mse(fixed, moving):
    moving_dt = compute_DT(moving)
    fixed_array = sitk.GetArrayFromImage(fixed)
    moving_dt_array = sitk.GetArrayFromImage(moving_dt)
    return mse_numpy(fixed_array, moving_dt_array)

def mse_stddev(fixed, moving):
    fixed_array = sitk.GetArrayFromImage(fixed)
    moving_array = sitk.GetArrayFromImage(moving)
    fixed_stddev = stddev_image(fixed_array)
    moving_stddev = stddev_image(moving_array)
    return mse_numpy(fixed_stddev, moving_stddev)


def export_figure_matplotlib(f_name, arr, arr2=None, dpi=200, resize_fact=1, plt_show=False):
    """
    Export array as figure in original resolution
    :param arr: array of image to save in original resolution
    :param f_name: name of file where to save figure
    :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
    :param dpi: dpi of your screen
    :param plt_show: show plot or not
    """
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1]/dpi, arr.shape[0]/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap="gray")
    if arr2 is not None:
        ax.imshow(arr2, cmap="Reds", alpha=0.5)
    plt.savefig(f_name, dpi=(dpi * resize_fact))
    if plt_show:
        plt.show()
    else:
        plt.close()
