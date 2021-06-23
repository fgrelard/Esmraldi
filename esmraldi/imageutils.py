import numpy as np
import SimpleITK as sitk
import esmraldi.segmentation as seg
import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.transform as transform
import skimage.feature as feature

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import uniform_filter, gaussian_filter1d
from dtw import *
from scipy.stats.stats import pearsonr
import skimage.draw as draw
import bresenham as bresenham
import math

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

def radius_maximal_balls(image):
    dt_image = compute_DT(image)
    dt_array = sitk.GetArrayFromImage(dt_image)
    sorted_ind = np.argsort(dt_array, axis=None)
    xy_indices = np.column_stack(np.unravel_index(sorted_ind[::-1], dt_array.shape))
    local_max_dt = np.zeros_like(dt_array)
    for ind in xy_indices:
        x,y = ind
        current_value = dt_array[x, y]
        if not dt_array.any():
            break
        elif not current_value:
            continue
        rr, cc = draw.disk(ind, current_value, shape=dt_array.shape)
        dt_array[rr, cc] = 0
        image_disk = np.zeros_like(dt_array)
        image_disk[rr, cc] = 1
        current_disk = np.where((local_max_dt == 0) & (image_disk == 1))
        local_max_dt[current_disk] = current_value
    return local_max_dt


def local_max_dt(image):
    dt_image = compute_DT(image)
    dt_array = sitk.GetArrayFromImage(dt_image)
    indices = feature.peak_local_max(dt_array, 1)
    x, y = indices.T
    local_max = np.zeros_like(dt_array)
    local_max[x, y] = 1
    local_max_itk = sitk.GetImageFromArray(local_max)
    return local_max_itk

def normalized_dt(image):
    dt_image = compute_DT(image)
    local_max_dt = radius_maximal_balls(image)
    image_array = sitk.GetArrayFromImage(image)
    normalized_dt_image = np.zeros_like(image_array)
    np.divide(sitk.GetArrayFromImage(dt_image), local_max_dt, out=normalized_dt_image, where=local_max_dt!=0)
    normalized_dt_itk = sitk.GetImageFromArray(normalized_dt_image**2)
    return normalized_dt_itk

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


def voronoi_diagram(points, shape):
    width, height = shape
    centers_x, centers_y = points[:, 0], points[:, 1]
    # Create grid containing all pixel locations in image
    x, y = np.meshgrid(np.arange(width), np.arange(height))


    # Find squared distance of each pixel location from each center: the (i, j, k)th
    # entry in this array is the squared distance from pixel (i, j) to the kth center.
    squared_dist = (x[:, :, np.newaxis] - centers_x[np.newaxis, np.newaxis, :]) ** 2 + \
                   (y[:, :, np.newaxis] - centers_y[np.newaxis, np.newaxis, :]) ** 2

    # Find closest center to each pixel location
    indices = np.argmin(squared_dist, axis=2)  # Array containing index of closest center

    return indices.T

def simple_vcm(voronoi, point, r):
    #Find cells in disk of radius r
    sites, cells = voronoi
    rr, cc = draw.disk(point, r, shape=cells.shape)
    cell_indices = np.unique(cells[rr, cc])
    cell_intersection = np.isin(cells, cell_indices)
    cell_intersection_xy = np.argwhere(cell_intersection > 0)

    cell_vectors = []
    for cell_index in cell_indices:
        site = sites[cell_index]
        cell = np.argwhere(np.isin(cells, cell_index) > 0)
        current_cell_vectors = (cell - site).tolist()
        cell_vectors += current_cell_vectors
    cell_vectors = np.array(cell_vectors)


    dist = (cell_vectors**2).sum(axis=-1)
    cov = np.cov(cell_vectors.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    sortedeigen  = np.array(sorted(zip(eigvals,eigvecs.T),\
                                    key=lambda x: x[0].real, reverse=True))
    eigvals, eigvecs = sortedeigen[:, 0], sortedeigen[:, 1]
    #Farthest point
    d_max = np.sqrt(dist.max())
    return eigvecs, d_max


def estimate_plane(obj, voronoi, point, max_r=np.inf):
    sites, cells = voronoi
    eigvecs, d_max = simple_vcm(voronoi, point, 2.0)
    d_max = min(d_max, max_r)
    eigvecs, _ = simple_vcm(voronoi, point, d_max)
    eigvec_0 = eigvecs[0]
    end = [math.ceil(point[i] + eigvec_0[i]*d_max) for i in range(len(point))]
    end2 = [math.ceil(point[i] - eigvec_0[i]*d_max) for i in range(len(point))]
    plane_xy = list(bresenham.bresenham(point[0], point[1], end[0], end[1]))
    plane_xy += list(bresenham.bresenham(point[0], point[1], end2[0], end2[1]))
    plane_xy = np.array(plane_xy)

    set_obj = set((tuple(i) for i in obj))
    set_plane_xy = set((tuple(i) for i in plane_xy))

    set_intersection = set_obj.intersection(set_plane_xy)
    set_intersection.add(tuple(point))
    plane_xy = np.array(list(set_intersection))

    d_max = np.sqrt(((plane_xy - point)**2).max())
    plane = np.zeros_like(cells)
    plane[plane_xy[:, 0], plane_xy[:, 1]] = 1
    return plane, d_max

def local_radius(image):
    edges = feature.canny(image)
    obj = np.argwhere(image > 0)
    sites = np.argwhere(edges > 0)
    cells = voronoi_diagram(sites, edges.shape)
    cells_restricted = np.where(image > 0, cells, 0)

    voronoi = [sites, cells]
    dt_itk = compute_DT(sitk.GetImageFromArray(image))
    dt_array = sitk.GetArrayFromImage(dt_itk)
    sorted_ind = np.argsort(dt_array, axis=None)
    xy_indices = np.column_stack(np.unravel_index(sorted_ind[::-1], dt_array.shape))
    local_radius_map = np.zeros_like(image)
    for ind in xy_indices:
        x, y = ind
        if dt_array[x, y] == 0:
            break
        if local_radius_map[x,y] > 0:
            continue
        plane, d_max = estimate_plane(obj, voronoi, ind, dt_array.max())
        local_radius_map[plane > 0] = d_max
    return local_radius_map
