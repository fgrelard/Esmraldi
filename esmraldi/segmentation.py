"""
Module for the segmentation
"""
import sys
import math
import numpy as np
import pyimzml.ImzMLParser as imzmlparser
import scipy.spatial.distance as dist
import esmraldi.imageutils as imageutils
import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import esmraldi.fusion as fusion
import cv2 as cv
import SimpleITK as sitk

from sklearn.decomposition import PCA
from sklearn.feature_extraction import grid_to_graph
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage.measure import find_contours
from skimage import measure
from skimage.filters import threshold_otsu, rank, sobel, threshold_multiotsu
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import data, color, util
from skimage.draw import disk as drawdisk
from skimage.morphology import binary_erosion, closing, disk, remove_small_objects
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import binary_fill_holes

import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import pearsonr

def max_variance_sort(image_maldi):
    """
    Sort a stack image along the z-axis
    according to the maximum intensity variation

    Parameters
    ----------
    image_maldi: numpy.ndarray
        input image

    Returns
    ----------
    numpy.ndarray
        the sorted image stack

    """
    x, y = image_maldi.getspectrum(0)
    image_list = []
    for mz in x:
        im = imzmlparser.getionimage(image_maldi, mz, tol=0.1)
        image_list.append({"mz": mz, "im": im})
    image_list.sort(key=lambda elem: np.var(elem["im"]), reverse=True)
    return image_list


def properties_largest_area_cc(ccs):
    """
    Extracts the connected component
    with the largest area.

    Parameters
    ----------
    ccs: numpy.ndarray
        connected components

    Returns
    ----------
    RegionProperties
        connected component with largest area

    """
    regionprops = measure.regionprops(ccs)
    if len(regionprops) == 0:
        return -1
    areas = lambda r: r.area
    argmax = max(regionprops, key=areas)
    return argmax

def properties_median_perimeter(ccs):
    regionprops = measure.regionprops(ccs)
    if len(regionprops) == 0:
        return -1
    perimeters = [r.perimeter for r in regionprops]
    arg_median = regionprops[np.argsort(perimeters)[len(perimeters)//2]]
    return arg_median


def region_property_to_cc(ccs, regionprop):
    """
    Extracts the connected component associated
    with the region.

    Parameters
    ----------
    ccs: numpy.ndarray
        connected components
    regionprop: RegionProperties
        desired region

    Returns
    ----------
    numpy.ndarray
        the binary image (mask) of the desired region
    """
    label = regionprop.label
    cc = np.where(ccs == label, 0, 255)
    return cc



def sort_size_ascending(images, threshold):
    """
    Sort images in ascending order
    of the number of pixels inside
    greather than a given threshold.

    Parameters
    ----------
    images: np.ndarray
        array of images
    threshold: int
        threshold to count pixels

    Returns
    ----------
    np.ndarray
        sorted array of images

    """
    sizes = []
    for index in np.ndindex(images.shape[2:]):
        current_index = (slice(None), slice(None)) + (index)
        current_image = images[current_index]
        current_image = np.uint8(cv.normalize(current_image, None, 0, 255, cv.NORM_MINMAX))
        size = np.count_nonzero(current_image > threshold)
        sizes.append(size)
    indices_sort = np.argsort(sizes)
    new_images = images[..., indices_sort]
    return new_images




def region_growing(images, seedList, lower_threshold):
    """
    Region growing in an image stack
    with ITK.

    All the images in the stack are processed sequentially
    and the seeds at step n depends on the segmentation
    by region growing at step n-1.

    Parameters
    ----------
    images: numpy.ndarray
        image stack
    seedList: list
        list of 2D points to initialize the region growing
    lower_threshold: int
        lower threshold for the region growing

    Returns
    ----------
    list
        seeds as 2d points

    """
    seeds = seedList.copy()
    evolution_segmentation = np.zeros_like(images)
    for index in np.ndindex(images.shape[2:]):
        current_index = (slice(None), slice(None)) + (index)
        current_image = images[current_index]
        current_image = np.uint8(cv.normalize(current_image, None, 0, 255, cv.NORM_MINMAX))
        sitk_image = sitk.GetImageFromArray(current_image)
        seg_con = sitk.ConnectedThreshold(sitk_image, seedList=list(seeds),
                                          lower=int(lower_threshold), upper=255, replaceValue=1)

        np_seg_con = sitk.GetArrayFromImage(seg_con)
        locations_seeds = np.where(np_seg_con == 1)
        labels = measure.label(np_seg_con, background=0)
        regionprop = properties_largest_area_cc(labels)
        if regionprop != -1:
            largest_cc = region_property_to_cc(labels, regionprop)
            seeds = seeds.union(set(((int(coord[0]), int(coord[1])) for coord in regionprop.coords)))
            image = np.zeros_like(current_image)
            x = [elem[0] for elem in seeds]
            y = [elem[1] for elem in seeds]
            image[x,y] = 1
            evolution_segmentation[current_index] = image
    return list(seeds), evolution_segmentation



def average_area(images):
    """
    Average area of largest CCs on a collection of images.

    Parameters
    ----------
    images: np.ndarray
        collection of images

    Returns
    ----------
    int
        average area of largest CCs
    """
    sum_area = 0
    z = images.shape[-1]
    count = 0
    for k in range(z):
        slice2D = images[..., k]
        slice2DNorm = np.uint8(cv.normalize(slice2D, None, 0, 255, cv.NORM_MINMAX))
        try:
            otsu = threshold_otsu(slice2DNorm)
            labels = measure.label(slice2DNorm > otsu, background=0)
            regionprop = properties_largest_area_cc(labels)
            if regionprop != -1:
                sum_area += regionprop.area
                count += 1
        except Exception as e:
            pass
    return sum_area / count if count != 0 else 0

def select_class_max_value(image_maldi, y_kmeans, nb_class):
    """
    Chooses label associated with kmeans cluster where
    images have highest average intensity.

    Parameters
    ----------
    image_maldi: np.ndarray
        images
    y_kmeans: np.ndarray
        labels
    nb_class: int
        number of clusters used by kmeans

    Returns
    ----------
    int
        class label

    """
    max_value = 0
    index = -1
    for i in range(nb_class):
        similar_images = image_maldi[..., y_kmeans==i]
        reshaped = similar_images.reshape(-1, similar_images.shape[-1])
        av_max_value = np.mean(np.amax(reshaped, axis=0))
        if av_max_value > max_value:
            index = i
            max_value = av_max_value
    return index

def select_class_area(image_maldi, y_kmeans, nb_class):
    """
    Chooses labels associated with highest average
    area of largest CCs.

    Parameters
    ----------
    image_maldi: np.ndarray
        images
    y_kmeans: np.ndarray
        labels
    nb_class: int
        number of clsuters used by kmeans

    Returns
    ----------
    int
        class label
    """
    max_area = 0
    index = -1
    for i in range(nb_class):
        similar_images = image_maldi[..., y_kmeans==i]
        av_area = average_area(similar_images)
        if av_area > max_area:
            index = i
            max_area = av_area
    return index


def detect_circle(image, threshold, min_radius, max_radius):
    """
    Detects a circle in an image.

    Uses hough transform over several radii.

    Parameters
    ----------
    image: np.ndarray
        image
    threshold: int
        threshold for binary image
    min_radius: float
        lower bound for radii
    max_radius: float
        upper bound for radii

    Returns
    ----------
    tuple
        x,y,r: circle center + radii

    """
    cond = np.where(image < threshold)
    image_copy = np.copy(image)
    image_copy[cond] = 0
    edges = canny(image_copy, sigma=3, low_threshold=10, high_threshold=40)

    # Detect two radii
    hough_radii = np.arange(min_radius, max_radius, 10)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    if len(cx) > 0:
        return cx[0], cy[0], radii[0]
    return -1, -1, -1



def detect_tube(image, threshold=150, min_radius=10, max_radius=50):
    """
    Detects the most frequent circle across several slices
    (3D volume).

    Parameters
    ----------
    image: np.ndarray
        image
    threshold: int
        threshold for binary image
    min_radius: float
        lower bound for radii
    max_radius: float
        upper bound for radii

    Returns
    ----------
    tuple
        x,y,r: circle center + radii
    """
    cy, cx, radii = [], [], []
    for i in range(image.shape[0]):
        center_x, center_y, radius = detect_circle(image[i, :,:], threshold, min_radius, max_radius)
        if center_y >= 0:
            cy.append(center_y)
            cx.append(center_x)
            radii.append(radius)
    center_y = np.median(cy)
    center_x = np.median(cx)
    radius = np.median(radii)
    return center_x, center_y, radius

def fill_circle(center_x, center_y, radius, image, color=0):
    """
    Fills a circle with a given value (default: 0).

    Parameters
    ----------
    center_x: float
        center_x of circle
    center_y: float
        center_y of circle
    radius: float
        radius of circle
    image: np.ndarray
        image where the circle must be filled
    color: int
        value to replace

    Returns
    ----------
    np.ndarray
        image with filled circle

    """
    image2 = np.copy(image)
    dim = len(image2.shape)
    rr, cc = drawdisk((int(center_y), int(center_x)), int(radius), shape=image2.shape[dim-2:])
    if dim == 2:
        image2[rr, cc] = color
    if dim == 3:
        image2[:, rr,cc] = color
    return image2

def binary_closing(image, radius_selem=1):
    """
    Specific function to remove thin structures
    in the image.

    Performs a morphological closing.


    Parameters
    ----------
    image: np.ndarray
        image
    radius_selem: int
        radius in pixel for the structuring element

    Returns
    ----------
    np.ndarray
        morphologically closed image

    """
    otsu = threshold_otsu(image)
    binary = np.where(image > otsu, 0, 255)
    selem = disk(radius_selem)
    mask = closing(binary, selem)
    masked_image = np.ma.array(image, mask=mask)
    masked_image = masked_image.filled(0)
    return masked_image



def distances_closest_neighbour(points):
    """
    Distances between each point and its closest neighbour
    in a set of points.

    Parameters
    ----------
    points: np.ndarray
        points as (x-y) coordinates

    Returns
    ----------
    np.ndarray
        distances between each point and its closest neighbour


    """
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    # distances = dist.squareform(dist.pdist(points))
    distances = np.ma.masked_equal(distances, 0)
    min_dist = np.min(distances, axis=1)
    return min_dist


def average_distance_graph(image, threshold):
    """
    Average edge length in a graph.

    The graph is constructed by a binarization
    of an image with a given threshold.

    A node in the graph corresponds to a pixel
    above this threshold.

    Parameters
    ----------
    image: np.ndarray
        the image
    threshold: int
        threshold for binary image

    Returns
    ----------
    float
        average edge length in the graph

    """
    binary = np.where(image > threshold, 255, 0)
    indices = np.where(binary > 0)
    indices_array = np.asarray(indices).T
    if len(indices_array) > 0:
        distances = distances_closest_neighbour(indices_array)
        average_distance = np.mean(distances)
        return average_distance
    return 0

def spatial_chaos(image, quantiles=[]):
    """
    Spatial chaos measure

    See: Testing for Presence of Known and Unknown Molecules in Imaging Mass Spectrometry
    Alexandrov et al. (2013)

    Parameters
    ----------
    image: np.ndarray
        image
    quantiles: list
        list of quantile threshold values

    Returns
    ----------
    list
        list of spatial chaos values for each image

    """
    chaos_measures = []
    for i in range(image.shape[-1]):
        image_2D = image[..., i]
        norm_img = np.uint8(cv.normalize(image_2D, None, 0, 255, cv.NORM_MINMAX))
        edges = sobel(norm_img)
        if len(quantiles):
            min_distance = sys.maxsize
            for quantile in quantiles:
                threshold = np.percentile(edges, quantile)
                dist_edges = average_distance_graph(edges, threshold)
                if dist_edges < min_distance:
                    min_distance = dist_edges
            min_distance = sys.maxsize
            for quantile in quantiles:
                threshold = np.percentile(edges, quantile)
                dist_edges = average_distance_graph(edges, threshold)
                if dist_edges < min_distance:
                    min_distance = dist_edges
        else:
            try:
                threshold = threshold_otsu(edges)
                min_distance = average_distance_graph(edges, threshold)
            except:
                min_distance = 0
        chaos_measures.append(min_distance)
    return chaos_measures


def find_similar_images_spatial_chaos(img, threshold, quantiles, return_indices=False):
    """
    Finds images with spatial
    chaos values greater than a given threshold.

    Parameters
    ----------
    img: np.ndarray
        image
    threshold: int
        threshold for spatial chaos values
    quantiles: list
        list of quantile threshold values

    Returns
    ----------
    nd.ndarray
        images whose spatial chaos values are above threshold
    """
    chaos_measures = spatial_chaos(img, quantiles)
    chaos_array = np.array(chaos_measures)
    chaos_indices = np.where( (chaos_array >= 1) & (chaos_array < threshold))
    spatially_coherent = np.take(img, chaos_indices[0], axis=-1)
    to_return = (spatially_coherent,)
    if return_indices:
        to_return += (chaos_array, chaos_indices)
    return to_return

def spatial_coherence(image):
    """
    Spatial coherence of a binary image,
    that is to say the area of the largest
    connected component.

    Parameters
    ----------
    image: np.ndarray
        binarized image

    Returns
    ----------
    float
        spatial coherence

    """
    labels = measure.label(image, background=0)
    r = properties_largest_area_cc(labels)
    if r == -1:
        return -1
    else:
        return r.area


def median_perimeter(image):
    labels = measure.label(image, background=0)
    r = properties_median_perimeter(labels)
    if r == -1:
        return -1
    else:
        return r.perimeter

def find_similar_images_spatial_coherence(image_maldi, factor, quantiles=[], upper=100, fn=spatial_coherence, return_indices=False):
    """
    Finds images with spatial
    coherence values greater than a given threshold.

    Spatial coherence values are computed
    for several quantile thresholds. The minimum area
    over the thresholded images is kept.

    Parameters
    ----------
    image_maldi: np.ndarray
        MALDI image
    factor: int
        threshold for spatial coherence values
    quantiles: list
        quantile threshold values (list of integers)
    upper: int
        quantile upper threshold

    Returns
    ----------
    np.ndarray
        images whose spatial coherence values are above factor
    """
    values = []
    for i in range(image_maldi.shape[-1]):
        image2D = image_maldi[..., i]
        norm_img = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
        min_area = sys.maxsize
        upper_threshold = np.percentile(norm_img, upper)
        for quantile in quantiles:
            threshold = int(np.percentile(norm_img, quantile))
            mask = (norm_img > threshold) & (norm_img <= upper_threshold)
            sc = fn(mask)
            if sc < min_area:
                min_area = sc
        values.append(min_area)
    value_array = np.array(values)
    indices = (value_array > factor)
    similar_images = image_maldi[..., indices]
    if return_indices:
        return similar_images, indices
    return similar_images, indices

def find_similar_images_spatial_coherence_percentage(image_maldi, percentage, quantiles=[], upper=100, fn=spatial_coherence, return_indices=False):
    """
    Finds images with spatial
    coherence values greater than a threshold defined as a
    factor (percentage) multiplied by the maximum spatial
    coherence value.

    Spatial coherence values are computed
    for several quantile thresholds. The minimum area
    over the thresholded images is kept.

    Parameters
    ----------
    image_maldi: np.ndarray
        MALDI image
    percentage: float
        multiplicative factor for spatial coherence values
    quantiles: list
        quantile threshold values (list of integers)
    upper: int
        quantile upper threshold

    Returns
    ----------
    np.ndarray
        images whose spatial coherence values are above factor
    """
    values = []
    for i in range(image_maldi.shape[-1]):
        image2D = image_maldi[..., i]
        norm_img = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
        min_area = sys.maxsize
        upper_threshold = np.percentile(norm_img, upper)
        for quantile in quantiles:
            threshold = int(np.percentile(norm_img, quantile))
            mask = (norm_img > threshold) & (norm_img <= upper_threshold)
            sc = fn(mask)
            if sc < min_area:
                min_area = sc
        values.append(min_area)
    value_array = np.array(values)
    max_sc_value = np.amax(value_array)
    t = percentage * max_sc_value
    indices = (value_array > t)
    similar_images = image_maldi[..., indices]
    if return_indices:
        return similar_images, indices
    return similar_images

def median_minima(maxima, minima):
    groups = [ [] for i in range(len(maxima)+1) ]
    for i, minimum in enumerate(minima):
        added = False
        for j, maximum in enumerate(maxima):
            if minimum < maximum:
                groups[j].append(minimum)
                added = True
                break
        if not added:
            groups[-1].append(minimum)
    medians = [np.median(g) for g in groups]
    return np.array(medians, dtype=int)

def find_similar_images_variance(image_maldi, factor_variance=0.1, return_indices=False):
    """
    Finds images that have a high variance in their intensities.

    Selects images according to a factor of max variance.

    Parameters
    ----------
    image_maldi: np.ndarray
        input image
    factor_variance: int
        factor by which max variance is multiplied to
        determine a threshold above which images are selected

    Returns
    ----------
    np.ndarray
        array of high variability images

    """
    reshaped = image_maldi.reshape(-1, image_maldi.shape[-1])
    variance = np.var(reshaped, axis=0)
    max_variance = np.amax(variance)
    indices = variance < factor_variance * max_variance
    similar_images = image_maldi[..., indices]
    if return_indices:
        return similar_images, indices
    return similar_images


def find_associated_distance_transforms(image_maldi, masks, quantiles, add_otsu_thresholds=True):
    dt_masks = []
    for mask in masks:
        threshold = threshold_otsu(mask)
        mask_invert = 255-np.where(mask>threshold, 255, 0)
        dt_mask = distance_transform_edt(mask_invert)
        dt_masks.append(dt_mask)

    dt_ions = []
    for i in range(image_maldi.shape[-1]):
        image2D = image_maldi[..., i]
        norm_img = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
        upper_threshold = np.percentile(norm_img, 100)
        if add_otsu_thresholds:
            otsu_thresholds = threshold_multiotsu(norm_img)
        else:
            otsu_thresholds = []
        quantile_thresholds = [int(np.percentile(norm_img, quantile)) for quantile in quantiles]
        thresholds = np.concatenate((quantile_thresholds, otsu_thresholds))
        distances = []
        images_invert = []
        indices_distances_masks = []
        for threshold in thresholds:
            condition = (norm_img > threshold) & (norm_img <= upper_threshold)
            image_binary = np.where(condition, 255, 0)
            image_invert = 255 - image_binary
            dt_cleaned = distance_transform_edt(image_invert)
            min_d = []
            distances_masks = []
            for j, mask in enumerate(masks):
                d_mask_cleaned = dt_cleaned[masks[j] > 0]
                dist = max(d_mask_cleaned)
                distances_masks.append(dist)
            index_masks = np.argmin(distances_masks)
            distances.append(distances_masks[index_masks])
            images_invert.append(image_invert)
            indices_distances_masks.append(index_masks)
        index_distance = np.argmax(distances)
        best_im_invert = images_invert[index_distance]
        dt_best = distance_transform_edt(best_im_invert)
        dt_ions.append(dt_best)
        #     fig, ax = plt.subplots(2, 3)
        #     ax[0, 0].imshow(masks[indices_distances_masks[index_distance]])
        #     ax[0, 1].imshow(255-best_im_invert)
        #     ax[0, 2].imshow(image2D)
        #     ax[1, 0].imshow(dt_masks[indices_distances_masks[index_distance]])
        #     # ax[1, 1].imshow(dt)
        #     plt.show()
    return dt_masks, dt_ions

def find_similar_image_distance_map_percentile(image_maldi, masks, factor, quantiles=[], add_otsu_thresholds=True, reverse=False, is_mean=False, return_indices=False, return_distances=False):
    values = []
    # dt_masks = []
    # for mask in masks:
    #     threshold = threshold_otsu(mask)
    #     mask_invert = 255-np.where(mask>threshold, 255, 0)
    #     dt_mask = distance_transform_edt(mask_invert)
    #     dt_masks.append(dt_mask)

    dt_masks, dt_ions = find_associated_distance_transforms(image_maldi, masks, quantiles, add_otsu_thresholds)

    if reverse:
        dt_ions, dt_masks = dt_masks, dt_ions

    all_distances = []
    for i, dt_ion in enumerate(dt_ions):
        distances = []
        for j, dt_mask in enumerate(dt_masks):
            d_mask_cleaned = dt_ion[dt_mask == 0]
            d_ion = dt_mask[dt_ion == 0]
            dist = np.percentile(d_mask_cleaned, 95)
            dist_mask = np.percentile(d_ion, 95)
            if is_mean:
                dist_both = np.mean([dist, dist_mask])
            else:
                dist_both = max([dist, dist_mask])
            distances.append(dist_both)
        all_distances.append(distances)
        index_distance = np.argmin(distances)
        min_distance = distances[index_distance]
        values.append(min_distance)
        # fig, ax = plt.subplots(2, 2)
        # print(min_distance)
        # if reverse:
        #     ax[0, 0].imshow(image_maldi[..., index_distance].T)
        #     ax[0, 1].imshow(masks[i].T)
        # else:
        #     ax[0, 0].imshow(image_maldi[..., i].T)
        #     ax[0, 1].imshow(masks[index_distance].T)
        # ax[1, 0].imshow(dt_ion.T)
        # ax[1, 1].imshow(dt_masks[index_distance].T)
        # # ax[1, 1].imshow(dt)
        # plt.show()
    value_array = np.array(values)
    all_distances = np.array(all_distances)
    indices = (value_array < factor)
    if reverse:
        similar_images = np.array(masks)[indices]
    else:
        similar_images = image_maldi[..., indices]
    to_return = (similar_images,)
    if return_indices:
        to_return += (value_array, indices)
    if return_distances:
        to_return += (all_distances,)
    return to_return



def find_similar_image_distance_map_cc(image_maldi, masks, factor, quantiles=[], add_otsu_thresholds=True, return_indices=False, reverse=False):
    values = []
    dt_masks = []
    for mask in masks:
        threshold = threshold_otsu(mask)
        mask_invert = 255-np.where(mask>threshold, 255, 0)
        dt_mask = distance_transform_edt(mask_invert)
        dt_masks.append(dt_mask)

    for i in range(image_maldi.shape[-1]):
        image2D = image_maldi[..., i]
        norm_img = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
        upper_threshold = np.percentile(norm_img, 100)
        distances = []
        distances_both = []
        if add_otsu_thresholds:
            otsu_thresholds = threshold_multiotsu(norm_img)
        else:
            otsu_thresholds = []
        quantile_thresholds = [int(np.percentile(norm_img, quantile)) for quantile in quantiles]
        thresholds = np.concatenate((quantile_thresholds, otsu_thresholds))
        print(thresholds)
        indices_distances_masks = []
        images_invert = []
        for threshold in thresholds:

            condition = (norm_img > threshold) & (norm_img <= upper_threshold)
            image_binary = np.where(condition, 255, 0)
            image_invert = 255 - image_binary
            image_invert_clean = binary_fill_holes(image_binary)*255
            image_invert_clean = remove_small_objects(image_invert_clean>0, min_size=10, connectivity=2).astype(int)*255
            image_invert_clean = 255-image_invert_clean
            dt_cleaned = distance_transform_edt(image_invert)
            min_d = []
            distances_masks = []
            for j, mask in enumerate(masks):
                d_mask_cleaned = dt_cleaned[masks[j] > 0]
                d_ion = dt_masks[j][image_binary > 0]
                dist = max(d_mask_cleaned)
                dist_mask = max(d_ion)
                images_invert.append(image_binary)
                distances_masks.append(dist)
            index_masks = np.argmin(distances_masks)
            distances.append(distances_masks[index_masks])
            indices_distances_masks.append(index_masks)
        index_distance = np.argmax(distances)
        print(distances[index_distance])
        best_im_invert = images_invert[indices_distances_masks[index_distance]]
        dt_best = distance_transform_edt(best_im_invert)
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(masks[indices_distances_masks[index_distance]])
        ax[0, 1].imshow(255-best_im_invert)
        ax[0, 2].imshow(image2D)
        ax[1, 0].imshow(dt_masks[indices_distances_masks[index_distance]])
        # ax[1, 1].imshow(dt)
        plt.show()
        min_distance = distances[index_distance]
        values.append(min_distance)
    value_array = np.array(values)
    indices = (value_array < factor)
    similar_images = image_maldi[..., indices]
    to_return = (similar_images,)
    if return_indices:
        to_return += (value_array, indices)
    return to_return




def extract_peaks_from_distribution(min_hist, bins, threshold):
    peaks = signal.find_peaks(min_hist)[0]
    low_peaks = signal.argrelextrema(min_hist, np.less_equal)[0]
    if low_peaks[0] > peaks[0]:
        low_peaks = np.insert(low_peaks, 0, 0)
    if low_peaks[-1] < peaks[-1]:
        low_peaks = np.append(low_peaks, len(min_hist)-1)
    low_peaks = median_minima(peaks, low_peaks)
    min_int = min_hist[low_peaks]
    low_peaks_height = np.mean([min_int, np.roll(min_int, -1)], axis=0)[:-1]
    diff_height = min_hist[peaks] - low_peaks_height
    peaks = peaks[diff_height > threshold]
    # widths = signal.peak_widths(min_hist, peaks)[0]
    # x = np.arange(min_hist.shape[0])
    # fig, ax = plt.subplots(1, 2)
    # ax[1].bar(x, min_hist, width=1)
    # ax[1].plot(x[peaks], min_hist[peaks], "ro")
    # plt.show()
    return peaks, low_peaks

    # x = np.arange(min_hist.shape[0])


def distance_distribution(image, centroid, bins):
    diff = np.linalg.norm(np.argwhere(image) - centroid, axis=-1)
    hist, _ = np.histogram(diff, bins=bins)
    # hist = signal.savgol_filter(hist, 2, 1)
    return hist

def generate_random_distributions(image, centroid, quantiles, bins):
    random_distribs = []
    for q in quantiles:
        noisy_image = np.random.normal(1, 0.5, image.shape)
        t = np.percentile(noisy_image, q)
        noisy_image[noisy_image < t] = 0
        th_hist = distance_distribution(noisy_image, centroid, bins)
        random_distribs.append(th_hist)
    return random_distribs

def quantile_distance_distributions(image_maldi, quantiles=[], w=10):
    th_image = image_maldi[..., 0].copy()
    th_image[:] = 1
    width, height = th_image.shape
    centroid = [width//2, height//2]
    th_diff = np.linalg.norm(np.argwhere(th_image) - centroid, axis=-1)
    bins = int(max(centroid))//w
    distribs = generate_random_distributions(th_image, centroid, quantiles, bins)
    distributions = []
    for i in range(image_maldi.shape[-1]):
        image2D = image_maldi[..., i]
        image2D = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
        upper_threshold = np.percentile(image2D, 100)
        min_value = sys.maxsize
        min_distrib = []
        for ind_quantile, quantile in enumerate(quantiles):
            threshold = int(np.percentile(image2D, quantile))
            mask = (image2D > threshold) & (image2D <= upper_threshold)
            binaryimg = image2D.copy()
            binaryimg[mask] = 1
            binaryimg[~mask] = 0

            moments = measure.moments(binaryimg, order=1)
            number_non_zero = np.count_nonzero(binaryimg)
            if (number_non_zero == 0):
                continue
            centroid = [moments[1, 0]/moments[0, 0], moments[0, 1]/moments[0, 0]]
            min_hist = distance_distribution(mask, centroid, bins)
            th_distrib = distribs[ind_quantile]
            correlation = pearsonr(th_distrib.flatten(), min_hist.flatten()).statistic
            if correlation < min_value:
                min_value = correlation
                min_distrib = min_hist
        if min_value == sys.maxsize:
            min_value = 0
            min_value_sample = 0
        distributions.append(min_distrib)
    distributions = np.array(distributions)
    return distributions

def find_similar_images_distance_map(image_maldi, mzs, factor, quantiles=[], in_sample=False, return_indices=False, return_thresholds=False, normalize_max=False, size_elem=5):
    th_image = image_maldi[..., 0].copy()
    width, height = th_image.shape
    centroid = [width//2, height//2]
    values = []
    values_sample = []
    best_thresholds = []
    for i in range(image_maldi.shape[-1]):
        # if mzs[i] < 871.55 or mzs[i] > 871.58:
        #     continue
        image2D = image_maldi[..., i]
        norm_img = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
        upper_threshold = np.percentile(norm_img, 100)
        thresholds = [int(np.percentile(norm_img, quantile)) for quantile in sorted(quantiles)]
        images_invert = []
        indices_distances_masks = []
        min_value = 0
        min_value_sample = 0
        if normalize_max:
            min_value = sys.maxsize
            min_value_sample = sys.maxsize
        best_threshold = 0
        value_max = 0
        for ind_threshold, threshold in enumerate(thresholds):
            condition = (norm_img > threshold) & (norm_img <= upper_threshold)
            binaryimg = np.where(condition, 1, 0)
            number_non_zero = np.count_nonzero(binaryimg)
            if number_non_zero == 0:
                continue
            image_binary = np.where(condition, 255, 0)
            dt_cleaned = distance_transform_edt(image_binary)
            n2 = np.count_nonzero(dt_cleaned)
            dt_cleaned[dt_cleaned > 0] -= 1
            n1 = np.count_nonzero(dt_cleaned)
            n3 = n2 - n1
            dist = ((n1 - n3) * np.amax(dt_cleaned)) / ((n1 + n3) * np.sum(dt_cleaned) / n1)
            divisor = np.count_nonzero(dt_cleaned)
            if normalize_max:
                divisor *= np.amax(dt_cleaned)
            if divisor == 0:
                divisor = 1
            dist = np.sum(dt_cleaned) / divisor
            if normalize_max:
                dist = 1-dist
            ind = np.argwhere(condition)
            diff = np.linalg.norm(ind - centroid, axis=-1)
            value_sample = np.amin(diff)
            if (dist < min_value and normalize_max) or (dist > min_value and not normalize_max):
                min_value = dist
                min_value_sample = value_sample
                best_threshold = quantiles[ind_threshold]
        if (min_value == sys.maxsize and normalize_max) or (min_value == 0 and not normalize_max):
            min_value = 0
            min_value_sample = 0
            best_threshold = quantiles[ind_threshold]
        values.append(min_value)
        values_sample.append(min_value_sample)
        best_thresholds.append(best_threshold)
    value_array = np.array(values)
    value_sample_array = np.array(values_sample)
    best_thresholds = np.array(best_thresholds)
    if in_sample:
        off_sample_image, off_sample_cond = determine_on_off_sample(image_maldi, value_sample_array, size_elem)
        # off_sample_cond = np.array([np.median(off_sample_image[coord.T[0], coord.T[1]]) for coord in coords])
    indices = (value_array > factor) & (off_sample_cond < 0.1)
    similar_images = image_maldi[..., indices]
    to_return = (similar_images,)
    if return_indices:
        to_return += (value_array, indices)
    if in_sample:
        to_return += (off_sample_image, off_sample_cond)
    if return_thresholds:
        to_return += (best_thresholds,)
    print(best_thresholds)
    return to_return

def find_similar_images_dispersion_peaks(image_maldi, factor, quantiles=[], in_sample=False, return_indices=False, return_thresholds=False, size_elem=5):
    values = []
    values_sample = []
    coords = []
    th_image = image_maldi[..., 0].copy()
    th_image[:] = 1
    width, height = th_image.shape
    centroid = [width//2, height//2]
    w = 10
    bins = int(max(centroid))//w
    distribs = generate_random_distributions(th_image, centroid, quantiles, bins)
    thresholds = []
    for i in range(image_maldi.shape[-1]):
        image2D = image_maldi[..., i]
        image2D = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
        upper_threshold = np.percentile(image2D, 100)
        min_value = sys.maxsize
        min_value_sample = sys.maxsize
        min_distrib = []
        c = []
        best_threshold = 0
        for ind_quantile, quantile in enumerate(quantiles):
            threshold = int(np.percentile(image2D, quantile))
            mask = (image2D > threshold) & (image2D <= upper_threshold)
            binaryimg = image2D.copy()
            binaryimg[mask] = 1
            binaryimg[~mask] = 0

            moments = measure.moments(binaryimg, order=1)
            number_non_zero = np.count_nonzero(binaryimg)
            if (number_non_zero == 0):
                continue
            centroid = [moments[1, 0]/moments[0, 0], moments[0, 1]/moments[0, 0]]
            min_hist = distance_distribution(mask, centroid, bins)
            ind = np.argwhere(mask)
            diff = np.linalg.norm(ind - centroid, axis=-1)
            min_hist = distance_distribution(mask, centroid, bins)
            th_distrib = distribs[ind_quantile]
            correlation = pearsonr(th_distrib.flatten(), min_hist.flatten()).statistic
            value_sample = np.amin(diff)
            if correlation < min_value:
                min_value = correlation
                min_value_sample = value_sample
                min_distrib = min_hist
                c = ind
                best_threshold = quantiles[ind_quantile]
        if min_value == sys.maxsize:
            min_value = 0
            min_value_sample = 0
        values.append(min_value)
        values_sample.append(min_value_sample)
        coords.append(c)
        thresholds.append(best_threshold)
    value_array = np.array(values)
    value_sample_array = np.array(values_sample)
    coords = np.array(coords)
    thresholds = np.array(thresholds)
    if in_sample:
        off_sample_image, off_sample_cond = determine_on_off_sample(image_maldi, value_sample_array, size_elem)
        # off_sample_cond = np.array([np.median(off_sample_image[coord.T[0], coord.T[1]]) for coord in coords])
    indices = (value_array < factor) & (off_sample_cond < 0.1)
    similar_images = image_maldi[..., indices]
    to_return = (similar_images,)
    if return_indices:
        to_return += (value_array, indices)
    if in_sample:
        to_return += (off_sample_image, off_sample_cond)
    if return_thresholds:
        to_return += (thresholds,)
    return to_return


def find_similar_images_dispersion(image_maldi, factor, quantiles=[], in_sample=False, return_indices=False):
    values = []
    values_sample = []
    coords = []
    th_image = image_maldi[..., 0].copy()
    centroid = [th_image.shape[0]//2, th_image.shape[1]//2]
    th_image[:] = 1
    th_diff = np.linalg.norm(np.argwhere(th_image) - centroid, axis=-1)
    th_std = np.std(th_diff)
    w = 10
    bins = int(max(centroid))//w
    th_hist, bins = np.histogram(th_diff, bins=bins)
    for i in range(image_maldi.shape[-1]):
        image2D = image_maldi[..., i]
        image2D = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
        upper_threshold = np.percentile(image2D, 100)
        min_value = sys.maxsize
        min_value_sample = sys.maxsize
        min_distrib = []
        c = []
        for quantile in quantiles:
            threshold = int(np.percentile(image2D, quantile))
            mask = (image2D > threshold) & (image2D <= upper_threshold)
            binaryimg = image2D.copy()
            binaryimg[mask] = 1
            binaryimg[~mask] = 0

            moments = measure.moments(binaryimg, order=1)
            number_non_zero = np.count_nonzero(binaryimg)
            if (number_non_zero == 0):
                continue
            # centroid = [moments[1, 0]/moments[0, 0], moments[0, 1]/moments[0, 0]]
            ind = np.argwhere(mask)
            diff = np.linalg.norm(ind - centroid, axis=-1)
            variance = np.std(diff) / th_std
            value_sample = np.amin(diff)
            if variance < min_value:
                min_value = variance
                min_value_sample = value_sample
                min_distrib = diff
                c = ind
        if min_value == sys.maxsize:
            min_value = 0
            min_value_sample = 0

        values.append(min_value)
        values_sample.append(min_value_sample)
        coords.append(c)
    value_array = np.array(values)
    value_sample_array = np.array(values_sample)
    coords = np.array(coords)
    if in_sample:
        off_sample_image, off_sample_cond = determine_on_off_sample(image_maldi, value_sample_array)
        # off_sample_cond = np.array([np.median(off_sample_image[coord.T[0], coord.T[1]]) for coord in coords])
    indices = (value_array < factor) & (off_sample_cond < 0.5)
    similar_images = image_maldi[..., indices]
    to_return = (similar_images,)
    if return_indices:
        to_return += (value_array, indices)
    if in_sample:
        to_return += (off_sample_image, off_sample_cond)
    return to_return

def determine_on_off_sample(image_maldi, value_array, size_elem=1):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(value_array.reshape(-1, 1))
    labels = kmeans.labels_
    cond = np.mean(value_array[labels == 0]) > np.mean(value_array[labels == 1])
    im = io.normalize(image_maldi)
    number_cluster = 0 if cond else 1
    off_sample = np.zeros_like(im[..., 0])
    sub_image = image_maldi[..., labels==number_cluster]
    for i in range(sub_image.shape[-1]):
        current_sub = sub_image[..., i]
        thresh = threshold_otsu(current_sub)
        off_sample[current_sub > thresh] += 1
    thresh = threshold_otsu(off_sample)
    off_sample = np.where(off_sample > thresh, 1, 0)
    off_sample = closing(off_sample, disk(size_elem))
    off_sample_cond = []
    for i in range(image_maldi.shape[-1]):
        im = image_maldi[..., i]
        thresh = threshold_otsu(im)
        coord = np.argwhere(im > thresh)
        mean = np.mean(off_sample[coord.T[0], coord.T[1]])
        off_sample_cond.append(mean)
    off_sample_cond = np.array(off_sample_cond)
    return off_sample, off_sample_cond


def heterogeneity_mask(image, region, size=10):
    reduced_image = np.where(region > 0, image, -1)
    averages = []
    max_values = []
    for x in range(0, image.shape[0], size):
        for y in range(0, image.shape[1], size):
            currimg = reduced_image[x:x+size, y:y+size]
            origimg = image[x:x+size, y:y+size]
            origimg = origimg[origimg > 0]
            currimg = currimg[currimg >= 0]
            if currimg.size > 0 and origimg.size > 0:
                averages.append(np.median(origimg))
                max_values.append(origimg.max())
    average = np.median(averages)
    average /= np.median(max_values)/2
    return average, averages, max_values


def mapping_neighbors_average(image, radius):
    r = radius
    size = 2*r+1
    img_padded = np.pad(image, (r,r), 'constant')
    mapping_matrix = np.zeros_like(image)
    for index in np.ndindex(image.shape[:-1]):
        i, j = index
        neighbors = image[i-r:i+r+1, j-r:j+r+1]
        if neighbors.shape[0] != size or neighbors.shape[1] != size:
            continue
        neighbors = neighbors.reshape((size**2, neighbors.shape[-1]))
        mapping_matrix[index] = np.mean(neighbors, axis=0)
    return mapping_matrix

def clustering_with_centers(images, centers, is_subtract, metric, mean_spectra_matrix=None, radius=0):
    images = mapping_neighbors_average(images, radius=radius)
    image_flatten = fusion.flatten(images, is_spectral=True).T
    if is_subtract:
        for i, spectra in enumerate(image_flatten):
            image_flatten[i, :] = sp.subtract_spectra(spectra, mean_spectra_matrix)
    distances = dist.cdist(image_flatten.astype(float), centers.astype(float), metric=metric)
    distances = np.nan_to_num(distances, nan=1.0)
    labels = np.argmin(distances, axis=-1)
    norm_distance = np.take_along_axis(distances, labels[:, None], axis=-1)
    confidence = np.reshape(norm_distance, images.shape[:-1])
    label_image = np.reshape(labels, images.shape[:-1])
    return label_image, confidence
