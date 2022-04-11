"""
Module for the segmentation
"""
import sys
import math
import numpy as np
import pyimzml.ImzMLParser as imzmlparser
import scipy.spatial.distance as dist
import cv2 as cv
import SimpleITK as sitk

from sklearn.decomposition import PCA
from sklearn.feature_extraction import grid_to_graph
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage.measure import find_contours
from skimage import measure
from skimage.filters import threshold_otsu, rank, sobel
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import data, color
from skimage.draw import disk as drawdisk
from skimage.morphology import binary_erosion, closing, disk

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


def find_similar_images_spatial_chaos(img, threshold, quantiles):
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
    chaos_indices = np.where( (chaos_array > 1) & (chaos_array < threshold))
    spatially_coherent = np.take(img, chaos_indices[0], axis=-1)
    return spatially_coherent

def spatial_coherence(image):
    """
    Spatial coherence of a binary image,
    that is to say the area of the largest
    connected component.

    Parameters
    ----------
    image: np.ndarrau
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

def find_similar_images_variance(image_maldi, factor_variance=0.1):
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
    similar_images = image_maldi[..., variance < factor_variance * max_variance]
    return similar_images
