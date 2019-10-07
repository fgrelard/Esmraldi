import pyimzml.ImzMLParser as imzmlparser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.measure import find_contours
from skimage import measure
from skimage.filters import threshold_otsu, rank
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import data, color
from skimage.draw import circle
from skimage.morphology import binary_erosion, closing, disk
import cv2 as cv
import SimpleITK as sitk

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
    for elem in image_list:
        print(elem["mz"])
        plt.imshow(elem["im"], cmap='jet').set_interpolation('nearest')
        plt.show()

def preprocess_pca(image_maldi):
    """
    Preprocess for PCA : normalizes and flattens
    a stack image with OpenCV

    Parameters
    ----------
    image_maldi: numpy.ndarray
        input image

    Returns
    ----------
    numpy.ndarray
        normalized image

    """
    x = image_maldi.shape[0]
    y = image_maldi.shape[1]
    z = image_maldi.shape[2] if len(image_maldi.shape) > 2 else 0

    if z > 0:
        norm_img = np.zeros(shape=(x*y,z), dtype=np.uint8)
        for index in np.ndindex(image_maldi.shape[2:]):
            current_index = (slice(None), slice(None)) + (index,)
            norm_slice = np.uint8(cv.normalize(image_maldi[current_index], None, 0, 255, cv.NORM_MINMAX))
            norm_img[..., index[0]] = norm_slice.flatten()
    else:
        norm_img = np.zeros(shape=(x*y, 1), dtype=np.uint8)
        norm_img[..., 0] = np.uint8(cv.normalize(image_maldi, None, 0, 255, cv.NORM_MINMAX)).flatten()

    norm_img = norm_img.transpose()
    return norm_img

def properties_largest_area_cc(ccs):
    """
    Extracts the connected component
    with the largest area

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

def region_property_to_cc(ccs, regionprop):
    """
    Extracts the connected component associated
    with the region

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

def region_growing(images, seedList, lower_threshold):
    """
    Region growing in an image stack
    with ITK
    All the images in the stack are processed sequentially
    and the seeds at step n depends on the segmentation
    by region growing at step n-1

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
    print(images.shape)
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
    return list(seeds)


def find_similar_images_variance(image_maldi, factor_variance=0.1):
    """
    Find images that have a high variance in their intensities
    Selects images according to a factor of max variance

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
    similar_images = image_maldi[..., variance > factor_variance * max_variance]
    return similar_images


def find_similar_images(image_maldi):
    """
    Performs a PCA to group similar images based on
    their intensities

    Selects the first cluster

    Parameters
    ----------
    image_maldi: numpy.ndarray
        image stack

    Returns
    ----------
    numpy.ndarray
        trimmed stack with images of high similarity
    """
    nb_class = 2
    norm = preprocess_pca(image_maldi)
    pca = PCA(n_components=5)
    X_r = pca.fit(norm).transform(norm)
    kmeans = KMeans(n_clusters=nb_class, random_state=0)
    kmeans.fit(X_r)
    y_kmeans = kmeans.predict(X_r)
    index = select_class_max_value(image_maldi, y_kmeans, nb_class)
    similar_images = image_maldi[..., y_kmeans==index]
    return similar_images


def average_area(images):
    """
    Average area of largest CCs on a collection of images

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
    Chooses label associated with kmeans cluster with
    images with highest average intensity

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
    area of largest CCs

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
    image2 = np.copy(image)
    dim = len(image2.shape)
    rr, cc = circle(int(center_y), int(center_x), int(radius), image2.shape[dim-2:])
    if dim == 2:
        image2[rr, cc] = 0
    if dim == 3:
        image2[:, rr,cc] = 0
    return image2

def remove_pericarp(image, radius_selem=1):
    otsu = threshold_otsu(image)
    binary = np.where(image > otsu, 0, 255)
    selem = disk(radius_selem)
    mask = closing(binary, selem)
    masked_image = np.ma.array(image, mask=mask)
    masked_image = masked_image.filled(0)
    return masked_image


def resize(image, size):
    dim = len(image.GetSize())
    new_dims = [size for i in range(2)]
    spacing = [image.GetSize()[0]/size for i in range(2)]
    if dim == 3:
        new_dims.append(image.GetSize()[2])
        spacing.append(1)
    resampled_img = sitk.Resample(image,
                                  new_dims,
                                  sitk.Transform(),
                                  sitk.sitkNearestNeighbor,
                                  image.GetOrigin(),
                                  spacing,
                                  image.GetDirection(),
                                  0.0,
                                  image.GetPixelID())
    return resampled_img
