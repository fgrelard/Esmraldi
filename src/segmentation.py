import pyimzml.ImzMLParser as imzmlparser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.measure import find_contours
from skimage import measure
from skimage.filters import threshold_otsu, rank
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
        norm_img[..., 0] = np.uint8(cv.normalize(image_maldi, None, 0, 255, cv.NORM_MINMAX).flatten())

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


def find_similar_images_variance(image_maldi, threshold_variance=0):
    reshaped = image_maldi.reshape(-1, image_maldi.shape[-1])
    variance = np.var(reshaped, axis=0)
    max_variance = np.amax(variance)
    similar_images = image_maldi[..., variance > 0.1 * max_variance]
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
    max_area = 0
    index = -1
    for i in range(nb_class):
        similar_images = image_maldi[..., y_kmeans==i]
        av_area = average_area(similar_images)
        if av_area > max_area:
            index = i
            max_area = av_area
    return index
