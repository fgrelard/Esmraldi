import pyimzml.ImzMLParser as imzmlparser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.measure import find_contours
from skimage import measure
import cv2 as cv
import SimpleITK as sitk

def max_variance_sort(image_maldi):
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

def normalize(image_maldi):
    x = image_maldi.shape[0]
    y = image_maldi.shape[1]
    z = image_maldi.shape[2]
    norm_img = np.zeros(shape=(x*y, z))
    for index in np.ndindex(image_maldi.shape[2:]):
        current_index = (slice(None), slice(None)) + (index,)
        norm_slice = cv.normalize(image_maldi[current_index], None, 0.0, 1.0, cv.NORM_MINMAX)
        norm_img[..., index[0]] = norm_slice.flatten()

    norm_img = norm_img.transpose()
    return norm_img

def properties_largest_area_cc(ccs):
    regionprops = measure.regionprops(ccs)
    areas = lambda r: r.area
    argmax = max(regionprops, key=areas)
    return argmax

def region_property_to_cc(ccs, regionprop):
    label = regionprop.label
    cc = np.where(ccs == label, 0, 255)
    return cc

def region_growing(images, seedList, lower_threshold):
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
        largest_cc = region_property_to_cc(labels, regionprop)
        seeds = seeds.union(set(((int(coord[0]), int(coord[1])) for coord in regionprop.coords)))
    return list(seeds)

def find_similar_images(image_maldi):
    norm = normalize(image_maldi)
    pca = PCA(n_components=5)
    X_r = pca.fit(norm).transform(norm)
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X_r)
    y_kmeans = kmeans.predict(X_r)
    similar_images = image_maldi[..., y_kmeans==0]
    return similar_images

def extraction_roi_contour(image_maldi):
    mean = np.average(image_maldi, axis=2)
    binary = np.zeros_like(mean)
    binary[mean > 0] = 255.0
    edges = find_contours(binary, 0.0)[0]
    return edges


def segmentation(image_maldi):
    max_variance_sort(image_maldi)
