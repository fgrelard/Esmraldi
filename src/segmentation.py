import pyimzml.ImzMLParser as imzmlparser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.measure import find_contours
import cv2 as cv

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

def find_similar_images(image_maldi):
    norm = normalize(image_maldi)
    average_img =np.average(norm, axis=1)
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
