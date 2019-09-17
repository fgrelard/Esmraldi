import numpy as np
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import TSNE

def clustering(image, X_r):
    af = AffinityPropagation(preference=-50).fit(X_r)
    return af

def pca(image):
    pca = TSNE(n_components=2)
#    pca = PCA(n_components=5)
    fit_pca = pca.fit(image)
    return fit_pca

def weighted_distance(X, weights):
    return np.sqrt(np.sum(X**2 * weights))

def select_images(fit_pca, images, mzs, mri_norm, centers, weights, labels, top=1):
    point = fit_pca.transform(mri_norm)
    distances = np.array([weighted_distance(center-point, weights) for center in centers])
    indices = [i for i in range(len(distances))]
    indices.sort(key=lambda x: distances[x])
    if top is None:
        similar_images = images[..., indices].T
        similar_mzs = mzs[indices]
    else:
        indices = np.array(indices)
        condition = np.any(np.array([labels == indices[i] for i in range(top)]), axis=0)
        similar_images = images[..., condition].T
        similar_mzs = mzs[condition]
    return np.uint8(similar_images), similar_mzs

def extract_ratio_images(image, mzs):
    z = image.shape[-1]
    c = 0
    new_mzs = np.zeros(((z**2-z)//2,), dtype='object')
    ratio_images = np.zeros((image.shape[0], image.shape[1], (z**2-z)//2), dtype=np.uint8)
    for i in range(z-1, 0, -1):
        for j in range(i):
            first_image = image[..., i]
            second_image = image[..., j]
            divided = np.zeros_like(first_image, dtype=np.float64)
            np.divide(first_image, second_image, out=divided, where=second_image!=0)
            divided = np.uint8(cv.normalize(divided, None, 0, 255, cv.NORM_MINMAX))
            ratio_images[..., c] = divided
            current_ratio = mzs[i] + "/" + mzs[j]
            new_mzs[c] = current_ratio
            c += 1
    return ratio_images, new_mzs
