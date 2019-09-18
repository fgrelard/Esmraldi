import numpy as np
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def clustering_affinity(X_r):
    af = AffinityPropagation(preference=-50).fit(X_r)
    return af

def clustering_kmeans(X_r):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_r)
    return kmeans

def pca(image):
#    pca = TSNE(n_components=2)
    pca = PCA(n_components=5)
    fit_pca = pca.fit(image)
    return fit_pca

def post_processing(pca_maldi, pca_mri):
    size_train = pca_maldi.shape[0]
    X = np.vstack((pca_maldi,pca_mri))
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform( X )
    X_train_tsne = X_tsne[0:size_train,:]
    X_test_tsne  = X_tsne[size_train:,:]
    return X_train_tsne, X_test_tsne

def weighted_distance(X, weights):
    return np.sqrt(np.sum(X**2 * weights))

def select_images(images, mzs, point_mri, centers, weights, labels, top=1):
    distances = np.array([weighted_distance(center-point_mri, weights) for center in centers])
    indices = [i for i in range(len(distances))]
    indices.sort(key=lambda x: distances[x])
    if top is None:
        similar_images = images[..., indices].T
        similar_mzs = mzs[indices]
        distances = distances[indices]
    else:
        indices = np.array(indices)
        condition = np.any(np.array([labels == indices[i] for i in range(top)]), axis=0)
        similar_images = images[..., condition].T
        similar_mzs = mzs[condition]
    return np.uint8(similar_images), similar_mzs, distances

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

            divided2 = np.uint8(cv.normalize(divided, None, 0, 255, cv.NORM_MINMAX))
            fig, ax = plt.subplots(1, 4)

            ratio_images[..., c] = divided
            current_ratio = mzs[i] + "/" + mzs[j]
            new_mzs[c] = current_ratio
            c += 1
    return ratio_images, new_mzs
