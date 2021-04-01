"""
Module for the joint statistical
analysis of images
"""

import esmraldi.imzmlio as imzmlio
import numpy as np
import cv2 as cv
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


def clustering_affinity(X_r):
    """
    Clustering by affinity propagation

    Based on scikit module

    Parameters
    ----------
    X_r: np.ndarray
        fitted data (e.g. by PCA)

    Returns
    ----------
    np.ndarray
        clustered data

    """
    af = AffinityPropagation(preference=-50).fit(X_r)
    return af

def clustering_kmeans(X_r):
    """
    Clustering by kmeans

    Based on scikit module

    Parameters
    ----------
    X_r: np.ndarray
        fitted data (e.g. by PCA)

    Returns
    ----------
    np.ndarray
        clustered data

    """
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_r)
    return kmeans

def flatten(image_maldi, is_spectral=False):
    """
    Preprocess for reduction : flattens
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
    shape = image_maldi.shape

    if is_spectral:
        flatten_first_dims = np.prod(shape[:-1])
        norm_img = np.zeros(shape=(flatten_first_dims, shape[-1]), dtype=image_maldi.dtype)
        for index in range(image_maldi.shape[-1]):
            norm_img[..., index] = image_maldi[..., index].flatten()
    else:
        flatten_first_dims = np.prod(shape)
        norm_img = np.zeros(shape=(flatten_first_dims, 1), dtype=image_maldi.dtype)
        norm_img[..., 0] = image_maldi.flatten()

    norm_img = norm_img.transpose()
    return norm_img


def pca(image, n=5):
    """
    Performs PCA on image array.

    Each image is represented as a point after fitting.

    Based on scikit module

    Parameters
    ----------
    image: np.ndarray
        collection of images
    n: int
        number of components

    Returns
    ----------
    sklearn.decomposition.PCA
        pca object
    """
    pca = PCA(n_components=n)
    fit_pca = pca.fit(image)
    return fit_pca

def nmf(image, n=5):
    """
    Performs NMF on image array.

    Each image is represented as a point after fitting.

    Based on scikit module

    Parameters
    ----------
    image: np.ndarray
        collection of images
    n: int
        number of components

    Returns
    ----------
    sklearn.decomposition.NMF
        nmf object
    """
    nmf_obj = NMF(n_components=n, init='nndsvda', solver='mu', random_state=0, beta_loss="kullback-leibler")
    fit_nmf = nmf_obj.fit(image)
    return fit_nmf


def post_processing(pca_maldi, pca_mri):
    """
    Computes t-SNE from dimension reduction data.

    Based on scikit module

    Parameters
    ----------
    pca_maldi: np.ndarray
        PCA coordinates for the MALDI images
    pca_mri: np.ndarray
        PCA coordinates for the MRI image

    Returns
    ----------
    np.ndarray
        tSNE coordinates for MALDI
    np.ndarray
        tSNE coordinates for MRI
    """
    size_train = pca_maldi.shape[0]
    X = np.vstack((pca_maldi,pca_mri))
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform( X )
    X_train_tsne = X_tsne[0:size_train,:]
    X_test_tsne  = X_tsne[size_train:,:]
    return X_train_tsne, X_test_tsne

def weighted_distance(X, weights):
    """
    Weighted euclidean distance
    for similarity measure

    Parameters
    ----------
    X: np.ndarray
        coordinates
    weights: list
        weights for each coordinate

    """
    return np.sqrt(np.sum(X**2 * weights))

def select_images(images, point_mri, centers, weights, mzs, labels, top=1):
    """
    Sort the (MALDI) images according to their proximity to
    the MRI image

    Parameters
    ----------
    images: np.ndarray
        MALDI images
    centers: np.ndarray
        MALDI images (reduced)
    point_mri: np.ndarray
        MRI image (reduced)
    weights: list
        weights for each coordinate
    mzs: list
        m/z ratios
    labels: np.ndarray
        clustering labels (optional)
    top: int
        number of images to extract

    Returns
    ----------
    np.ndarray
        sorted MALDI images based on proximity to MRI
    np.ndarray
        sorted mzs based on proximity to MRI
    np.ndarray
        metric values used to sort MALDI
    """
    distances = np.array([weighted_distance(center-point_mri, weights) for center in centers])
    indices = [i for i in range(len(distances))]
    indices.sort(key=lambda x: distances[x])
    if top is None:
        similar_images = images[..., indices]
        similar_mzs = mzs[indices]
        distances = distances[indices]
    else:
        indices = np.array(indices)
        condition = np.any(np.array([labels == indices[i] for i in range(top)]), axis=0)
        similar_images = images[..., condition]
        similar_mzs = mzs[condition]
    return similar_images, similar_mzs, distances

def extract_ratio_images(image, mzs):
    """
    Extracts ratio images : image_n / image_{n-x}, x > 0

    Computing (n**2-n)/2 images

    Parameters
    ----------
    image: np.ndarray
        array of input images
    mzs: list
        associated labels (e.g. m/z ratios)
    """
    z = image.shape[-1]
    c = 0
    new_mzs = np.zeros(((z**2-z)//2,), dtype='object')
    ratio_images = np.zeros(image.shape[:-1] + ((z**2-z)//2, ), dtype=np.uint8)
    for i in range(z-1, 0, -1):
        for j in range(i):
            first_image = image[..., i].astype(np.float64)
            second_image = image[..., j].astype(np.float64)
            divided = np.zeros_like(first_image, dtype=np.float64)
            np.divide(first_image, second_image, out=divided, where=second_image!=0)
            divided = np.uint8(cv.normalize(divided, None, 0, 255, cv.NORM_MINMAX))
            if np.all((divided == divided.min()) | (divided == divided.max())):
                continue
            ratio_images[..., c] = divided
            current_ratio = mzs[i] + "/" + mzs[j]
            new_mzs[c] = current_ratio
            c += 1
    return ratio_images, new_mzs


def get_score(model, data, scorer=metrics.explained_variance_score):
    """
    Estimate performance of the model on the data

    Parameters
    ----------
    model: sklearn.decomposition
        matrix-factorization technique
    data: np.ndarray
        matrix used to estimate performance
    scorer: sklearn.metrics
        metric

    Returns
    ----------
    float
        performance metric

    """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)


def reconstruct_image_from_components(components, weights):
    return np.sum([components[..., i].T * weights[i] for i in range(len(weights))], axis=0)

def closest_pixels_cosine(image1, image2):
    indices = np.where((image1 != 0) & (image2 != 0))[0]
    image1_positive = np.float64(image1[indices])
    image2_positive = np.float64(image2[indices])
    print(np.linalg.norm(image1_positive.flatten()))
    print(image1_positive.max())
    print(image2_positive.max())
    image1_positive /= np.linalg.norm(image1_positive.flatten())
    image2_positive /= np.linalg.norm(image2_positive.flatten())
    abs_diff = np.abs(image1_positive - image2_positive)
    indices_abs_diff = [i for i in range(len(indices))]
    indices_abs_diff.sort(key=lambda x:abs_diff[x], reverse=False)
    return indices[indices_abs_diff]

def cosine_neighborhood(image1, image2, r):
    size = (2*r+1)**2
    shape = np.prod(image1.shape[:-1])*size
    image1_neighborhood = np.zeros((image1.shape[-1], shape))
    image2_neighborhood = np.zeros((1, shape))
    for k in range(image1.shape[-1]):
        for index in np.ndindex(image1.shape[:-1]):
            image1_index = index + (k,)
            values_im1 = np.array([image1[image1_index]] * size)

            flat_index = np.ravel_multi_index(index, image1.shape[:-1])
            flat_index *= size
            image1_neighborhood[k, flat_index:flat_index+size] = values_im1
            if k == 0:
                values_im2 = image2[index[0]-r:index[0]+r+1, index[1]-r:index[1]+r+1].copy().flatten()
                if not values_im2.any():
                    values_im2 = np.zeros((size,))
                image2_neighborhood[k, flat_index:flat_index+size] = values_im2

    sim = cosine_similarity(image1_neighborhood, image2_neighborhood)
    return sim



def closest_reconstruction(image, image1, image2, image_eigenvectors):
    w_2 = image2 / np.sum(image2)
    reconstructed_image2 = reconstruct_image_from_components(image_eigenvectors, w_2.T)
    reconstructed_image2 = imzmlio.normalize(reconstructed_image2)
    diff = np.zeros((image1.shape[0],))
    for index in range(image1.shape[0]):
        w = image1[index, ...] / np.sum(image1[index, ...])
        reconstructed_image = reconstruct_image_from_components(image_eigenvectors, w.T)
        reconstructed_image = imzmlio.normalize(reconstructed_image)
        diff[index] = np.mean(np.abs(reconstructed_image2 - reconstructed_image))
        # fig, ax = plt.subplots(1,3)
        # print(np.mean(np.abs(image[..., index].T - reconstructed_image)), diff[index], w)
        # ax[0].imshow(image[..., index].T)
        # ax[1].imshow(reconstructed_image2)
        # ax[2].imshow(reconstructed_image)
        # plt.show()

    return diff

def remove_indices(image):
    to_remove = []
    for i in range(image.shape[-1]):
        current_image = image[..., i]
        obj_image = current_image[current_image > 0]
        if not obj_image.any() or np.median(obj_image) == current_image.max():
            to_remove.append(i)
    return to_remove
