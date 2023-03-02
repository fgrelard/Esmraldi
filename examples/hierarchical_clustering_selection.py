import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
import xlsxwriter
import os

from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from skimage.filters import threshold_multiotsu, threshold_niblack, threshold_sauvola, threshold_otsu
from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth, DBSCAN, KMeans
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, TSNE
from sklearn.decomposition import KernelPCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp, _vifp_single, _uqi_single

import scipy.cluster.hierarchy as hc
import scipy.spatial.distance as distance
from scipy.stats import pearsonr, norm
from scipy.signal import correlate
import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import esmraldi.imageutils as imageutils
import networkx as nx
from sklearn.mixture import GaussianMixture
from skimage import feature
from skimage.exposure import match_histograms
import umap

from scipy import ndimage as nd
from skimage.filters import gabor_kernel
import esmraldi.haarpsi as haarpsi
from image_similarity_measures.quality_metrics import fsim

def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask

def find_indices(image, shape):
    indices = np.where(image > 0)
    return np.ravel_multi_index(indices, shape, order='F')

def onpick(event, mzs, image, im_display):
    ind = event.ind
    if not ind.size:
        return
    ind = ind[0]
    print(mzs[ind])
    currimg = image[..., ind]
    im_display.set_data(currimg.T)
    im_display.set_clim(vmin=currimg.min(), vmax=currimg.max())
    im_display.axes.figure.canvas.draw()

def onclick(event, linkage, shape):
    if event.inaxes != ax[0]:
        return
    if event.dblclick:
        clusters = hc.fcluster(linkage, t=event.ydata, criterion="distance")
        cluster_image = clusters.reshape(shape).T
        print("k=", cluster_image.max())
        ax[1].imshow(cluster_image)
        fig.canvas.draw_idle()

def get_linkage(model):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = model.distances_
    print("maxdist",distance.max())

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    return linkage_matrix




def plot_tree(P, ax, pos=None):
    icoord = np.array(P['icoord'])
    dcoord = np.array(P['dcoord'])
    color_list = np.array(P['color_list'])
    xmin, xmax = icoord.min(), icoord.max()
    ymin, ymax = dcoord.min(), dcoord.max()
    if pos is not None:
        icoord = icoord[pos]
        dcoord = dcoord[pos]
        color_list = color_list[pos]
    for xs, ys, color in zip(icoord, dcoord, color_list):
        ax.plot(xs, ys, "b")
    if pos is None:
        ax.set_xlim(xmin-10, xmax + 0.1*abs(xmax))
        ax.set_ylim(ymin, ymax + 0.1*abs(ymax))





def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def cosdistance(x,y):
    x = x.astype(float)
    y = y.astype(float)
    return 1-(np.dot(x, y)/(np.sqrt(np.dot(x,x))*np.sqrt(np.dot(y,y))))

def cosdistanceweighted(x,y):
    x = x.astype(float)
    y = y.astype(float)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    product = x*y
    w = np.minimum(x, y)
    product[w < np.percentile(w, 15)] = 1
    distance = 1-(np.sum(product))
    return distance

def cosw(x,y):
    xn = x.astype(float)
    yn = y.astype(float)
    return distance.cosine(xn, yn)

def call_metrics(x, y, s):
    u = x.astype(float).copy()
    v = y.astype(float).copy()
    metric = getattr(distance, s)
    return metric(u, v)


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    out_im = np.zeros_like(image)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        out_im += filtered
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[1].imshow(out_im)
    # plt.show()
    return out_im


def metric_histmatching(x,y,s):
    x = x.astype(float)
    y = y.astype(float)
    xn = x.reshape(image.shape[:-1])
    yn = y.reshape(image.shape[:-1])
    matched = match_histograms(yn, xn).astype(float)
    metric = getattr(distance, s)
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(xn)
    # ax[1].imshow(matched)
    # ax[2].imshow(d.reshape(image.shape[:-1]))
    # plt.show()
    return metric(x, matched.flatten())

def correlation_histmatching(x,y):
    x = x.astype(float)
    y = y.astype(float)
    xn = x.reshape(image.shape[:-1])
    yn = y.reshape(image.shape[:-1])
    matched = match_histograms(yn, xn).astype(float)
    return distance.correlation(x, matched.flatten())

def cospos(x,y):
    cond = (x>0) & (y>0)
    x_n = x[cond].astype(float)
    y_n = y[cond].astype(float)
    return distance.cosine(x_n,y_n)

def eucnorm(x,y):
    x = x.astype(float)
    y = y.astype(float)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    return distance.sqeuclidean(x, y)

def haar_similarity(x, y):
    x = x.astype(float)
    y = y.astype(float)
    s, i, w = haarpsi.haar_psi(x, y)
    return 1.0-s
    # fig, ax = plt.subplots(1, 4)
    # ax[0].imshow(x)
    # ax[1].imshow(y)
    # ax[2].imshow(i[..., 0])
    # ax[3].imshow(i[..., 1])
    # plt.show()

def fsim_distance(x, y):
    print("sh", x[..., np.newaxis].shape)
    value = 1-fsim(x[..., np.newaxis], y[..., np.newaxis])
    print(value)
    return value


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-p", "--preprocess", help="Normalize", action="store_true")
parser.add_argument("--normalization_dataset", help="Normalization dataset", default=None)
parser.add_argument("-n", "--normalization", help="Normalize w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
output_name = args.output

is_normalized = args.preprocess
normalization = float(args.normalization)
normalization_dataset = args.normalization_dataset

if input_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
    spectra = io.get_spectra(imzml)

    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]

    full_spectra = io.get_full_spectra(imzml)
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
    print(len(mzs))
    image = io.get_images_from_spectra(full_spectra, (max_x, max_y, max_z))
else:
    image_itk = sitk.ReadImage(input_name)
    image = sitk.GetArrayFromImage(image_itk).T
    mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv")



norm_img = None
if normalization>0:
    if normalization_dataset is not None:
        normalization_image = sitk.ReadImage(normalization_dataset)
        normalization_image = sitk.GetArrayFromImage(normalization_image).T
        mzs = np.loadtxt(os.path.splitext(normalization_dataset)[0] + ".csv")
        norm_img = imageutils.get_norm_image(normalization_image, normalization, mzs)
        plt.imshow(norm_img)
        plt.show()
    else:
        norm_img = imageutils.get_norm_image(image, normalization, mzs)
    for i in range(image.shape[-1]):
        image[..., i] = imageutils.normalize_image(image[...,i], norm_img)

if is_normalized:
    # for i in range(image.shape[-1]):
    #     currimg = image[..., i]
    #     image[..., i] = currimg/np.std(currimg)
    image = io.normalize(image)

current_image = image.copy()

image_norm = fusion.flatten(current_image, is_spectral=True)
image_norm = image_norm.T
shape = image.shape[:-1]

print(image_norm.shape)
print(image.shape)


metrics = ["cosine"]

distance_matrix = distance.squareform(distance.pdist(image_norm, metric="cosine"))
distance_matrix = np.nan_to_num(distance_matrix)

print(distance_matrix.shape)

draw_hierarchical = True
is_3D = False

if draw_hierarchical:

    model = AgglomerativeClustering(linkage="average", affinity="precomputed", n_clusters=None, distance_threshold=0)
    model = model.fit(distance_matrix)

    fig, ax = plt.subplots(1, 2)
    linkage_matrix = get_linkage(model)

    # Plot the corresponding dendrogram
    dendro = hc.dendrogram(linkage_matrix, truncate_mode="level", p=30, no_plot=True)
    plot_tree(dendro, ax[0])

    artists = []
    dim = 2
    if is_3D:
        dim = 3
        ax_scatter = plt.axes(projection="3d")
    else:
        ax_scatter = ax[1]


    im_display = ax[1].imshow(image[..., 0].T)
    cid = fig.canvas.mpl_connect('button_press_event', lambda event:onclick(event, linkage_matrix, shape))
    fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, mzs, current_image, im_display))


plt.tight_layout()
plt.show()
