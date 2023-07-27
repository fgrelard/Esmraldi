import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
import xlsxwriter
import os
import pandas as pd

from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from skimage.filters import threshold_multiotsu, threshold_niblack, threshold_sauvola, threshold_otsu
from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth, DBSCAN, KMeans, OPTICS, BisectingKMeans
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, TSNE
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp, _vifp_single, _uqi_single

import scipy.cluster.hierarchy as hc
import scipy.spatial.distance as distance
from scipy.stats import pearsonr, norm
from scipy.signal import correlate
import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import esmraldi.segmentation as seg
import esmraldi.imageutils as imageutils
import esmraldi.utils as utils
from esmraldi.sliceviewer import SliceViewer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import networkx as nx
from sklearn.mixture import GaussianMixture
from skimage import feature
from skimage.exposure import match_histograms
import umap

from scipy import ndimage as nd
from skimage.filters import gabor_kernel
import esmraldi.haarpsi as haarpsi


current_clusters = None
variable = 1.0

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
    global current_clusters
    ind = event.ind
    if not ind.size:
        return
    ind = ind[0]
    if ind < mzs.size:
        if current_clusters is None:
            print(mzs[ind])
        elif ind < current_clusters.size:
            print(mzs[ind], current_clusters[ind])
    currimg = image[..., ind]
    im_display.set_data(currimg.T)
    im_display.set_clim(vmin=currimg.min(), vmax=currimg.max())
    im_display.axes.figure.canvas.draw()


def update_graph_clustering(ax, clusters, mzs, distance_matrix, mzs_target=None, image=None, output_name=None, mzs_display=None):
    ax[1, 1].clear()
    smallest_value = distance_matrix[distance_matrix > 0].min()
    print("Smallest value", smallest_value)
    colors = clusters.copy()
    cm = plt.cm.get_cmap("tab20")
    all_colors = cm(colors)
    colors_fixed = ["r", "g", "b"]
    n_colors = distance_matrix.shape[0]-colors.size
    for i in range(n_colors):
        current_color = [0, 0, 0, 1]
        current_color[i] = 1
        current_color = np.array(current_color).reshape(1, -1)
        all_colors = np.concatenate((all_colors, current_color))
        colors = np.append(colors, colors_fixed[i%3])
    # k = np.where(colors[:, None] == colors[:, None].T, 1, 1/(smallest_value-np.floor(smallest_value)))
    k = np.where(colors[:, None] == colors[:, None].T, 1, 10)
    k[-n_colors:] = 1
    k[:, -n_colors:]=1
    print(k)
    print(distance_matrix.shape,  k.shape)
    matrix = distance_matrix*k
    for i in range(n_colors):
        current_ind = -n_colors+i
        row = matrix[current_ind, :]
        index = np.ma.masked_less_equal(row, 0).argmin()
        matrix[current_ind, :] = matrix[index]

    mds = umap.UMAP(random_state=1, n_neighbors=3, min_dist=0.9, metric="euclidean")
    pos_array = mds.fit_transform(matrix).T
    scatter = ax[1, 1].scatter(*pos_array, marker='o', s=50, c=all_colors, edgecolor='None', picker=True)
    if image is not None and output_name is not None:
        os.makedirs(output_name, exist_ok=True)
        if mzs_display is None:
            mzs_display = mzs
        info_export = np.vstack((mzs_display, pos_array, colors)).T
        np.savetxt(output_name + os.path.sep + "2D_coordinates.csv", info_export, fmt="%s", delimiter=",")
        for k, color in enumerate(np.unique(colors[:-n_colors])):
            indices = (colors == color)
            av_image = np.mean(image[..., indices], axis=-1).astype(np.float32)
            sitk.WriteImage(sitk.GetImageFromArray(av_image.T), output_name + "av_image" + str(k) + ".tif")
    if mzs_target is not None:
        ind = utils.indices_search_sorted(mzs_target, mzs)
        for k, p in enumerate(pos_array.T):
            if k in ind and k < mzs.size:
                ax[1, 1].text(*p, "{:.2f}".format(mzs[k]))


def determine_when_same_group(linkage, shape, mzs, mzs_target, max_value):
    for i in range(max_value):
        clusters = hc.fcluster(linkage, t=i, criterion="maxclust")
        print(np.max(clusters))
        cluster_image = clusters.reshape(shape)
        label_target = []
        ind = utils.indices_search_sorted(mzs_target, mzs)
        label_target = clusters[ind]
        label = label_target[0]
        points_label = clusters[clusters == label]
        unique_size = len(np.unique(label_target))
        print(i, unique_size, len(points_label))
        if len(points_label) == len(mzs_target) and unique_size == 1:
            print("First found at cluster", i)
            return i
    print("No exclusive cluster.")
    return -1

def onclick(event, pos_array, shape, matrix=None, linkage=None, cluster_image=None, mzs_target=None):
    if event.inaxes != ax[0, 0]:
        return
    if event.dblclick:
        update_clustering(pos_array, event.ydata, shape, criterion="distance", matrix=matrix, linkage=linkage, cluster_image=cluster_image, mzs_target=mzs_target)

def tree_bisecting_kmeans(image):
    max_lim = int(np.floor(np.log2(image.shape[0])))
    print(max_lim)
    a_range = [2**(i+1) for i in range(max_lim)]
    inertias=[]
    cluster_hierarchy = []
    for i in a_range:
        bisecting = BisectingKMeans(n_clusters=i, random_state=0, n_init=2)
        bisecting_fit = bisecting.fit(image)
        clusters = bisecting_fit.labels_
        inertias.append(bisecting_fit.inertia_)
        print(i, inertias)
        cluster_hierarchy.append(clusters)
    plt.plot(a_range, inertias)
    plt.show()
    return cluster_hierarchy


def update_clustering(pos_array, y, shape, criterion="distance", matrix=None, linkage=None, cluster_image=None, mzs_target=None, image=None, output_name=None, mzs_display=None):
    global variable, current_clusters
    if cluster_image is None:
        clusters = hc.fcluster(linkage, t=y, criterion=criterion, depth=5)
        current_clusters = clusters.copy()
        print(np.max(clusters))
        cluster_image = clusters.reshape(shape)
    print("k=", cluster_image.max())
    if len(shape) >= 2:
        ax[1, 0].imshow(cluster_image.T, cmap="Set1")
    else:
        for a in artists:
            a.remove()
        artists.clear()
        cm = plt.cm.get_cmap("gist_rainbow")
        points = ax_scatter.collections[0]
        if hasattr(cm, "colors"):
            n_cm = len(cm.colors)
            points.set_color(cm(cluster_image%n_cm))
        else:
            n_cm = cluster_image.max()
            points.set_color(cm(cluster_image/n_cm))
        update_graph_clustering(ax, clusters, mzs, matrix, mzs_target, image, output_name, mzs_display)
        for i in np.unique(cluster_image):
            indices = np.where(cluster_image == i)[0]
            av_image = np.mean(image[..., indices], axis=-1)
            x0, y0 = np.median([pos_array[0, indices], pos_array[1, indices]], axis=-1)
            img = OffsetImage(av_image.T, zoom=0.1, cmap='gray')
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            # artists.append(ax_scatter.add_artist(ab))
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

def elbow(linkage):
    inc = hc.inconsistent(linkage, d=5)
    print(inc)
    print(hc.maxinconsts(linkage, inc))
    fig, ax = plt.subplots()
    last = linkage[-linkage.shape[0]:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    ax.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    ax.plot(idxs[:-2] + 1, acceleration_rev)
    fig.show()


def inconsistencies(linkage, output_name):
    arange = np.arange(0.5, 2, 0.05)
    clust_numbers = []
    for i in arange:
        clusters = hc.fcluster(linkage, t=i, criterion="inconsistent", depth=5)
        clust_numbers.append(np.amax(clusters))
    label = np.vstack((clust_numbers, arange)).T
    if output_name is not None:
        np.savetxt(output_name + os.path.sep + "inconsistencies.csv", label, delimiter=",", comments="", header="clust_number,inconsistency")
    fig, ax = plt.subplots()
    ax.plot(clust_numbers, arange)
    fig.show()


def draw_graph(matrix, ax, mzs, is_mds, new_separation=False, color_regions="b", is_text=True):
    ax.clear()
    diffs = matrix[matrix>0]
    smallest_value = diffs.max() - diffs.min()
    distance_matrix = matrix
    not_diag = ~np.eye(distance_matrix.shape[0], dtype=bool)
    not_diag_min = distance_matrix[not_diag].min()
    distance_matrix = (distance_matrix - not_diag_min) / (distance_matrix.max() - not_diag_min)
    np.fill_diagonal(distance_matrix, 0)
    if new_separation:
        k = 1/smallest_value
        # k = 5
        distance_matrix = matrix*k
    if is_mds:
        # mds = TSNE(n_components=2, metric="precomputed", perplexity=1, learning_rate=1, early_exaggeration=1)
        # mds = MDS(n_components=2, dissimilarity="precomputed")
        # mds = KernelPCA(n_components=2, kernel='rbf', gamma=10)
        mds = umap.UMAP(random_state=0, n_neighbors=3, min_dist=0.1, metric="euclidean")
        pos_array = mds.fit_transform(distance_matrix).T


    else:
        np.divide(1.0, distance_matrix, out=distance_matrix, where=distance_matrix!=0)
        G = nx.from_numpy_matrix(distance_matrix)

        # test_distance_layout(G)
        # test_compare_layouts(G)

        pos = nx.spring_layout(G, k=1, dim=dim)
        # pos = nx.kamada_kawai_layout(G, dim=dim)
        pos_array = np.array(list(pos.values())).T

    ax.scatter(*pos_array, marker='o', s=50, edgecolor='None', c=color_regions, picker=True)
    # mplcursors.cursor(multiple=True).connect("add", lambda sel: sel.annotation.set_text("{:.3f}".format(mzs[sel.target.index])))
    if is_text and mzs.size == pos_array.T.size:
        for k, p in enumerate(pos_array.T):
            ax.text(*p, "{:.2f}".format(mzs[k]))

    if not is_3D:
        ax_scatter.axis('equal')
    return pos_array

def test_distance_layout(G):
    initial_pos = None
    fixed_pos = None
    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        inpos = initial_pos
        if initial_pos is not None:
            inpos[0] *= i+1
        pos = nx.spring_layout(G, k=1.0/(i+1))
        if initial_pos is None:
            initial_pos = {0: pos[0]}
            fixed_pos = [0]
        pos_array = np.array(list(pos.values())).T
        ax[i].scatter(pos_array[0], pos_array[1], marker='o', s=50, edgecolor='None')

    plt.show()

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

def on_lims_change(axes):
    global image, color_regions
    if axes == ax[0, 1]:
        return
    print("Lims change")
    xmin, xmax = ax[0, 0].xaxis.get_view_interval()
    xs = dendro["icoord"]
    if xmin < np.amin(xs) and xmax > np.amax(xs):
        print("Full reset")
        pos_array = draw_graph(distance_matrix, ax_scatter, mzs, is_mds, new_separation=False, color_regions=color_regions)
    else:
        print("Partial view")
        indices = np.where((xs >= xmin) & (xs <= xmax))[0]
        indices, order = np.unique(indices, return_index=True)
        indices = indices[np.argsort(order)]
        indices_mzs = np.array(dendro["leaves"])[indices]
        m = mzs[indices_mzs]
        print(color_regions.shape)
        c = color_regions[indices_mzs]
        d = distance_matrix[indices_mzs, :][:, indices_mzs]
        pos_array = draw_graph(d, ax_scatter, m, is_mds, new_separation=True, color_regions=c)
    # plot_tree(dendro, axes, pos=indices)


def test_compare_layouts(G):
    initial_pos = None
    fixed_pos = None
    fig, ax = plt.subplots(1, 4)
    for i in range(4):
        if i == 0:
            pos = nx.spring_layout(G, k=1)
        elif i == 1:
            pos = nx.kamada_kawai_layout(G)
        elif i == 2:
            pos = nx.spectral_layout(G)
        elif i == 3:
            pos = nx.circular_layout(G)
        pos_array = np.array(list(pos.values())).T
        ax[i].scatter(pos_array[0], pos_array[1], marker='o', s=50, edgecolor='None')

    plt.show()


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


def analyse_intensity_distributions(image_norm):
    best_regions = None
    for i in range(image_norm.shape[0]):
        values = image_norm[i, ...]
        X = values.reshape(image.shape[:-1])
        best_coeff = -1
        for i in range(3):
            number = i + 2
            thresholds = threshold_multiotsu(X, number)
            regions = np.digitize(X, bins=thresholds)
            region_flatten = regions.flatten()
            coeffr = cosine_similarity(region_flatten[region_flatten>0].reshape((1, -1)), values[region_flatten>0].reshape((1, -1)))
            if coeffr > best_coeff:
                best_coeff = coeffr
                best_regions = regions
                best_number = number
                best_thresholds = thresholds

        # for t in best_thresholds:
        #     img = np.where(X > t, X, 0)
        #     plt.imshow(img)
        #     plt.show()

        gm = GaussianMixture(n_components=best_number, random_state=0).fit(values.reshape(-1, 1))
        means = gm.means_
        covariances = gm.covariances_
        weights = gm.weights_
        x_axis = np.linspace(1, values.max(), 1000)

        print(thresholds, means)
        print(best_number, best_thresholds)
        im = np.digitize(X, best_thresholds)
        fig, ax = plt.subplots(1, 3)
        logbins = np.geomspace(1, values.max(), 200)
        img = np.where(X > best_thresholds[-1], 255, 0)
        varimg = imageutils.variance_image(img, 2)
        varimg = feature.canny(X, sigma=10)
        ax[0].imshow(X)
        ax[1].imshow(varimg)
        hist, bin_edges = np.histogram(values, bins=200)
        pdfs = [norm.pdf(x_axis, means[i][0], np.sqrt(covariances[i][0]))*weights[i] for i in range(len(means)) if means[i][0] != 0]
        ax[2].hist(values, bins=logbins, density=True)
        # for j in range(len(pdfs)):
        #     ax[2].plot(x_axis, pdfs[j], "r")
        ax[2].set_xscale("log")
        ax[2].set_yscale("log")
        plt.show()

def estimate_inertia(X, labels):
    print( X.shape, labels.shape)
    centroids = []
    clusters = labels - 1
    for i in np.unique(clusters):
        current_points = X[clusters == i, ...]
        centroid = np.mean(current_points, axis=0)
        centroids.append(centroid)
    inertia = 0
    for i in range(X.shape[0]):
        point = X[i, ...]
        centroid = centroids[clusters[i]]
        inertia += np.sum((point - centroid)**2)
    return inertia



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-p", "--preprocess", help="Normalize", action="store_true")
parser.add_argument("--normalization_dataset", help="Normalization dataset", default=None)
parser.add_argument("-n", "--normalization", help="Normalize w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output .csv files with stats")
parser.add_argument("-r", "--regions", help="Subregions inside mask to plot inside UMAP", nargs="+", type=str)
parser.add_argument("--mds", help="Use Multidimensional scaling to project points", action="store_true")
parser.add_argument("--roc", help="ROC file (.xlsx)")
parser.add_argument("--names", help="Region names to restrict ROC", nargs="+", default=None)
parser.add_argument("--value", help="Threshold for correlation using the AUC-ROC or distance criterion, depending on whether a ROC file is supplied.", default=None)
parser.add_argument("--correlation_names", help="Images used to compute correlation with ion images. When correlation is high, corresponding ion images are discarded.", nargs="+", default=None)
parser.add_argument("--restrict_coordinates", help="Restrict coordinates", nargs="+", type=int)
parser.add_argument("--inconsistency", help="Inconsistency parameter to find optimal number of clusters", default=1.7)

args = parser.parse_args()

input_name = args.input
output_name = args.output
region_names = args.regions
is_normalized = args.preprocess
normalization = float(args.normalization)
normalization_dataset = args.normalization_dataset
is_mds = args.mds
roc_name = args.roc
roc_names = args.names
correlation_names = args.correlation_names
correlation_value = args.value
inconsistency = float(args.inconsistency)

if correlation_value is not None:
    correlation_value = float(correlation_value)
restrict_coordinates = args.restrict_coordinates

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
    else:
        norm_img = imageutils.get_norm_image(image, normalization, mzs)
    for i in range(image.shape[-1]):
        image[..., i] = imageutils.normalize_image(image[...,i], norm_img)


if restrict_coordinates:
    min_c = [restrict_coordinates[0], restrict_coordinates[1]]
    max_c = [restrict_coordinates[2], restrict_coordinates[3]]
    image = image[min_c[0]:max_c[0], min_c[1]:max_c[1]]


indices_restrict = [True for i in range(image.shape[-1])]
if correlation_names is None:
    if roc_name is not None:
        roc_values_df = pd.read_excel(roc_name)
        roc_auc_scores = np.array(roc_values_df)
        names = roc_values_df.columns
        if roc_names is None:
            end = roc_auc_scores.shape[-1]
            ind_names = np.arange(end).astype(int)
        else:
            ind_names = np.array([n in roc_names for n in names])
        mzs_roc = roc_auc_scores[:, 0]
        roc_auc_scores = roc_auc_scores[:, ind_names]
        cond = (roc_auc_scores < correlation_value)

        indices_restrict = np.all(cond, axis=-1)
        indices_restrict = np.where(indices_restrict)[0]
        indices_mzs = [i for i, mz in enumerate(mzs) if mz in mzs_roc[indices_restrict]]
        indices_restrict = np.intersect1d(indices_mzs, indices_restrict)
else:
    correlation_images = []
    for correlation_name in correlation_names:
        correlation = read_image(correlation_name)
        correlation_images.append(correlation)
    if correlation_value is not None:
        similar_images, distances, indices_restrict = seg.find_similar_image_distance_map_percentile(image, correlation_images, correlation_value, quantiles=[0], add_otsu_thresholds=True, return_indices=True)
        indices_restrict = np.invert(indices_restrict)
        label = np.vstack((mzs, distances, indices_restrict)).T
        fig, ax = plt.subplots()
        ax.plot(mzs, distances)
        fig.show()
        if output_name is not None:
            np.savetxt(output_name + os.path.sep + "distances.csv", label, delimiter=",", comments="", header="mzs,distances,restrict")
        np.set_printoptions(suppress=True)
        fig, ax = plt.subplots(1)
        label = np.vstack((mzs, distances)).T
        tracker = SliceViewer(ax, np.transpose(image, (2, 1, 0)), labels=label)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

if not np.all(indices_restrict):
    indices_discard = np.invert(indices_restrict)
    image_discard = image[..., indices_discard]
    mzs_discard = mzs[indices_discard]
    fig, ax = plt.subplots(1)
    label = mzs_discard
    tracker = SliceViewer(ax, np.transpose(image_discard, (2, 1, 0)), labels=label)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

print("Keeping", np.count_nonzero(indices_restrict), "images.")

indices_restrict = np.array(indices_restrict).astype(bool)
image = image[..., indices_restrict]
print(image.shape)
mzs = mzs[indices_restrict]
io.to_tif(image.T, mzs, os.path.splitext(input_name)[0] + "_filtered.tif")
print(image.shape)


if is_normalized:
    # for i in range(image.shape[-1]):
    #     currimg = image[..., i]
    #     image[..., i] = currimg/np.std(currimg)
    image = io.normalize(image)
    image = image.astype(np.float128) / 255.0





# mzs_target = [837.549, 863.56,
#               773.534, 771.51,
#               885.549, 437.2670,
#               871.57, 405.2758,#dispersion
#               859.531372070312, 714.5078, #LB
#               644.5015869, 715.5759, #LT
#               287.0937, 296.0824, 746.512]

mzs_target = [837.549, 838.56, 863.56, 889.58]
indices = [np.abs(mzs - mz).argmin() for mz in mzs_target]


color_regions = ["m"] * image.shape[-1]
regions = []
mzs_display = mzs.copy().astype(str)
if region_names is not None:
    c = ["r", "g", "b"]
    for i, region_name in enumerate(region_names):
        name_trimmed, ext = os.path.splitext(os.path.basename(region_name))
        region = read_image(region_name)
        if is_normalized:
            region = io.normalize(region)
            region = region.astype(np.float128) / 255.0
        regions.append(region)
        mzs = np.append(mzs, -1)
        mzs_display = np.append(mzs_display, name_trimmed)
        color_regions.append(c[i%3])
    image = np.dstack((image, np.dstack(regions)))

image_norm = fusion.flatten(image, is_spectral=True)

is_spectral = True
shape = (image_norm.shape[0]-len(regions),)

if not is_spectral:
    image_norm = image_norm.T
    shape = image.shape[:-1]

color_regions = np.array(color_regions)
color_regions[indices] = "y"



is_distribution = False
if is_distribution:
    distributions = seg.quantile_distance_distributions(image, [60, 70, 80], w=10)
    image_norm = distributions

print(image_norm.shape)
print(image.shape)


# # analyse_intensity_distributions(image_norm)
metrics = ["cosine", "correlation", "euclidean", "sqeuclidean", "cityblock", "hamming", "chebyshev", "canberra", "braycurtis", "jensenshannon"]

metrics = ["cosine"]

# fig, ax = plt.subplots(2, max(2,len(metrics)//2))

for i, s in enumerate(metrics):
    distance_matrix = distance.squareform(distance.pdist(image_norm, metric="correlation"))
    distance_matrix[np.isnan(distance_matrix)] = 0

draw_hierarchical = True
is_3D = False

if draw_hierarchical:
    # tree_bisecting_kmeans(image_norm)
    # bisecting = BisectingKMeans(n_clusters=32, random_state=0, n_init=2)
    # bisecting_fit = bisecting.fit(image_norm)
    # clusters = bisecting_fit.labels_


    model = AgglomerativeClustering(linkage="ward", affinity="euclidean", n_clusters=None, distance_threshold=0)
    # model = model.fit(distance_matrix)
    model = model.fit(image_norm[:-len(region_names), ...])

    fig, ax = plt.subplots(2, 2)

    # update_graph_clustering(ax, clusters, mzs, distance_matrix, mzs_target=mzs_target)

    linkage_matrix = get_linkage(model)
    elbow(linkage_matrix)
    inconsistencies(linkage_matrix, output_name)

    # Plot the corresponding dendrogram
    # dendro = hc.dendrogram(linkage_matrix, truncate_mode=None, p=10, no_plot=True)
    # plot_tree(dendro, ax[0, 0])

    artists = []
    dim = 2
    if is_3D:
        dim = 3
        ax_scatter = plt.axes(projection="3d")
    else:
        ax_scatter = ax[0, 1]


    pos_array = draw_graph(distance_matrix, ax_scatter, mzs, is_mds, False, color_regions)

    nb_clust = determine_when_same_group(linkage_matrix, shape, mzs, mzs_target, len(indices_restrict))

    update_clustering(pos_array, inconsistency, shape, criterion="inconsistent", matrix=distance_matrix, linkage=linkage_matrix, cluster_image=None, mzs_target=mzs_target, image=image, output_name=output_name, mzs_display=mzs_display)

    # update_clustering(pos_array, nb_clust, shape, criterion="maxclust", matrix=distance_matrix, linkage=linkage_matrix, cluster_image=None, mzs_target=mzs_target, image=image, output_name=output_name)

    im_display = ax[1, 0].imshow(image[..., 0].T)
    cid = fig.canvas.mpl_connect('button_press_event', lambda event:onclick(event, pos_array, shape, distance_matrix, linkage=linkage_matrix, cluster_image=None, mzs_target=mzs_target))
    fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, mzs, image, im_display))
    ax[0, 0].callbacks.connect('xlim_changed', on_lims_change)

fig, ax_graph = plt.subplots()
draw_graph(distance_matrix, ax_graph, mzs, is_mds, False, color_regions, is_text=True)

plt.tight_layout()
plt.show()

# G = nx.drawing.nx_agraph.to_agraph(G)

# G.node_attr.update(color="red", style="filled")
# G.edge_attr.update(color="blue", width="0.001")
