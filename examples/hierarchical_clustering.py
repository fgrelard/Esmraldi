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

def onclick(event, linkage, pos_array, shape):
    if event.inaxes != ax[0]:
        return
    if event.dblclick:
        clusters = hc.fcluster(linkage, t=event.ydata, criterion="distance")
        cluster_image = clusters.reshape(shape)
        print("k=", cluster_image.max())
        if len(shape) >= 2:
            ax_scatter.imshow(cluster_image, cmap="Set1")
        else:
            for a in artists:
                a.remove()
            artists.clear()
            cm = plt.cm.get_cmap("Set3")
            n_cm = len(cm.colors)
            points = ax_scatter.collections[0]
            points.set_color(cm(cluster_image%n_cm))
            for i in np.unique(cluster_image):
                indices = np.where(cluster_image == i)[0]
                print(current_image.shape)
                av_image = np.mean(current_image[..., indices], axis=-1)
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



def draw_graph(matrix, mzs, is_mds, new_separation=False, color_regions="b", is_text=True):
    ax_scatter.clear()
    print(is_mds)
    diffs = matrix[matrix>0]
    smallest_value = diffs.max() - diffs.min()
    print(smallest_value)
    distance_matrix = matrix
    not_diag = ~np.eye(distance_matrix.shape[0], dtype=bool)
    not_diag_min = distance_matrix[not_diag].min()
    distance_matrix = (distance_matrix - not_diag_min) / (distance_matrix.max() - not_diag_min)
    print(matrix[0,0])
    np.fill_diagonal(distance_matrix, 0)
    if new_separation:
        k = 1/smallest_value
        k = 5
        distance_matrix = matrix*k
    if is_mds:
        mds = TSNE(n_components=2, metric="precomputed", perplexity=2, learning_rate="auto", early_exaggeration=5)
        # mds = MDS(n_components=2, dissimilarity="precomputed")
        # mds = KernelPCA(n_components=2, kernel='rbf', gamma=10)
        # mds = umap.UMAP(random_state=0, n_neighbors=3, min_dist=0.4, metric="euclidean")
        pos_array = mds.fit_transform(distance_matrix).T
        print(pos_array.shape)


    else:
        np.divide(1.0, distance_matrix, out=distance_matrix, where=distance_matrix!=0)
        G = nx.from_numpy_matrix(distance_matrix)

        # test_distance_layout(G)
        # test_compare_layouts(G)

        pos = nx.spring_layout(G, k=1, dim=dim)
        # pos = nx.kamada_kawai_layout(G, dim=dim)
        pos_array = np.array(list(pos.values())).T

    print(pos_array.shape)
    ax_scatter.scatter(*pos_array, marker='o', s=50, edgecolor='None', c=color_regions, picker=True)
    # mplcursors.cursor(multiple=True).connect("add", lambda sel: sel.annotation.set_text("{:.3f}".format(mzs[sel.target.index])))
    if is_text:
        for k, p in enumerate(pos_array.T):
            ax_scatter.text(*p, "{:.2f}".format(mzs[k]))

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
            print(initial_pos)
        pos_array = np.array(list(pos.values())).T
        print(pos_array.shape)
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
    global current_image, color_regions
    if axes == ax[1]:
        return
    print("Lims change")
    xmin, xmax = ax[0].xaxis.get_view_interval()
    xs = dendro["icoord"]
    if xmin < np.amin(xs) and xmax > np.amax(xs):
        print("Full reset")
        pos_array = draw_graph(distance_matrix, mzs, is_mds, new_separation=False, color_regions=color_regions)
        # current_image = current_image
    else:
        print("Partial view")
        indices = np.where((xs >= xmin) & (xs <= xmax))[0]
        indices, order = np.unique(indices, return_index=True)
        indices = indices[np.argsort(order)]
        indices_mzs = np.array(dendro["leaves"])[indices]
        m = mzs[indices_mzs]
        print(color_regions.shape)
        c = color_regions[indices_mzs]
        # current_image = current_image[..., indices_mzs]
        d = distance_matrix[indices_mzs, :][:, indices_mzs]
        pos_array = draw_graph(d, m, is_mds, new_separation=True, color_regions=c)
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

def fsim_distance(x, y):
    print("sh", x[..., np.newaxis].shape)
    value = 1-fsim(x[..., np.newaxis], y[..., np.newaxis])
    print(value)
    return value

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


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-p", "--preprocess", help="Normalize", action="store_true")
parser.add_argument("-n", "--normalization", help="Normalize w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output .csv files with stats")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("--mds", help="Use Multidimensional scaling to project points", action="store_true")
args = parser.parse_args()

input_name = args.input
output_name = args.output
region_names = args.regions
is_normalized = args.preprocess
normalization = float(args.normalization)
is_mds = args.mds

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
    norm_img = imageutils.get_norm_image(image, normalization, mzs)
    for i in range(image.shape[-1]):
        image[..., i] = imageutils.normalize_image(image[...,i], norm_img)

if is_normalized:
    # for i in range(image.shape[-1]):
    #     currimg = image[..., i]
    #     image[..., i] = currimg/np.std(currimg)
    image = io.normalize(image)

color_regions = ["m"] * image.shape[-1]
regions = []
if region_names is not None:
    c = ["r", "g", "b"]
    for i, region_name in enumerate(region_names):
        region = read_image(region_name)
        if is_normalized:
            region = io.normalize(region)
        regions.append(region)
        mzs = np.append(mzs, -1)
        color_regions.append(c[i%3])
    image = np.dstack((image, np.dstack(regions)))

color_regions = np.array(color_regions)


# mzs_target = [837.549, 863.56,
#               773.534, 771.51,
#               885.549, 437.2670,
#               871.57, 405.2758,#dispersion
#               859.531372070312, 714.5078, #LB
#               644.5015869, 715.5759, #LT
#               287.0937, 296.0824, 746.512]

# mzs_target = [837.549, 871.57]
# indices = [np.abs(mzs - mz).argmin() for mz in mzs_target]

# color_regions[indices] = "y"

current_image = image.copy()

# mzs = np.array(mzs_target)
# color_regions = color_regions[indices]
# current_image = current_image[..., indices]
# current_image = current_image.astype(float)

image_norm = fusion.flatten(current_image, is_spectral=True)

is_spectral = True
shape = (image_norm.shape[0],)

if not is_spectral:
    image_norm = image_norm.T
    shape = image.shape[:-1]

print(image_norm.shape)
print(image.shape)


# # analyse_intensity_distributions(image_norm)
metrics = ["cosine", "correlation", "euclidean", "sqeuclidean", "cityblock", "hamming", "chebyshev", "canberra", "braycurtis", "jensenshannon"]

metrics = ["cosine"]

fig, ax = plt.subplots(2, max(2,len(metrics)//2))
for i, s in enumerate(metrics):
    # distance_matrix = distance.squareform(distance.pdist(image_norm, metric=lambda u, v: vifp(u.reshape(current_image.shape[:-1])[..., np.newaxis],v.reshape(current_image.shape[:-1])[..., np.newaxis])))
    # distance_matrix = distance.squareform(distance.pdist(image_norm, metric=lambda u, v: call_metrics(u,v,s)))
    distance_matrix = distance.squareform(distance.pdist(image_norm, metric="cosine"))
    # distance_matrix = distance.squareform(distance.pdist(image_norm, metric=lambda u, v: correlation_histmatching(u,v)))
    # plt.imshow(distance_matrix, cmap="RdBu", interpolation="nearest")
    # ix = i//(len(metrics)//2)
    # iy = i%(len(metrics)//2)
    # ax[ix, iy].set_title(metrics[i])
    # ax[ix, iy].imshow(distance_matrix, cmap="RdBu", interpolation="nearest")
    # pos = np.arange(0, len(mzs))
    # ax[ix, iy].set_xticks(pos)
    # ax[ix, iy].set_xticklabels(np.around(mzs, 2))
    # ax[ix, iy].set_yticks(pos)
    # ax[ix, iy].set_yticklabels(np.around(mzs, 2))
# plt.imshow(distance_matrix, cmap="RdBu", interpolation="nearest")
# plt.show()

model = AgglomerativeClustering(linkage="average", affinity="precomputed", n_clusters=None, distance_threshold=0)
model = model.fit(distance_matrix)

fig, ax = plt.subplots(1, 3)
linkage_matrix = get_linkage(model)

# Plot the corresponding dendrogram
dendro = hc.dendrogram(linkage_matrix, truncate_mode=None, p=10, no_plot=True)
plot_tree(dendro, ax[0])

artists = []
is_3D = False
dim = 2
if is_3D:
    dim = 3
    ax_scatter = plt.axes(projection="3d")
else:
    ax_scatter = ax[1]

pos_array = draw_graph(distance_matrix, mzs, is_mds, False, color_regions)

im_display = ax[2].imshow(image[..., 0].T)
cid = fig.canvas.mpl_connect('button_press_event', lambda event:onclick(event, linkage_matrix, pos_array, shape))
fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, mzs, current_image, im_display))
ax[0].callbacks.connect('xlim_changed', on_lims_change)

fig, ax_scatter = plt.subplots()
draw_graph(distance_matrix, mzs, is_mds, False, color_regions, is_text=True)

plt.tight_layout()
plt.show()

# G = nx.drawing.nx_agraph.to_agraph(G)

# G.node_attr.update(color="red", style="filled")
# G.edge_attr.update(color="blue", width="0.001")
