import argparse
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import esmraldi.imageutils as imageutils

from sklearn.decomposition import NMF, PCA
import scipy.spatial.distance as distance
from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as hc

from esmraldi.registration import precision, recall, fmeasure
from yellowbrick.cluster import KElbowVisualizer
import sklearn.metrics as metrics

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

def onclick(event, linkage, ax_scatter, shape):
    if event.inaxes != ax[0]:
        return
    if event.dblclick:
        clusters = hc.fcluster(linkage, t=event.ydata, criterion="distance")
        cluster_image = clusters.reshape(shape)
        print("k=", cluster_image.max())
        if len(shape) >= 2:
            ax_scatter.imshow(cluster_image.T, cmap="Set1")
            fig.canvas.draw_idle()

def fmeasure_stat(clusters, target_image):
    fmeasures = []
    max_k = 0
    range_image = np.unique(clusters)
    for k in range_image:
        k_cluster = np.where(clusters == k, 1, 0)
        if np.count_nonzero(k_cluster) > 10:
            r = recall(k_cluster, target_image)
            p = precision(k_cluster, target_image)
            f = fmeasure(p, r)
            if len(fmeasures) > 0 and f > max(fmeasures):
                max_k = k
            fmeasures.append(f)
    k_cluster = np.where(clusters == max_k, 255, 0)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(clusters)
    # ax[1].imshow(k_cluster)
    # plt.show()
    imageutils.export_figure_matplotlib("clusters_30_small.tif", clusters)
    imageutils.export_figure_matplotlib("best_30_small.tif", k_cluster)
    if len(fmeasures) > 0:
        return max(fmeasures)
    return 0


def aucroc_stat(eigenvectors, target_image):
    auc_roc = []
    range_image = eigenvectors.shape[-1]
    t = 0
    for k in range(range_image):
        k_cluster = eigenvectors[..., k].T
        k_cluster_neg = np.where(k_cluster < 0, -k_cluster, 0)
        k_cluster_pos = np.where(k_cluster > 0, k_cluster, 0)
        auc = metrics.roc_auc_score(target_image.flatten(), k_cluster_neg.flatten())
        auc2 = metrics.roc_auc_score(target_image.flatten(), k_cluster_pos.flatten())
        aux = max(auc, auc2)
        if len(auc_roc) > 0 and auc > max(auc_roc):
            max_k = k
            fpr, tpr, thresholds = metrics.roc_curve(target_image.flatten(), k_cluster.flatten())
            d, index = fusion.cutoff_distance(fpr, tpr, thresholds, return_index=True)
            t = thresholds[index]
        auc_roc.append(auc)
    k_cluster = eigenvectors[..., max_k].T
    k_cluster_bin = np.where(k_cluster > t, 1, 0)
    # imageutils.export_figure_matplotlib("components_10.tif", eigenvectors)
    sitk.WriteImage(sitk.GetImageFromArray(eigenvectors.T), "components_30.tif")
    imageutils.export_figure_matplotlib("best_30.tif", k_cluster)
    imageutils.export_figure_matplotlib("best_30_bin.tif", k_cluster_bin)
    if len(auc_roc) > 0:
        return max(auc_roc)
    return 0

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-p", "--preprocess", help="Normalize", action="store_true")
parser.add_argument("-n", "--normalization", help="Normalize w.r.t. to given m/z", default=0)
parser.add_argument("--nb_components", help="Nb components for PCA", default=0)
parser.add_argument("--clustering", help="Determines whether to use hierarchical clustering instead of PCA", action="store_true")
parser.add_argument("--target", help="Target image or m/z to find", default=None)
args = parser.parse_args()

input_name = args.input
is_normalized = args.preprocess
normalization = float(args.normalization)
n = int(args.nb_components)
is_clustering = args.clustering
target = args.target

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
    image = io.normalize(image)


mzs_target = [837.549, 863.56, 889.58]
index_target = np.abs(mzs - 837.549).argmin()

if target is not None:
    if os.path.exists(target):
        image_target = sitk.ReadImage(target)
        image_target = sitk.GetArrayFromImage(image_target)
        image_target = np.where(image_target>0, 1, 0)
    else:
        index_target = np.abs(mzs - target).argmin()
        image_target = image[..., index_target]


image_shape = image.shape[:-1]
print(image_shape)
image_norm = fusion.flatten(image, is_spectral=True)
M = image_norm.T
print(M.shape)



if is_clustering:
    recalls = []
    silhouettes = []
    inertias = []
    x = np.arange(5, 100, 1)
    x = [26]
    x = [n]
    for i in x:
        kmeans = KMeans(i, random_state=0)
        fit_kmeans = kmeans.fit(M)
        labels = kmeans.predict(M)
        label_image = labels.reshape(image_shape).T
        sitk.WriteImage(sitk.GetImageFromArray(label_image.astype(np.uint8)), "cluster_" + str(i) + ".tif")
        inertias.append(fit_kmeans.inertia_)
       #  print(label_image.shape)
    #     max_recall = fmeasure_stat(label_image, image_target)
        plt.imshow(label_image)
        plt.show()
    # distance_matrix = distance.squareform(distance.pdist(M, metric="euclidean"))
    # distance_matrix[np.isnan(distance_matrix)] = 0
    # np.save("distance_matrix_nan.npy", distance_matrix)
    # distance_matrix = np.load("distance_matrix_nan.npy", mmap_mode="r+")
    model = AgglomerativeClustering(linkage="ward", affinity="euclidean", n_clusters=None, distance_threshold=0)
    model = model.fit(M)
    linkage_name = "linkage.npy"
    if os.path.exists(linkage_name):
        linkage_matrix = np.load(linkage_name)
    else:
        linkage_matrix = get_linkage(model)
        np.save(linkage_name, linkage_matrix)
    recalls = []
    x = np.arange(100, linkage_matrix.shape[0], 100)
    x = np.arange(5, 1000, 5)
    # x = [20]
    for i in x:
        clusters = hc.fcluster(linkage_matrix, t=i, criterion="maxclust")
        cluster_image = clusters.reshape(image_shape).T

        max_recall = fmeasure_stat(cluster_image, image_target)
        print(max_recall)
        recalls.append(max_recall)
    plt.plot(x, recalls)
    plt.show()
    fig, ax = plt.subplots(1, 2)
    dendro = hc.dendrogram(linkage_matrix, truncate_mode=None, p=10, no_plot=True)
    plot_tree(dendro, ax[0])
    cid = fig.canvas.mpl_connect('button_press_event', lambda event:onclick(event, linkage_matrix, ax[1], image_shape))
    plt.show()
else:
    iterations = np.arange(5, 100, 5)
    iterations = [30]
    cosines = []
    aucs = []
    for nb in iterations:
        pca = PCA(nb, random_state=0)
        fit_pca = pca.fit(M)
        eigenvectors = fit_pca.components_
        eigenvalues = fit_pca.transform(M)
        inverse_transform = pca.inverse_transform(eigenvalues)
        eigenvectors_transposed = eigenvalues.T

        print(inverse_transform.shape)

        image_eigenvectors = eigenvectors_transposed.T
        new_shape = image_shape + (image_eigenvectors.shape[-1],)
        image_eigenvectors = image_eigenvectors.reshape(new_shape)
        print(image_eigenvectors.shape)

        # weights = eigenvectors[..., index_target] # / np.sum(eigenvectors[..., index_target])
        # image_target = fusion.reconstruct_image_from_components(image_eigenvectors, weights).T
        # image_target = (image_target - image_target.min()) / (image_target.max() - image_target.min())

        # imageutils.export_figure_matplotlib(str(nb)+".png", image_target)
        auc = aucroc_stat(image_eigenvectors, image_target)
        cos = distance.cosine(image_target.flatten(), image[..., index_target].flatten())
        aucs.append(auc)
        cosines.append(cos)
        print(auc)

    plt.plot(iterations, aucs)
    plt.show()
    print(eigenvectors.shape)
