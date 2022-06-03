import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xlsxwriter
import os

from skimage.color import rgb2gray
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

import scipy.cluster.hierarchy as hc
import scipy.spatial.distance as distance

import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import networkx as nx

def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    mask = rgb2gray(mask)
    mask = mask.T
    return mask

def find_indices(image, shape):
    indices = np.where(image > 0)
    return np.ravel_multi_index(indices, shape, order='F')

def onclick(event, linkage, shape):
    if event.inaxes != ax[0]:
        return
    if event.dblclick:
        clusters = hc.fcluster(linkage, t=event.ydata, criterion="distance")
        cluster_image = clusters.reshape(shape)
        print("k=", cluster_image.max())
        if len(shape) >= 2:
            ax_scatter.imshow(cluster_image, cmap="Set1")
        else:
            cm = plt.cm.get_cmap("Set3")
            n_cm = len(cm.colors)
            points = ax_scatter.collections[0]
            points.set_color(cm(cluster_image%n_cm))
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



def draw_graph(distance_matrix, is_mds):
    ax_scatter.clear()
    print(is_mds)
    if is_mds:
        mds = MDS(n_components=2, dissimilarity="precomputed")
        pos_array = mds.fit_transform(distance_matrix).T

    else:
        np.divide(1.0, distance_matrix, out=distance_matrix, where=distance_matrix!=0)
        G = nx.from_numpy_matrix(distance_matrix)

        # test_distance_layout(G)
        # test_compare_layouts(G)

        # pos = nx.spring_layout(G, k=1, dim=dim)
        pos = nx.kamada_kawai_layout(G, dim=dim)
        pos_array = np.array(list(pos.values())).T

    print(pos_array.shape)
    ax_scatter.scatter(*pos_array, marker='o', s=50, edgecolor='None')
    for k, p in enumerate(pos_array.T):
        ax_scatter.text(*p, "{:.2f}".format(mzs[k]))

    if not is_3D:
        ax_scatter.axis('equal')

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
        ax.plot(xs, ys, color)
    if pos is None:
        ax.set_xlim(xmin-10, xmax + 0.1*abs(xmax))
        ax.set_ylim(ymin, ymax + 0.1*abs(ymax))

def on_lims_change(axes):
    xmin, xmax = axes.xaxis.get_view_interval()
    xs = dendro["icoord"]
    if xmin < np.amin(xs) and xmax > np.amax(xs):
        return
    indices = np.where((xs >= xmin) & (xs <= xmax))[0]
    indices, order = np.unique(indices, return_index=True)
    indices = indices[np.argsort(order)]
    indices_mzs = np.array(dendro["leaves"])[indices]
    d = distance_matrix[indices_mzs, :][:, indices_mzs]
    draw_graph(d, is_mds)
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

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-p", "--preprocess", help="Normalize", action="store_true")
parser.add_argument("-o", "--output", help="Output .csv files with stats")
parser.add_argument("--mds", help="Use Multidimensional scaling to project points", action="store_true")
args = parser.parse_args()

input_name = args.input
output_name = args.output
is_normalized = args.preprocess
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

if is_normalized:
    image = io.normalize(image)

is_spectral = True
image_norm = fusion.flatten(image, is_spectral=True)

shape = (image_norm.shape[0],)

if not is_spectral:
    image_norm = image_norm.T
    shape = image.shape[:-1]

print(image_norm.shape)
# image_norm = image_norm[:100]

distance_matrix = distance.squareform(distance.pdist(image_norm, metric="correlation"))



model = AgglomerativeClustering(linkage="average", affinity="precomputed", n_clusters=None, distance_threshold=0)
model = model.fit(distance_matrix)

fig, ax = plt.subplots(1, 2)
linkage_matrix = get_linkage(model)

# Plot the corresponding dendrogram
dendro = hc.dendrogram(linkage_matrix, truncate_mode=None, p=10, no_plot=True)
plot_tree(dendro, ax[0])
cid = fig.canvas.mpl_connect('button_press_event', lambda event:onclick(event, linkage_matrix, shape))
ax[0].callbacks.connect('xlim_changed', on_lims_change)

is_3D = False
dim = 2
if is_3D:
    dim = 3
    ax_scatter = plt.axes(projection="3d")
else:
    ax_scatter = ax[1]

draw_graph(distance_matrix, is_mds)

plt.tight_layout()
plt.show()

# G = nx.drawing.nx_agraph.to_agraph(G)

# G.node_attr.update(color="red", style="filled")
# G.edge_attr.update(color="blue", width="0.001")
