import joblib
import argparse
import numpy as np
import os
import SimpleITK as sitk
from sklearn.cross_decomposition import PLSRegression, CCA
import esmraldi.imzmlio as io
import esmraldi.utils as utils
import esmraldi.imageutils as imageutils
import esmraldi.fusion as fusion
import esmraldi.spectraprocessing as sp
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, TSNE
import umap
from mpl_toolkits import mplot3d
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN, OPTICS
import esmraldi.fusion as fusion
from matplotlib import colors

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def sample(x, y, sample_size):
    n_features = y.shape[-1]
    x_sampled = np.zeros((n_features*sample_size, x.shape[-1]))
    y_sampled = np.zeros((n_features*sample_size, y.shape[-1]))
    for i in range(n_features):
        indices = np.where(y[..., i] > 0)[0]
        cond = (y[..., i]  > 0) & (np.all([y[..., j] == 0 for j in range(n_features) if j != i], axis=0))
        print(cond)
        indices = np.where(cond)[0]
        indices = indices[:sample_size]
        x_sampled[i*sample_size:(i+1)*sample_size, ...] = x[indices, ...]
        y_sampled[i*sample_size:(i+1)*sample_size, ...] = y[indices, ...]
    return x_sampled, y_sampled


def roc_indices(y, out, labels):
    t = []
    for i in range(y.shape[-1]):
        # plt.title(names[i])
        t_curr = []
        for j in range(y.shape[-1]):
            fpr, tpr, thresholds = fusion.roc_curve(y[..., j], out[..., i])
            dist, index = fusion.cutoff_distance(fpr, tpr, thresholds, return_index=True)
            t_curr.append(thresholds[index])
            # plt.plot(fpr, tpr)
        t.append(t_curr)
        # plt.legend(names)
        # plt.show()

    t = np.array(t)
    indices = np.array([out[i, l] > t[l, l] for i, l in enumerate(labels)])
    indices2 = np.array([out[i, ...] < t[l] for i, l in enumerate(labels)])
    indices2 = np.delete(indices2, labels, axis=-1).all(axis=-1)
    indices = np.logical_and(indices, indices2)
    return ~indices

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input joblib")
parser.add_argument("--msi", help="MSI tif format")
parser.add_argument("--names", help="Names to analyze (default all)", nargs="+", type=str, default=None)
parser.add_argument("--gmm", help="GMM model (default None)", default=None)
args = parser.parse_args()

input_name = args.input
msi_name = args.msi
analysis_names = args.names
gmm_name = args.gmm

image_itk = sitk.ReadImage(msi_name)
images = sitk.GetArrayFromImage(image_itk).T

mzs_name = os.path.splitext(input_name)[0] + "_mzs.csv"
names_name = os.path.splitext(input_name)[0] + "_names.csv"
y_original_name = os.path.splitext(input_name)[0] + "_y.csv"
peaks = np.loadtxt(mzs_name)
names = np.loadtxt(names_name, dtype=str)
print(names)
y = np.genfromtxt(y_original_name, delimiter=",", skip_header=False)
x = images.reshape(images.shape[1:])

if analysis_names is not None:
    inside = np.in1d(names, analysis_names)
    names = names[inside]
    y = y[..., inside]


sample_size = 100
x, y = sample(x, y, sample_size)
regression = joblib.load(input_name)
out = regression.predict(x)
y = np.where(y>0, 1, 0)

if analysis_names is not None:
    out = out[..., inside]


if "ET&LO" in names:
    order = np.array([0, 1, 3, 4, 5, 2])
    names = names[order]
    out = out[..., order]
    y = y[..., order]


labels = np.argmax(out, axis=-1)
labels = np.repeat(np.arange(y.shape[-1]), sample_size)


distance_matrix = distance.squareform(distance.pdist(out, metric="sqeuclidean"))
# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
mds = TSNE(n_components=3, perplexity=50.0, metric="precomputed", random_state=0)
# mds = umap.UMAP(random_state=0)


fig = plt.figure()
ax = plt.axes(projection='3d')
# ax = plt.axes()

pos_array = mds.fit_transform(distance_matrix)
test = pos_array.T
uncertain_label = labels.max() + 1

print(gmm_name)
if gmm_name is None:
    means_init = np.array([[255 if i == j else 0 for j in range(y.shape[-1]) ] for i in range(y.shape[-1])])

    gmm = GaussianMixture(n_components=out.shape[-1], covariance_type="tied", means_init=means_init)
    clusters_gmm = gmm.fit(out)
else:
    print("Loading model")
    clusters_gmm = joblib.load(gmm_name)

labels = clusters_gmm.predict(out)
probas = clusters_gmm.predict_proba(out)
means = clusters_gmm.means_
reorganize_indices = np.argmax(means, axis=-1)
labels = reorganize_indices[labels]
labels[probas.max(axis=-1) < 0.999] = uncertain_label
print(clusters_gmm.means_, reorganize_indices)


# roc_ind = roc_indices(y, out, labels)
# labels[roc_ind] = uncertain_label

cm = plt.get_cmap("Set3")
array_colors = np.array(cm.colors)
k = np.array([0, 0, 0])
array_colors[uncertain_label, :] = k
cm = colors.ListedColormap(array_colors)

set_colors = cm(range(labels.max()+1))
colors = set_colors[labels]

# color_barycenter = cm(range(n_features))
# ax.scatter(*barycenter, marker="o", s=75, c=color_barycenter, ec="k")
ax.scatter(*test, marker="o", s=50, edgecolor='None', picker=True, c=colors)
handles = [plt.Rectangle((0, 0), 0, 0, color=cm(int(i)), label=name) for i, name in enumerate(names)]
ax.legend(handles=handles, title="Binders", loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
print(distance_matrix.shape)
print(regression.coef_.shape)
