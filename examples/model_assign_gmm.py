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
import umap
from mpl_toolkits import mplot3d
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from matplotlib.patches import Ellipse
import esmraldi.fusion as fusion
from matplotlib import colors
from sklearn.cluster import DBSCAN


def compute_gmm(x, y, n_repetitions, names, add_class=False):
    n = len(names)
    means_init = np.array([[1 if i == j else 0 for j in range(n) ] for i in range(n)])
    if add_class:
        means_init = np.vstack([means_init, [1/len(names) for j in range(len(names))]])
    all_params = {}
    for i in range(n_repetitions):
        sample_size = 200
        x_curr, y_curr = sample(x, y, sample_size)
        regression = joblib.load(input_name)
        out = regression.predict(x_curr)
        y_curr = np.where(y_curr>0, 1, 0)
        if analysis_names is not None:
            # out = restrict_names(out, inside, sample_size)
            out = out[..., inside]

        if "ET&LO" in names:
            order = np.array([0, 1, 3, 4, 5, 2])
            names = names[order]
            out = out[..., order]
            y_curr = y_curr[..., order]

        k = out.shape[-1]
        if add_class:
            k += 1
        gmm = GaussianMixture(n_components=k, covariance_type="tied", means_init=means_init)
        clusters_gmm = gmm.fit(out)
        labels = clusters_gmm.predict(out)
        probas = clusters_gmm.predict_proba(out)
        means = clusters_gmm.means_
        params = get_properties(gmm)
        if len(all_params) == 0:
            all_params = {p: [getattr(gmm, p)] for p in params}
        all_params = {p: v + [getattr(gmm, p)] for p, v in all_params.items()}
    return all_params

def get_properties(model):
  return [i for i in model.__dict__ if i.endswith('_')]

def sample(x, y, sample_size):
    n_features = y.shape[-1]
    x_sampled = np.zeros((n_features*sample_size, x.shape[-1]))
    y_sampled = np.zeros((n_features*sample_size, y.shape[-1]))
    for i in range(n_features):
        # indices = np.where(y[..., i] > 0)[0]
        cond = (y[..., i]  > 0) & (np.all([y[..., j] == 0 for j in range(n_features) if j != i], axis=0))
        indices = np.where(cond)[0]
        np.random.shuffle(indices)
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

def restrict_names(out, inside, sample_size):
    labels = np.argmax(out, axis=-1)
    indices_analysis = np.where(inside)[0]
    sample_count = np.zeros(out.shape[1])
    L = []
    for i, line in enumerate(out):
        curr_label = labels[i]
        if labels[i] in indices_analysis and sample_count[curr_label] < sample_size:
            L.append(line)
            sample_count[curr_label] += 1
    return np.array(L)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input joblib")
parser.add_argument("--msi", help="MSI tif format")
parser.add_argument("--names", help="Names to analyze (default all)", nargs="+", type=str, default=None)
parser.add_argument("-o", "--output", help="Output GMM file (.joblib)")
parser.add_argument("--add_class", action="store_true")
args = parser.parse_args()

input_name = args.input
msi_name = args.msi
analysis_names = args.names
output_name = args.output
add_class = args.add_class

mzs_name = os.path.splitext(input_name)[0] + "_mzs.csv"
names_name = os.path.splitext(input_name)[0] + "_names.csv"
y_original_name = os.path.splitext(input_name)[0] + "_y.csv"
peaks = np.loadtxt(mzs_name)
names = np.loadtxt(names_name, dtype=str)
y = np.genfromtxt(y_original_name, delimiter=",", skip_header=False)

super_gmm = GaussianMixture(n_components=len(names), covariance_type="tied")

if analysis_names is not None:
    inside = np.in1d(names, analysis_names)
    names = names[inside]
    y = y[..., inside]

all_params = {}

if os.path.isdir(msi_name):
    for root, dirs, files in os.walk(msi_name):
        for f in files:
            if f == "train.tif":
                current_file = f
            else:
                continue
            image_itk = sitk.ReadImage(root + os.path.sep + current_file)
            images = sitk.GetArrayFromImage(image_itk).T
            x = images.reshape(images.shape[1:])
            params = compute_gmm(x, y, 5, names, add_class)
            if len(all_params) == 0:
                all_params = {p: v for p, v in params.items()}
            else:
                all_params = {p: v + params[p] for p, v in all_params.items()}
else:
    image_itk = sitk.ReadImage(msi_name)
    images = sitk.GetArrayFromImage(image_itk).T
    x = images.reshape(images.shape[1:])
    all_params = compute_gmm(x, y, 10, names)

for p, v in all_params.items():
    setattr(super_gmm, p, np.mean(v, axis=0))

joblib.dump(super_gmm, output_name)
