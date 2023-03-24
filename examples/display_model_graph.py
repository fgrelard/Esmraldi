import joblib
import argparse
import numpy as np
import os
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

def normalize_flatten(spectra, coordinates, shape, normalization=True):
    if normalization:
        print("normalization")
        spectra = sp.normalization_tic(spectra, inplace=True)
    full_spectra = io.get_full_spectra_dense(spectra, coordinates, shape)
    images = io.get_images_from_spectra(full_spectra, shape)
    images = io.normalize(images)
    image_flatten = fusion.flatten(images, is_spectral=True).T
    return image_flatten

def read_imzml(input_name, normalization):
    if input_name.lower().endswith(".imzml"):
        imzml = io.open_imzml(input_name)
        spectra = io.get_spectra(imzml)
        coordinates = imzml.coordinates
        max_x = max(coordinates, key=lambda item:item[0])[0]
        max_y = max(coordinates, key=lambda item:item[1])[1]
        max_z = max(coordinates, key=lambda item:item[2])[2]
        shape = (max_x, max_y, max_z)
        mzs = np.unique(np.hstack(spectra[:, 0]))
        mzs = mzs[mzs>0]
    return spectra, mzs, shape, imzml.coordinates

def indices_peaks(peaks, other_peaks):
    indices = utils.indices_search_sorted(other_peaks, peaks)
    print(len(indices), len(other_peaks), len(peaks))
    current_step = 14 * other_peaks / 1e6
    indices_ppm = np.abs(peaks[indices] - other_peaks) < current_step
    indices[~indices_ppm] = -1
    return indices

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input json")
parser.add_argument("-t", "--target", help="Target .imzML")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", action="store_true")
parser.add_argument("--sample_size", help="Sample size (in pixels)", default=0)
args = parser.parse_args()

input_name = args.input
target_name = args.target
normalization = args.normalization

mzs_name = os.path.splitext(input_name)[0] + "_mzs.csv"
names_name = os.path.splitext(input_name)[0] + "_names.csv"
x_original_name = os.path.splitext(input_name)[0] + "_train.csv"
y_original_name = os.path.splitext(input_name)[0] + "_y.csv"
peaks = np.loadtxt(mzs_name)
names = np.loadtxt(names_name, dtype=str)
x_original = np.genfromtxt(x_original_name, delimiter=",", skip_header=False)
y_original = np.genfromtxt(y_original_name, delimiter=",", skip_header=False)

spectra, mzs, shape, coords = read_imzml(target_name, normalization)
sample_size = int(args.sample_size)

indices = indices_peaks(mzs, peaks)

target_im = normalize_flatten(spectra, coords, shape, normalization=normalization)

blank_image = np.zeros((target_im.shape[0], 1))
print(blank_image.shape, target_im.shape)
target_im = np.hstack((target_im, blank_image))
print(target_im.shape)
target_im = target_im[..., indices]

regression = joblib.load(input_name)
out = regression.predict(target_im)

separation = np.array([len(s.split("_")) for s in names])
end_pigments = np.where(separation==2)[0][-1]
end_binders = np.where(separation==1)[0][-1]
names_binders = names[end_pigments+1:end_binders+1]
out_binders = out[..., end_pigments+1:end_binders+1]

x_binders = x_original[..., end_pigments+1:end_binders+1]
y_binders = y_original[..., end_pigments+1:end_binders+1]


out = out_binders
x = x_binders
y = y_binders

test_length = out.shape[0]
train_length = x_original.shape[0]
n_features = out.shape[-1]
n_binders = end_binders-end_pigments

print(test_length, train_length, n_features, n_binders)
sampled_out = np.zeros((test_length + train_length, n_features))
sampled_out[:test_length, ...] = out
if sample_size:
    # sampled_out = np.zeros((n_features*(sample_size+1), n_features))
    sampled_out = np.zeros((n_features*(sample_size*2+1), n_features))
    for i in range(n_features):
        indices = np.argsort(out[..., i])[::-1]
        indices = indices[:sample_size]
        sampled_out[i*sample_size:(i+1)*sample_size, ...] = out[indices, ...]
    for i in range(n_features):
        indices = np.where(y[..., i] > 0)[0]
        indices = indices[:sample_size]
        x_indices = x[indices, ...]
        sampled_out[(i+n_features)*sample_size:(i+n_features+1)*sample_size, ...] = x_indices
    labels = np.argmax(sampled_out, axis=-1)

for i in range(n_features):
    indices = np.argwhere(y[..., i] > 0)
    x_indices = x[indices, ...]
    barycenter = np.median(x_indices, axis=0)
    sampled_out[-n_features+i, ...] = barycenter

print(sampled_out.shape)

cm= plt.get_cmap("Set3")
set_colors = cm(range(labels.max()+1))

distance_matrix = distance.squareform(distance.pdist(sampled_out, metric="sqeuclidean"))
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
# mds = TSNE(n_components=2, perplexity=50.0, metric="precomputed", random_state=0)
# mds = umap.UMAP(random_state=0)


fig = plt.figure()
# ax = plt.axes(projection='3d')
ax = plt.axes()

pos_array = mds.fit_transform(distance_matrix)
separation = n_features
separation = pos_array.shape[0]//2+n_features//2
test = pos_array[:-separation,...].T
train = pos_array[-separation:-n_features,...].T

print(test.shape, train.shape, separation, n_features)
barycenter = pos_array[-n_features:,...].T
# color_train = cm(range(n_features))
colors = set_colors[labels[:-separation]]
color_train = set_colors[labels[-separation:-n_features]]/1.2
color_barycenter = cm(range(n_features))
print(train.shape[1])
ax.scatter(*test, marker="o", s=50, edgecolor='None', picker=True, c=colors)
# ax.scatter(*train, marker="o", s=30, c=color_train)
# ax.scatter(*barycenter, marker="o", s=75, c=color_barycenter, ec="k")
plt.show()
print(distance_matrix.shape)
print(regression.coef_.shape)
