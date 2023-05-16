import joblib
import argparse
import numpy as np
import os
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.preprocessing import StandardScaler
import esmraldi.imzmlio as io
import esmraldi.utils as utils
import esmraldi.imageutils as imageutils
import esmraldi.fusion as fusion
import esmraldi.spectraprocessing as sp
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from skimage.filters import threshold_multiotsu
from skimage.morphology import opening, closing, erosion, dilation
from scipy.ndimage import binary_fill_holes

def normalize_flatten(spectra, coordinates, shape, normalization_tic=True, normalization_minmax=True):
    if normalization:
        print("normalization")
        spectra = sp.normalization_tic(spectra, inplace=True)
    full_spectra = io.get_full_spectra_dense(spectra, coordinates, shape)
    images = io.get_images_from_spectra(full_spectra, shape)
    if normalization_minmax:
        images = io.normalize(images)
    images = images.astype(np.float128) / 255.0
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
parser.add_argument("-o", "--output", help="Output files")
parser.add_argument("--names", help="Names to analyze (default all)", nargs="+", type=str, default=None)
parser.add_argument("--gmm", help="GMM model (.joblib)", default=None)
parser.add_argument("--proba", help="Proba", default=0)
parser.add_argument("--rank", help="Nth rank for prediction", default=1)
args = parser.parse_args()

input_name = args.input
target_name = args.target
normalization = args.normalization
outname = args.output
analysis_names = args.names
gmm_name = args.gmm
proba = float(args.proba)
rank = int(args.rank)

mzs_name = os.path.splitext(input_name)[0] + "_mzs.csv"
names_name = os.path.splitext(input_name)[0] + "_names.csv"
peaks = np.loadtxt(mzs_name)
names = np.loadtxt(names_name, dtype=str)


# cm = plt.get_cmap("terrain")
cm = plt.get_cmap("Dark2")
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(analysis_names))
spectra, mzs, shape, coords = read_imzml(target_name, normalization)


indices = indices_peaks(mzs, peaks)

target_im = normalize_flatten(spectra, coords, shape, normalization_tic=normalization, normalization_minmax=True)
blank_image = np.zeros((target_im.shape[0], 1))
target_im = np.hstack((target_im, blank_image))
target_im = target_im[..., indices]

# Load data from file
regression = joblib.load(input_name)
coef = regression.coef_
if coef.shape[0] < coef.shape[1]:
    coef = coef.T


data = np.column_stack((peaks, coef))
peaks_coef_name = os.path.dirname(outname) + os.path.sep + "coef.csv"
np.savetxt(peaks_coef_name, data, header="mzs,"+",".join(names), delimiter=",", comments="")


# scaler = StandardScaler()
# target_im = scaler.fit_transform(target_im)
out = regression.predict(target_im)
# separation = np.array([len(s.split("_")) for s in names])
# end_pigments = np.where(separation==2)[0][-1]
# end_binders = np.where(separation==1)[0][-1]
# names_binders = names[end_pigments+1:end_binders+1]
# out_binders = out[..., end_pigments+1:end_binders+1]

if analysis_names is not None:
    inside = np.in1d(names, analysis_names)
    names = names[inside]
    out = out[..., inside]

if "ET&LO" in names:
    order = np.array([0, 1, 3, 4, 2])
    names = names[order]
    out = out[..., order]


labels = np.argmax(out, axis=-1)
labels = np.argsort(out, axis=-1)[..., -rank]

array_colors = np.array(cm.colors)
names_array = np.array(analysis_names)
ind_mat = np.where((names_array == "Matrix") | (names_array == "Tape"))[0]
black = np.array([0.15, 0.15, 0.15])
array_colors[ind_mat, :] = black
cm = colors.ListedColormap(array_colors)

if gmm_name is not None:
    gmm = joblib.load(gmm_name)
    labels = gmm.predict(out)
    print(labels.shape)
    probas = gmm.predict_proba(out)
    labels = np.argsort(probas, axis=-1)[..., -rank]
    uncertain_label = len(analysis_names)
    if proba > 0:
        labels[probas.max(axis=-1) < proba] = uncertain_label
    label_image = np.reshape(labels, shape[:-1]).T
    array_colors = np.array(cm.colors)
    gray = np.array([0.3, 0.3, 0.3])
    array_colors[uncertain_label, :] = gray
    cm = colors.ListedColormap(array_colors)
    plt.imshow(label_image, cmap=cm, vmin=0, vmax=cm.N, interpolation="nearest")

    names = np.append(names, ["Uncertain"])
else:
    min_value, max_value = np.amin(out, axis=0), np.amax(out, axis=0)
    min_value, max_value = np.amin(out), np.amax(out)
    opacity = (out - min_value) / (max_value - min_value)
    opacity = np.take_along_axis(opacity, labels[:, None], axis=-1)

    opacity_image = np.reshape(opacity, shape[:-1]).T.astype(np.float64)
    label_image = np.reshape(labels, shape[:-1]).T

    blacks = np.zeros_like(label_image)
    plt.imshow(blacks, cmap="gray")
    plt.imshow(label_image, cmap=cm, vmin=0, vmax=cm.N, alpha=opacity_image,  interpolation="nearest")

handles = [plt.Rectangle((0, 0), 0, 0, color=cm(int(i)), label=name) for i, name in enumerate(analysis_names)]
# handles = [plt.Rectangle((0, 0), 0, 0, color=cm(norm(i+1)), label=name) for i, name in enumerate(analysis_names)]

plt.legend(handles=handles, title="Binders", loc='center left', bbox_to_anchor=(1, 0.5))
plt.axis("off")
plt.savefig(outname, bbox_inches='tight', pad_inches = 0)
# plt.show()


# for i in range(coef.shape[-1]):
#     plt.title(names[i])
#     score = out[..., i]
#     im_target = np.reshape(score, shape[:-1]).T
#     plt.imshow(im_target,vmax=255)
#     plt.figure()
#     plt.plot(peaks, coef[..., i])
#     print(peaks[coef[...,i] != 0])
#     plt.show()
