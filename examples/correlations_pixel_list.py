import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import esmraldi.fusion as fusion
import esmraldi.segmentation as segmentation
import pandas as pd
import SimpleITK as sitk
import scipy.spatial.distance as distance
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, TSNE
import re
import esmraldi.imageutils as imageutils
from matplotlib.colors import ListedColormap
from skimage.color import rgb2gray

def display_matrix(distance_matrix, region_names, test_region_names):
    mds = MDS(n_components=2, dissimilarity="precomputed")
    pos_array = mds.fit_transform(distance_matrix).T
    colors = ["b"] * len(region_names)
    names = region_names
    if test_region_names is not None:
        names += test_region_names
        colors += ["y"] * len(test_region_names)
    plt.scatter(*pos_array, marker="o", s=50, c=colors, edgecolor=None, picker=True)
    for k, p in enumerate(pos_array.T):
        name = names[k]
        name = os.path.splitext(os.path.basename(name))[0]
        plt.text(*p, name)
    plt.show()

def unique_region_names(region_names):
    unique_names = []
    names = []
    for name in region_names:
        name = os.path.splitext(os.path.basename(name))[0]
        names.append(re.sub("\d", "", name))
    unique, indices = np.unique(names, return_inverse=True)
    indices = np.array([np.where(indices == i)[0].astype(int) for i in range(len(unique))])
    return unique, indices

def average_spectra(all_spectra, unique_names, unique_indices):
    av_spectra = []
    for i, name in enumerate(unique_names):
        current_indices = unique_indices[i]
        average = np.mean(all_spectra[current_indices], axis=0)
        av_spectra.append(average)
    return np.array(av_spectra)


def extract_coordinates(array):
    coords = array[:, 0].astype(int)
    inside = array[:, -1]
    return coords[inside == 1]





parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-r", "--regions", help="Pixel lists", nargs="+", type=str)
parser.add_argument("-t", "--test_regions", help="Test regions", nargs="+", type=str)
parser.add_argument("-a", "--aucs", help="AUC values", default=None)
parser.add_argument("--auc_threshold", help="AUC threshold to restrict peaks", default=0)
parser.add_argument("-s", "--subtract", help="Whether to subtract the mean spectra from matrix peaks", action="store_true")
parser.add_argument("-n", "--normalization", help="Whether to normalize with TIC or not", action="store_true")
parser.add_argument("--clustering", help="Perform clustering w.r.t average spectra", action="store_true")
parser.add_argument("-o", "--output", help="Output image")

args = parser.parse_args()

input_name = args.input
region_names = args.regions
test_region_names = args.test_regions
auc_name = args.aucs
auc_threshold = float(args.auc_threshold)
normalization = args.normalization
is_subtract = args.subtract
clustering = args.clustering
outname = args.output

if input_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
    spectra = io.get_spectra(imzml)
    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]
    shape = (max_x, max_y, max_z)
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
else:
    image_itk = sitk.ReadImage(input_name)
    images = sitk.GetArrayFromImage(image_itk).T
    mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv")

if normalization:
    print("normalization")
    spectra = sp.normalization_tic(spectra)

if  auc_name is not None:
    table_aucs = pd.read_excel(auc_name, header=None).values
    peaks = table_aucs[0, 1:]
    aucs = table_aucs[1:-1, 1:]
    cond = (np.amax(aucs, axis=0) > auc_threshold) | (np.amin(aucs, axis=0) < 1 - auc_threshold)
    peaks = peaks[cond]
    spectra = sp.realign_generic(spectra, peaks, 14, is_ppm=True)
    lengths = [len(spectra[i, 0]) for i in range(spectra.shape[0])]
    mzs = peaks

coordinates = []
coordinates_test = []
coordinates_matrix = np.array([], dtype=int)
indices_matrix = []
indices_other = []

for i, region_name in enumerate(region_names):
    array = pd.read_csv(region_name, delimiter="\t", header=None).values
    if "Matrix" in region_name:
        indices_matrix.append(i)
        coordinates_matrix = np.concatenate((coordinates_matrix, array[:, 0]), dtype=int)
    else:
        indices_other.append(i)
    c = extract_coordinates(array)
    coordinates.append(c)

restricted_spectra = spectra[coordinates_matrix]
mean_spectra_matrix = sp.spectra_mean_centroided(restricted_spectra, mzs)

all_spectra = sp.extract_mean_spectra_coordinates(spectra, coordinates, mzs, is_subtract, mean_spectra_matrix)

unique_names, unique_indices = unique_region_names(region_names)
av_spectra = average_spectra(all_spectra, unique_names, unique_indices)

if test_region_names is not None:
    for i, region_name in enumerate(test_region_names):
        array = pd.read_csv(region_name, delimiter="\t", header=None).values
        c = extract_coordinates(array)
        coordinates_test.append(c)

spectra_test = sp.extract_mean_spectra_coordinates(spectra, coordinates_test, mzs, is_subtract, mean_spectra_matrix)

if test_region_names is not None:
    all_spectra = np.concatenate((all_spectra, spectra_test))

distance_matrix = distance.squareform(distance.pdist(all_spectra, metric="cosine"))

outname_distance = os.path.splitext(outname)[0] + "_matrix.png"
imageutils.export_figure_matplotlib(outname_distance, distance_matrix, cmaps=["RdBu"], plt_show=True)


if clustering:
    print("Clustering")
    imsize = max_x*max_y*max_z
    full_spectra = io.get_full_spectra_sparse(spectra, imsize)
    images = io.get_images_from_spectra(full_spectra, shape)
    label_image, opacity_image = segmentation.clustering_with_centers(images, av_spectra, is_subtract, "cosine", mean_spectra_matrix, radius=1)
    blacks = np.zeros_like(label_image.T)
    outname_fig = os.path.splitext(outname)[0] + ".png"
    colors = np.array([[0.067, 0.114, 0.669], [0.608, 0.125, 0.093], [1,1,1], [1, 0.671, 0], [0,0,0]])
    newcmp = ListedColormap(colors)
    imageutils.export_figure_matplotlib(outname_fig, blacks, label_image.T, cmaps=["gray", newcmp], alpha=1-opacity_image.T, plt_show=True)
    print(images.shape)

for i, test in enumerate(spectra_test):
    index = np.argmin([distance.cosine(test, reference) for reference in av_spectra])
    print(test_region_names[i], unique_names[index])


out_spectra = np.vstack((mzs, av_spectra))
np.savetxt(outname, out_spectra.T, delimiter=",", header="mzs, " + ", ".join(unique_names))


for i, s in enumerate(av_spectra):
    plt.figure()
    plt.stem(mzs, s)
    plt.title(unique_names[i])
    plt.xlabel("m/z")
    plt.ylabel("Abundance (TIC normalized)")
    outname_spectra = os.path.splitext(outname)[0] + "_spectra" + str(i) + ".png"
    plt.savefig(outname_spectra)

plt.figure()
display_matrix(distance_matrix, region_names, test_region_names)
plt.show()

print(distance_matrix.shape)
