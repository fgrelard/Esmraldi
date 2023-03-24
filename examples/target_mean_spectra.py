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


def extract_coordinates(array):
    coords = array[:, 0]
    inside = array[:, -1]
    return coords[inside == 1]

def unique_region_names(region_names):
    unique_names = []
    names = []
    for name in region_names:
        name = os.path.splitext(os.path.basename(name))[0]
        names.append(re.sub("\d", "", name))
    unique, indices = np.unique(names, return_inverse=True)
    indices = np.array([np.where(indices == i)[0].astype(int) for i in range(len(unique))])
    return unique, indices

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

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Target .imzML")
parser.add_argument("-t", "--target_regions", help="Target regions", nargs="+", type=str)
parser.add_argument("-m", "--mean_spectra", help="Reference mean spectra (.csv)")
parser.add_argument("-s", "--subtract", help="Whether to subtract the mean spectra from matrix peaks", action="store_true")
parser.add_argument("--clustering", help="Perform clustering w.r.t average spectra", action="store_true")
parser.add_argument("-n", "--normalization", help="Whether to normalize with TIC or not", action="store_true")
parser.add_argument("-o", "--output", help="Output image")

args = parser.parse_args()

input_name = args.input
target_region_names = args.target_regions
mean_spectra_name = args.mean_spectra
normalization = args.normalization
is_subtract = args.subtract
normalization = args.normalization
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

data = pd.read_csv(mean_spectra_name, delimiter=",")

peaks = data.values[:, 0]
av_spectra = data.values[:, 1:].T
target_names = data.columns.values[1:].tolist()
reference_names = [os.path.splitext(os.path.basename(name))[0] for name in target_region_names]
print(reference_names, target_names)

if normalization:
    print("normalization")
    spectra = sp.normalization_tic(spectra)

spectra = sp.realign_generic(spectra, peaks, 14, is_ppm=True)

mzs = peaks
coordinates = []
coordinates_matrix = np.array([], dtype=int)
for i, region_name in enumerate(target_region_names):
    array = pd.read_csv(region_name, delimiter="\t", header=None).values
    if "Matrix" in region_name:
        coordinates_matrix = np.concatenate((coordinates_matrix, array[:, 0]), dtype=int)
    c = extract_coordinates(array)
    coordinates.append(c)

restricted_spectra = spectra[coordinates_matrix]
mean_spectra_matrix = sp.spectra_mean_centroided(restricted_spectra, mzs)

all_spectra = sp.extract_mean_spectra_coordinates(spectra, coordinates, mzs, is_subtract, mean_spectra_matrix)
all_spectra = np.concatenate((all_spectra, av_spectra))
print(all_spectra.shape)

distance_matrix = distance.squareform(distance.pdist(all_spectra, metric="cosine"))

outname_distance = os.path.splitext(outname)[0] + "_matrix.png"
imageutils.export_figure_matplotlib(outname_distance, distance_matrix, cmaps=["RdBu"], plt_show=True)

plt.figure()
display_matrix(distance_matrix, reference_names, target_names)
plt.show()

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
