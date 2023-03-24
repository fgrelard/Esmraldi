import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LinearRegression
import esmraldi.fusion as fusion
import esmraldi.imzmlio as io
import SimpleITK as sitk
import os
import scipy.spatial.distance as distance
from sklearn.metrics import confusion_matrix, roc_auc_score
import esmraldi.imageutils as imageutils

def extract_prediction(reg, x, y, i):
    result = np.sum(x * reg.coef_[i], axis=-1) + reg.intercept_[i]
    target = y[..., i]
    target = np.where(target > 0, 1, 0)
    region_bool = target.ravel().astype(np.bool_)
    fpr, tpr, thresholds = fusion.roc_curve(region_bool, result.ravel(), is_weighted=False)
    nb_ones = np.count_nonzero(region_bool)
    nb_zeroes = np.count_nonzero(~region_bool)
    cutoff, ind = fusion.cutoff_half_tpr(fpr, tpr, thresholds, nb_zeroes, nb_ones, return_index=True)
    t = thresholds[ind]
    result = np.where(result > t, 1, 0)
    tn, fp, fn, tp = confusion_matrix(result.flatten(), target.flatten()).ravel()
    se = tp / (tp+fn)
    return se, target, result

def extract_regions(image):
    max_im = int(image.max())
    regions = np.zeros(image.shape + (max_im+1,))
    for i in range(max_im+1):
        condition = (image == i)
        mask = np.where(condition, 255, 0).astype(np.uint8)
        regions[..., i] = mask
    return regions

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML")
parser.add_argument("-t", "--target", help="Target image(s)", nargs="+")
parser.add_argument("-c", "--clusters", help="Cluster image", default=None)
parser.add_argument("--convert_to_spectral", action="store_true")
parser.add_argument("--revert", action="store_true")
parser.add_argument("--normalization", help="Normalization", default=None)
parser.add_argument("-o", "--output", help="Output dir")
args = parser.parse_args()

msi_name = args.input
target_names = args.target
cluster_name = args.clusters
convert_to_spectral = args.convert_to_spectral
is_revert = args.revert
normalization = args.normalization
outdir = args.output

if msi_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(msi_name)
    spectra = io.get_spectra(imzml)
    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]
    shape = (max_x, max_y, max_z)

    full_spectra = io.get_full_spectra(imzml)
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
    print(len(mzs))
    images = io.get_images_from_spectra(full_spectra, shape)
else:
    image_itk = sitk.ReadImage(msi_name)
    images = sitk.GetArrayFromImage(image_itk).T
    if os.path.exists(os.path.splitext(msi_name)[0] + ".csv"):
        mzs = np.loadtxt(os.path.splitext(msi_name)[0] + ".csv", encoding="utf-8-sig")

targets = []
for target_name in target_names:
    target = sitk.GetArrayFromImage(sitk.ReadImage(target_name)).T
    targets.append(target)

targets = np.array(targets)
targets = np.transpose(targets, (1, 2, 0))

if convert_to_spectral:
    images = extract_regions(images)

if normalization != None:
    try:
        normalization = float(normalization)
    except:
        pass
    norm_img = imageutils.get_norm_image(images, normalization, mzs)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[...,i], norm_img)

if is_revert:
    tmp = targets.copy()
    targets = images
    images = tmp

images = io.normalize(images) / 255.0
targets = io.normalize(targets) / 255.0

X = fusion.flatten(images, is_spectral=True).T
y = fusion.flatten(targets, is_spectral=True).T

reg = LinearRegression().fit(X, y)

if cluster_name is not None:
    clusters = sitk.GetArrayFromImage(sitk.ReadImage(cluster_name)).T
    clusters = extract_regions(clusters)
    clusters = io.normalize(clusters) / 255.0
    cluster_flat = fusion.flatten(clusters, is_spectral=True).T
    reg2 = LinearRegression().fit(cluster_flat, y)

coef = reg.coef_

for i in range(coef.shape[0]):
    score, target, result = extract_prediction(reg, images, targets, i)
    fig, ax = plt.subplots(1, 2)
    if is_revert and convert_to_spectral:
        name = "Cluster" + str(i)
    else:
        name = os.path.splitext(os.path.basename(target_names[i]))[0]
    outname = outdir + os.path.sep + name
    sitk.WriteImage(sitk.GetImageFromArray(result.T.astype(np.uint8))*255, outname + ".tif")
    # plt.suptitle(name  + " {:.3f}".format(score))
    # ax[0].imshow(target)
    # ax[1].imshow(result)
    # [axi.set_axis_off() for axi in ax.ravel()]
    # plt.savefig(outname + ".png")
