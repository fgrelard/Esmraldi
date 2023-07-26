import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse
import esmraldi.imzmlio as io
import esmraldi.imageutils as imageutils
from esmraldi.sliceviewer import SliceViewer
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input image")
parser.add_argument("-f", "--first", help="First list")
parser.add_argument("-s", "--second", help="Second list")
parser.add_argument("-o", "--output", help="Output name")
parser.add_argument("-t", "--thresholds", help="Thresholds", default=None)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=None)
args = parser.parse_args()

input_name = args.input
first_name = args.first
second_name = args.second
output_name = args.output
threshold_name = args.thresholds
normalization = args.normalization

if input_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
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
    image_itk = sitk.ReadImage(input_name)
    images = sitk.GetArrayFromImage(image_itk).T
    try:
        mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv", encoding="utf-8-sig")
    except ValueError as ve:
        mzs = np.genfromtxt(os.path.splitext(input_name)[0] + ".csv", encoding="utf-8-sig", dtype='str')

first = np.loadtxt(os.path.splitext(first_name)[0] + ".csv", encoding="utf-8-sig")
second = np.loadtxt(os.path.splitext(second_name)[0] + ".csv", encoding="utf-8-sig")



union = np.union1d(first, second)
common = np.intersect1d(first, second)
diff = np.setdiff1d(union, common)

diff_indices = [np.abs(mzs - mz).argmin() for mz in diff]
common_indices = [np.abs(mzs - mz).argmin() for mz in common]


root, ext = os.path.splitext(output_name)
if common.size:
    io.to_tif(images[..., common_indices].T, common, root + "_common.tif")

if diff.size:
    io.to_tif(images[..., diff_indices].T, diff, root + "_diff.tif")

if threshold_name is not None:
    data = np.genfromtxt(threshold_name, encoding="utf-8-sig", delimiter=",", names=True, dtype=float)
    thresholds = data["thresholds"]

    thresholded_image = images.copy()
    if normalization != None:
        try:
            normalization = float(normalization)
        except:
            pass
        norm_img = imageutils.get_norm_image(thresholded_image, normalization, mzs)
        for i in range(thresholded_image.shape[-1]):
            thresholded_image[..., i] = imageutils.normalize_image(thresholded_image[...,i], norm_img)
    thresholded_image = io.normalize(thresholded_image)
    thresholds_quantiles = np.percentile(thresholded_image, thresholds, axis=-1)
    for i in range(thresholded_image.shape[-1]):
        curr_img = thresholded_image[..., i]
        thresh = np.percentile(curr_img, thresholds[i])
        curr_img[curr_img < thresh] = 0
        thresholded_image[..., i] = curr_img

    if common.size:
        io.to_tif(thresholded_image[..., common_indices].T, common, root + "_common_binary.tif")
    if diff.size:
        io.to_tif(thresholded_image[..., diff_indices].T, diff, root + "_diff_binary.tif")

print(common.size)
print(diff)
print(diff.size)
