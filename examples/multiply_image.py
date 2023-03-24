import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import esmraldi.imageutils as imageutils
import SimpleITK as sitk
import matplotlib.pyplot as plt
import xlsxwriter
import os

from skimage.color import rgb2gray


def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    mask = rgb2gray(mask)
    mask = mask.T
    return mask


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output .xlsx files with stats")

parser.add_argument("--mzs", help="Ions to combine", nargs="+", type=float)
args = parser.parse_args()

input_name = args.input
output_name = args.output
normalization = float(args.normalization)
mzs_target = args.mzs

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
    mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv")


if normalization > 0:
    norm_img = imageutils.get_norm_image(images, normalization, mzs)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[...,i], norm_img)


indices_mz = [np.abs(mzs - mz).argmin() for mz in mzs_target]
images = images[..., indices_mz]
mzs = mzs[indices_mz]
out_image = np.prod(images, axis=-1)
plt.imshow(out_image)
plt.show()
sitk.WriteImage(sitk.GetImageFromArray(out_image.T), output_name)
