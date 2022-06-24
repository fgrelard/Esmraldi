import argparse
import numpy as np

import esmraldi.imzmlio as io
import esmraldi.segmentation as segmentation
import SimpleITK as sitk
import matplotlib.pyplot as plt
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
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("--preprocess", help="Normalize to 0-255", action="store_true")
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()


input_name = args.input
mask_name = args.mask
region_names = args.regions
output_name = args.output
normalization = float(args.normalization)
is_normalized = args.preprocess


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


mask = read_image(mask_name)
regions = []
for region_name in region_names:
    region = read_image(region_name)
    regions.append(region)

n = len(np.where(mask>0)[0])

norm_img = None
if normalization > 0:
    norm_img = imageutils.get_norm_image(images, normalization, mzs)
    for i in range(image.shape[-1]):
        images[..., i] = imageutils.normalize_image(image[..., i], norm_img)

if is_normalized:
    images = io.normalize(images)


mzs_target = [837.549, 773.534, 869.554, #dispersion
           859.531372070312, 861.549438476562, 857.518188476562, #LB
           644.5015869,	788.5460815,	670.5178223, #LT
           286.9776, 296.0708]


indices = [np.abs(mzs - mz).argmin() for mz in mzs_target]
image = images[..., indices]

for i in range(image.shape[-1]):
    heterogeneities = []
    avs = []
    maxs = []
    for j, r in enumerate(regions):
        h, a, m = segmentation.heterogeneity_mask(image[..., i], r, 200)
        heterogeneities.append(h)
        avs.append(a)
        maxs.append(m)

    ind = np.argmax(heterogeneities)
    print("mzs", mzs_target[i], "Region", region_names[ind].split("crop ")[-1].split("-")[0], "hetero", heterogeneities[ind], avs[ind], maxs[ind])
