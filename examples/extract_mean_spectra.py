import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import esmraldi.imageutils as imageutils
from esmraldi.msimagefly import MSImageOnTheFly
import SimpleITK as sitk

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
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output .csv files with stats")

args = parser.parse_args()


input_name = args.input
region_names = args.regions
output_name = args.output
normalization = float(args.normalization)

imzml = io.open_imzml(input_name)
spectra = io.get_spectra(imzml)
print(spectra.shape)
coordinates = imzml.coordinates
max_x = max(coordinates, key=lambda item:item[0])[0]
max_y = max(coordinates, key=lambda item:item[1])[1]
max_z = max(coordinates, key=lambda item:item[2])[2]
mzs = np.unique(np.hstack(spectra[:, 0]))
mzs = mzs[mzs>0]

norm_img = None
if normalization > 0:
    img_data = MSImageOnTheFly(spectra, coords=imzml.coordinates, tolerance=0.01)
    norm_img = img_data.get_ion_image_mzs(normalization, img_data.tolerance, img_data.tolerance)
    norm_flatten = norm_img.flatten()[:, np.newaxis]
    np.divide(spectra[:, 1, :], norm_flatten, where=norm_flatten!=0, out=spectra[:, 1, :])

regions = []
for region_name in region_names:
    region = read_image(region_name)
    indices_regions = np.ravel_multi_index(np.where(region > 0), shape, order='F')
    mean_spectra = sp.spectra_mean_centroided(spectra[indices_regions], mzs)
    plt.plot(mzs, mean_spectra)
    plt.show()
