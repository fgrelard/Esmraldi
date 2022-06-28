import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import esmraldi.imageutils as imageutils
from esmraldi.msimagefly import MSImageOnTheFly
import SimpleITK as sitk
import xlsxwriter

import os

from skimage.color import rgb2gray
import matplotlib.pyplot as plt

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

workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})

name = "No norm"
if normalization > 0:
    name = str(normalization)

worksheet = workbook.add_worksheet(name)

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
    for i, intensities in enumerate(spectra[:, 1]):
        if norm_flatten[i] != 0:
            new_intensities = intensities / norm_flatten[i]
        else:
            new_intensities = np.zeros_like(intensities)
        spectra[i, 1] = new_intensities




regions = []
for i, region_name in enumerate(region_names):
    region = read_image(region_name)
    indices_regions = np.ravel_multi_index(np.where(region > 0), (max_x, max_y), order='F')
    curr_spectra = spectra[indices_regions]
    curr_mzs, intensities = sp.realign_mean_spectrum(mzs, curr_spectra[:, 1], curr_spectra[:, 0], 14, is_ppm=True)
    mean_spectra = sp.spectra_mean_centroided(curr_spectra, mzs)
    if i==0:
        worksheet.write_column(1, 0, curr_mzs, header_format)
    print(mzs.shape, curr_mzs.shape, intensities.shape)
    plt.plot(curr_mzs, intensities)
    plt.plot(mzs, mean_spectra)
    plt.show()
    name = os.path.splitext(os.path.basename(region_name))[0]
    worksheet.write(0, i+1, name)
    worksheet.write_column(1, i+1, intensities)

worksheet.freeze_panes(1, 1)
workbook.close()
