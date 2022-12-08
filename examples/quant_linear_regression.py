import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import esmraldi.imageutils as imageutils
import esmraldi.utils as utils
from esmraldi.msimagefly import MSImageOnTheFly
import SimpleITK as sitk
import xlsxwriter
import pandas as pd
import os

from scipy import stats

from skimage import measure
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from esmraldi.peakdetectionmeanspectrum import PeakDetectionMeanSpectrum

def compute_average_from_region(curr_spectra, mzs, step, peaks=None):
    if peaks is not None:
        curr_mzs = peaks
        out_spectra = sp.realign_generic(curr_spectra, peaks, step, is_ppm=True)
        out_spectra = io.get_full_spectra_sparse(out_spectra, out_spectra.shape[0])
        curr_intensities = out_spectra[:, 1]
        if not isinstance(curr_intensities, np.ndarray):
            curr_intensities = curr_intensities.todense()
        actual_mzs = np.unique(np.hstack(out_spectra[:, 0]))
        actual_mzs = actual_mzs[actual_mzs > 0]
        indices = utils.indices_search_sorted(actual_mzs, curr_mzs)
        actual_intensities = np.mean(curr_intensities, axis=0)
        actual_stds = np.std(curr_intensities, axis=0)
        intensities = np.zeros_like(curr_mzs)
        stds = np.zeros_like(curr_mzs)
        intensities[indices] = actual_intensities
        stds[indices] = actual_stds
        n = np.repeat(len(indices_regions), len(curr_mzs))
    else:
        curr_mzs, intensities, stds, n = sp.realign_mean_spectrum(mzs, curr_spectra[:, 1], curr_spectra[:, 0], step, is_ppm=True, return_stats=True)
    return intensities, stds,  n


def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask

def plot_reg(x, y, ax, name):
    res = stats.linregress(x,y)
    ax.set_title(name)
    ax.plot(x,y, "wo")
    # ax = plt.gca()
    # plt.errorbar(x, y, error, None, "o", ecolor="w", capsize=2)
    ax.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
    plt.text(0.8, 0.9, "R2="+"{:.3f}".format(res.rvalue), color="w", transform = ax.transAxes)
    plt.text(0.05, 0.9, "y="+"{:.3e}".format(res.slope)+"x " + "{:.3e}".format(res.intercept), color="w", transform = ax.transAxes)
    ax.set_xlabel("Concentration (µg/g)")
    ax.set_ylabel("Mean abundance")
    return res


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-m", "--mask", help="Mask with different subregions used to compute linear regression (.tif)")
parser.add_argument("--peak_list", help="File with quantitative ion mz, names and concentrations listed (.xlsx)", default=None)
parser.add_argument("--tissue_regions", help="Tissue regions from which average intensity and concentrations are derived (.tif)", nargs="+", type=str)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z.\n\
* 0: no norm;\n\
*-1: TIC;\n\
*>0: mz of the standard ion", default=0)
parser.add_argument("-o", "--output", help="Output .xlsx files with stats")
args = parser.parse_args()

plt.style.use('dark_background')

input_name = args.input
mask_name = args.mask
tissue_regions_name = args.tissue_regions
output_name = args.output
normalization = float(args.normalization)
peak_list = args.peak_list

region = read_image(mask_name)
all_labels = measure.label(region, background=0)

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
elif normalization == -1:
    name = "tic"
name += "_quant"

name2 = name + "_regions"
worksheet = workbook.add_worksheet(name)
worksheet2 = workbook.add_worksheet(name2)

imzml = io.open_imzml(input_name)
spectra = io.get_spectra(imzml)
print(spectra.shape)
coordinates = imzml.coordinates
max_x = max(coordinates, key=lambda item:item[0])[0]
max_y = max(coordinates, key=lambda item:item[1])[1]
max_z = max(coordinates, key=lambda item:item[2])[2]
mzs = np.unique(np.hstack(spectra[:, 0]))
mzs = mzs[mzs>0]

step = 14


peaks = None

if peak_list is not None:
    data = pd.read_excel(peak_list)
    peaks = np.array(data.mz)
    names = np.array(data.names)
    concentrations = np.array(data)[:, 2:]

print("Normalization")
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
elif normalization == -1:
    spectra = sp.normalization_tic(spectra)

print("Stats")

all_intensities = []
stds = []
for i in range(1, all_labels.max()+1):
    indices_regions = np.ravel_multi_index(np.where(all_labels == i), (max_x, max_y), order='F')
    curr_spectra = spectra[indices_regions]
    intensities, std, n = compute_average_from_region(curr_spectra, mzs, step, peaks)
    all_intensities.append(intensities)
    stds.append(std)
    step = 3
    curr_i = i-1
    if curr_i==0:
        curr_mzs = peaks if peaks is not None else mzs
        worksheet.write_column(1, 0, ["mean", "stddev", "slope", "intercept"], header_format)


    # plt.plot(curr_mzs, intensities)
    # plt.plot(mzs, mean_spectra)
    # plt.show()

all_intensities = np.array(all_intensities)
stds = np.array(stds)

regression_coefficients = []
fig, ax = plt.subplots(1, len(peaks), figsize=(17, 10))
for i in range(all_intensities.shape[-1]):
    ind = np.argsort(all_intensities[:, i])
    curr_i = all_intensities[:, i][ind]
    curr_stds = stds[:, i][ind]
    curr_c = np.sort(concentrations[i]).astype(float)
    res = plot_reg(curr_c, curr_i, ax[i], names[i])
    regression_coefficients.append(res)
    step = all_intensities.shape[0]+1
    worksheet.merge_range(0, i*step+1, 0, i*step+step-1, peaks[i], header_format)
    worksheet.write_row(1, i*step+1, curr_i)
    worksheet.write_row(2, i*step+1, curr_stds)
    worksheet.write(3, i*step+1, res.slope)
    worksheet.write(4, i*step+1, res.intercept)

if tissue_regions_name is not None:
    for i, region_name in enumerate(tissue_regions_name):
        region = read_image(region_name)
        indices_regions = np.ravel_multi_index(np.where(region > 0), (max_x, max_y), order='F')
        curr_spectra = spectra[indices_regions]
        intensities, stds, n = compute_average_from_region(curr_spectra, mzs, step, peaks)
        concentrations = []
        for j, res in enumerate(regression_coefficients):
            c = (intensities[j]  - res.intercept) / res.slope
            concentrations.append(c)
        print(intensities, concentrations)
        if i == 0:
            curr_mzs = peaks if peaks is not None else mzs
            worksheet2.write_row(0, 2, curr_mzs, header_format)
        step = 4
        mask_name = os.path.splitext(os.path.basename(region_name))[0]
        worksheet2.merge_range(i*step+1, 0, i*step+step-1, 0, mask_name, header_format)
        worksheet2.write_column(i*step+1, 1, ["mean", "stds", "concentrations"])
        worksheet2.write_row(i*step+1, 2, intensities)
        worksheet2.write_row(i*step+2, 2, stds)
        worksheet2.write_row(i*step+3, 2, concentrations)


worksheet.freeze_panes(1, 1)
worksheet2.freeze_panes(1, 1)
workbook.close()

fig_name = os.path.splitext(output_name)[0] + ".png"
fig.savefig(fig_name, dpi=200)
