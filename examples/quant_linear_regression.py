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


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-m", "--mask", help="Mask with different subregions")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output .csv files with stats")
parser.add_argument("--peak_list", help="Peak list", default=None)
parser.add_argument("-p", "--prominence", help="Prominence for peak picking", default=0)
args = parser.parse_args()

plt.style.use('dark_background')

input_name = args.input
mask_name = args.mask
output_name = args.output
normalization = float(args.normalization)
prominence = float(args.prominence)
peak_list = args.peak_list

region = read_image(mask_name)
all_labels = measure.label(region, background=0)


name = "No norm"
if normalization > 0:
    name = str(normalization)
elif normalization == -1:
    name = "tic"

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
if prominence > 0:
    print("Mean spectrum")
    npy_name = os.path.splitext(input_name)[0] + "_meanspectra.npy"
    if os.path.isfile(npy_name):
        mzs, mean_spectrum = np.load(npy_name)
    else:
        mzs = np.unique(np.hstack(spectra[:, 0]))
        mzs = mzs[mzs>0]
        mean_spectrum = sp.spectra_mean_centroided(spectra, mzs)
        np.save(npy_name, [mzs, mean_spectrum])
    print("Peak detection")
    mean_spectrum = sp.spectra_mean_centroided(spectra, mzs)
    peak_detection = PeakDetectionMeanSpectrum(mzs, mean_spectrum, prominence, step)
    peak_indices = peak_detection.extract_peaks()
    peaks = mzs[peak_indices]
    intensities = mean_spectrum[peak_indices]
    print("Found", peaks.shape[0], "peaks")

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
for i in range(1, all_labels.max()+1):
    indices_regions = np.ravel_multi_index(np.where(all_labels == i), (max_x, max_y), order='F')
    curr_spectra = spectra[indices_regions]
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
    all_intensities.append(intensities)
    # mean_spectra = sp.spectra_mean_centroided(curr_spectra, mzs)

all_intensities = np.array(all_intensities)
fig, ax = plt.subplots(1, len(peaks))
for i in range(all_intensities.shape[-1]):
    curr_i = np.sort(all_intensities[:, i])
    curr_c = np.sort(concentrations[i]).astype(float)
    plot_reg(curr_c, curr_i, ax[i], names[i])
plt.show()
