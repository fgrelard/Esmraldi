import joblib
import argparse
import numpy as np
import os
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.preprocessing import StandardScaler
import esmraldi.imzmlio as io
import esmraldi.utils as utils
import esmraldi.imageutils as imageutils
import esmraldi.fusion as fusion
import esmraldi.spectraprocessing as sp
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import glob
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix
import xlsxwriter


def extract_parent_paths(imzml_name, shape, names):
    paths = []
    regions = np.zeros(shape[:-1] + (len(names),))
    parent_path = os.path.dirname(imzml_name)
    mask_path = parent_path + os.path.sep + "masks/resized/*.tif"
    files = glob.glob(mask_path)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        binder, pigment = name.split("_")
        image = sitk.GetArrayFromImage(sitk.ReadImage(f)).T
        if binder == "ET&LO":
            ind_binder =  np.where(names == "ET")[0][0]
            regions[..., ind_binder] += image
            ind_binder =  np.where(names == "LO")[0][0]
            regions[..., ind_binder] += image
        else:
            ind_binder = np.where(names == binder)[0][0]
        ind_pigment = np.where(names == pigment)[0][0]
        regions[..., ind_binder] += image
        regions[..., ind_pigment] += image
    return regions

def normalize_flatten(spectra, coordinates, shape, normalization_tic=True, normalization_minmax=True):
    if normalization:
        print("normalization")
        spectra = sp.normalization_tic(spectra, inplace=True)
    full_spectra = io.get_full_spectra_dense(spectra, coordinates, shape)
    images = io.get_images_from_spectra(full_spectra, shape)
    if normalization_minmax:
        images = io.normalize(images)
    images = images.astype(np.float128) / 255.0
    image_flatten = fusion.flatten(images, is_spectral=True).T
    return image_flatten

def read_imzml(input_name, normalization):
    if input_name.lower().endswith(".imzml"):
        imzml = io.open_imzml(input_name)
        spectra = io.get_spectra(imzml)
        coordinates = imzml.coordinates
        max_x = max(coordinates, key=lambda item:item[0])[0]
        max_y = max(coordinates, key=lambda item:item[1])[1]
        max_z = max(coordinates, key=lambda item:item[2])[2]
        shape = (max_x, max_y, max_z)
        mzs = np.unique(np.hstack(spectra[:, 0]))
        mzs = mzs[mzs>0]
    return spectra, mzs, shape, imzml.coordinates

def indices_peaks(peaks, other_peaks):
    indices = utils.indices_search_sorted(other_peaks, peaks)
    print(len(indices), len(other_peaks), len(peaks))
    current_step = 14 * other_peaks / 1e6
    indices_ppm = np.abs(peaks[indices] - other_peaks) < current_step
    indices[~indices_ppm] = -1
    return indices


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input json")
parser.add_argument("-t", "--target", help="Target .imzML", nargs="+")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", action="store_true")
parser.add_argument("-o", "--output", help="Output files")
parser.add_argument("--names", help="Names to analyze (default all)", nargs="+", type=str, default=None)
parser.add_argument("--gmm", help="GMM model (.joblib)", default=None)
parser.add_argument("--proba", help="Proba", default=0)

args = parser.parse_args()

input_name = args.input
target_names = args.target
normalization = args.normalization
outname = args.output
analysis_names = args.names
gmm_name = args.gmm
proba = args.proba

mzs_name = os.path.splitext(input_name)[0] + "_mzs.csv"
names_name = os.path.splitext(input_name)[0] + "_names.csv"
peaks = np.loadtxt(mzs_name)
names = np.loadtxt(names_name, dtype=str)

# Load data from file
regression = joblib.load(input_name)

workbook = xlsxwriter.Workbook(outname, {'strings_to_urls': False})
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})
worksheet = workbook.add_worksheet("Stats")
worksheet.write_column(2, 0, ["Specificity", "Sensitivity"])

previous = 1

for i, target_name in enumerate(target_names):
    trimmed_name = os.path.splitext(os.path.basename(target_name))[0]
    trimmed_name = trimmed_name.split("_")
    index = trimmed_name.index("5um")+1
    trimmed_name = trimmed_name[index]
    print(trimmed_name)
    worksheet.write(0, previous, trimmed_name)

    spectra, mzs, shape, coords = read_imzml(target_name, normalization)
    masks = extract_parent_paths(target_name, shape, names)
    indices = indices_peaks(mzs, peaks)

    target_im = normalize_flatten(spectra, coords, shape, normalization_tic=normalization, normalization_minmax=True)
    blank_image = np.zeros((target_im.shape[0], 1))
    target_im = np.hstack((target_im, blank_image))
    target_im = target_im[..., indices]

    out = regression.predict(target_im)

    if analysis_names is not None:
        inside = np.in1d(names, analysis_names)
        currnames = names[inside]
        out = out[..., inside]
        masks = masks[..., inside]
        worksheet.write_row(1, previous, currnames)

    labels = np.argmax(out, axis=-1)

    gmm = joblib.load(gmm_name)
    labels = gmm.predict(out)
    probas = gmm.predict_proba(out)
    uncertain_label = len(analysis_names)
    labels[probas.max(axis=-1) < proba] = uncertain_label
    label_image = np.reshape(labels, shape[:-1])

    for i in range(out.shape[-1]):
        mask = masks[..., i]
        prediction = np.where(label_image==i, 1, 0)
        mask = np.where(mask>0, 1, 0)
        if mask.any():
            tn, fp, fn, tp = confusion_matrix(mask.flatten(), prediction.flatten()).ravel()
            sp = tn / (tn+fp)
            se = tp / (tp+fn)
            se = "nan" if np.isnan(se) else se
            sp = "nan" if np.isnan(sp) else sp
        else:
            sp = "nan"
            se = "nan"
        print(se, sp)
        worksheet.write(2, previous+i, sp)
        worksheet.write(3, previous+i, se)

    previous += len(currnames)


worksheet.freeze_panes(0, 1)
workbook.close()
