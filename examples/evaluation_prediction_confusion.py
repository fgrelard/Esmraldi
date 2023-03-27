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
    # mask_path = parent_path + os.path.sep + "masks/msi/*.tif"
    mask_path = parent_path + os.path.sep + "masks/msi/*.tif"
    files = glob.glob(mask_path)
    binders=[]
    pigments=[]
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        binder, pigment = name.split("_")
        image = sitk.GetArrayFromImage(sitk.ReadImage(f)).T
        if binder == "ET&LO":
            ind_binder =  np.where(names == "ET")[0][0]
            regions[..., ind_binder] += image
            # ind_binder =  np.where(names == "LO")[0][0]
            # regions[..., ind_binder] += image
            binders.append("ET")
            # binders.append("LO")
        else:
            ind_binder = np.where(names == binder)[0][0]
            binders.append(binder)
        pigments.append(pigment)
        ind_pigment = np.where(names == pigment)[0][0]
        regions[..., ind_binder] += image
        regions[..., ind_pigment] += image
    return regions, binders, pigments

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

def get_mask(index, masks, uncertain_label):
    if index != uncertain_label:
        mask = masks[..., index]
    else:
        return np.zeros_like(masks[..., 0])
    mask = np.where(mask>0, 1, 0)
    return mask

def get_prediction(index, label_image, uncertain_label):
    cond = (label_image == index)
    # if index == uncertain_label:
    #     for i in range(uncertain_label):
    #         if not inside[i]:
    #             cond |= (label_image == i)
    return np.where(cond, 1, 0)

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
proba = float(args.proba)

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

previous = 1


n = len(analysis_names)
uncertain_label = n

if proba > 0:
    analysis_names.append("Uncertain")

names_array = np.array(analysis_names)
sum_up = [ [[] for x in range(n) ] for _ in range(n) ]


for i, target_name in enumerate(target_names):
    trimmed_name = os.path.splitext(os.path.basename(target_name))[0]
    trimmed_name = trimmed_name.split("_")
    index = trimmed_name.index("5um")+1
    trimmed_name = trimmed_name[index]
    print(trimmed_name)
    worksheet.write(0, previous, trimmed_name)

    spectra, mzs, shape, coords = read_imzml(target_name, normalization)
    masks, binders, pigments = extract_parent_paths(target_name, shape, names)
    indices = indices_peaks(mzs, peaks)

    target_im = normalize_flatten(spectra, coords, shape, normalization_tic=normalization, normalization_minmax=True)
    blank_image = np.zeros((target_im.shape[0], 1))
    target_im = np.hstack((target_im, blank_image))
    target_im = target_im[..., indices]

    out = regression.predict(target_im)

    if analysis_names is not None:
        inside = np.in1d(names, analysis_names)
        out = out[..., inside]
        masks = masks[..., inside]
        currnames = names[inside]

    intersect = np.concatenate([np.intersect1d(currnames, pigments), np.intersect1d(currnames, binders)])
    inside2 = np.in1d(currnames, intersect)
    print(inside2, currnames, intersect)

    labels = np.argmax(out, axis=-1)

    if gmm_name is not None:
        gmm = joblib.load(gmm_name)
        probas = gmm.predict_proba(out)
        labels = gmm.predict(out)

        if proba > 0:
            labels[probas.max(axis=-1) < proba] = uncertain_label
    # else:
    #     uncertain_label -= 1

    label_image = np.reshape(labels, shape[:-1])
  
    worksheet.write_column(2, previous, intersect)
    previous += 1
    worksheet.write_row(1, previous, analysis_names)
    r = 0
    for i in range(n):
        mask = get_mask(i, masks, uncertain_label)
        c = 0
        mask_empty = np.count_nonzero(mask) == 0
        for j in range(n):
            prediction = get_prediction(j, label_image, uncertain_label)
            if mask.size == 0 and prediction.size == 0:
                continue
            prediction_empty = np.count_nonzero(prediction) == 0
            if mask_empty and prediction_empty:
                tp, fp, fn, tp = 0, 0, 0, 0
            else:
                tn, fp, fn, tp = confusion_matrix(mask.flatten(), prediction.flatten()).ravel()
            if tp+fn == 0:
                se = 0
            else:
                se = tp / (tp+fn)
            if not mask_empty:
                worksheet.write(2+r, previous+c, se)
                sum_up[i][j].append(se)
            # plt.imshow(mask, cmap="Reds")
            # plt.imshow(prediction, cmap="Greens", alpha=0.5)
            # plt.show()
            c += 1
        if not mask_empty:
            r += 1

        # print(se, sp)
        # worksheet.write(2, previous+i, sp)
        # worksheet.write(3, previous+i, se)

    previous += len(analysis_names)+1


worksheet.write_column(5+len(analysis_names), 0, analysis_names)
worksheet.write_row(4+len(analysis_names), 1, analysis_names)

for r in range(n):
    for c in range(n):
        mean = np.nan_to_num(np.mean(sum_up[r][c]))
        worksheet.write(5+len(analysis_names)+r, 1+c, mean)


worksheet.freeze_panes(0, 1)
workbook.close()
