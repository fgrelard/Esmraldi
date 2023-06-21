import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk
import os
import esmraldi.imzmlio as io
import esmraldi.imageutils as imageutils
from esmraldi.sliceviewer import SliceViewer

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=None)
parser.add_argument("--names", help="Names to restrict ROC", nargs="+", default=None)
parser.add_argument("--roc", help="Input ROC file (.xlsx)")
args = parser.parse_args()

input_name = args.input
normalization = args.normalization
roc_name = args.roc
roc_names = args.names

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

print("normalization", normalization)
if normalization != None:
    try:
        normalization = float(normalization)
    except:
        pass
    norm_img = imageutils.get_norm_image(images, normalization, mzs)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[...,i], norm_img)

roc_values_df = pd.read_excel(roc_name)
roc_auc_scores = np.array(roc_values_df)
names = roc_values_df.columns
if roc_names is None:
    end = roc_auc_scores.shape[-1]
    ind_names = np.arange(end).astype(int)
else:
    ind_names = np.array([n in roc_names for n in names])
roc_auc_scores = roc_auc_scores[:, ind_names]
for i in range(roc_auc_scores.shape[-1]):
    indices_roc = np.argsort(roc_auc_scores[..., i], axis=0)[::-1]
    image_roc_reordered = images[..., indices_roc]
    mzs_reordered = mzs[indices_roc]
    print(mzs_reordered)
    np.set_printoptions(suppress=True)
    fig, ax = plt.subplots(1)
    plt.suptitle(roc_names[i])
    label = np.vstack((mzs_reordered, roc_auc_scores[indices_roc, i])).T
    tracker = SliceViewer(ax, np.transpose(image_roc_reordered, (2, 1, 0)), labels=label)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
