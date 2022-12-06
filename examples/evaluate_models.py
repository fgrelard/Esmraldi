import numpy as np
import joblib
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.metrics import mean_squared_error
from esmraldi.registration import precision, recall, quality_registration
from sklearn.metrics import confusion_matrix, recall_score
from natsort import natsorted

def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask

def duplicate_ratio(all_mzs, total_length):
    unique = np.unique(all_mzs)
    length_unique = len(unique)
    length_initial = len(all_mzs)
    nb_duplicates = length_initial - length_unique
    return nb_duplicates / total_length

def average_value_from_roc(mzs, roc, name_cond):
    values = roc.loc[name_cond, mzs]
    return np.mean(values)



def analyze_model(input_name, x, y, region_names):
    names_name = os.path.splitext(input_name)[0] + "_names.csv"
    mzs_name = os.path.splitext(input_name)[0] + "_mzs.csv"
    y_train = os.path.splitext(input_name)[0] + "_y.csv"

    regression = joblib.load(input_name)
    coef = regression.coef_
    if coef.shape[0] < coef.shape[1]:
        coef = coef.T

    peaks = np.loadtxt(mzs_name)
    names = np.loadtxt(names_name, dtype=str)
    # y = np.loadtxt(y_train, dtype=float, delimiter=",")

    y_predict = regression.predict(x)

    indices_y = np.array([n in names for n in region_names])
    indices_y_predict = np.array([n in region_names for n in names])
    y = y[..., indices_y]
    y_predict = y_predict[..., indices_y_predict]

    y_bin = np.where(y>0, 1, 0).astype(np.uint8)
    y_predict_bin = np.where(y_predict>0, 1, 0).astype(np.uint8)
    r = recall(y_bin, y_predict_bin)
    p = precision(y_bin, y_predict_bin)
    se = recall_score(y_bin.flatten(), y_predict_bin.flatten())
    sp = recall_score(y_bin.flatten(), y_predict_bin.flatten(), pos_label=0)
    mse = mean_squared_error(y, y_predict)
    return r, p, sp, se, mse
    # all_mzs = []
    # total_length = 0
    # print(input_name)
    # for i, name in enumerate(names):
    #     score = coef[..., i]
    #     indices = np.argsort(score)[::-1]
    #     indices = indices[:10]
    #     mzs = peaks[indices].flatten()
    #     auc_av = average_value_from_roc(mzs, roc, name)
    #     print(auc_av)
    #     all_mzs.append(mzs)
    #     total_length += len(mzs)
    # all_mzs = np.concatenate(all_mzs)
    # dup_ratio = duplicate_ratio(all_mzs, total_length)
    # print(input_name, dup_ratio)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input dir containing models")
parser.add_argument("--validation_dataset", help="Validation dataset", default=None)
parser.add_argument("--roc", help="ROC values", default=None)
parser.add_argument("--lasso", help="Is Lasso", action="store_true")
args = parser.parse_args()

input_dir = args.input
roc_name = args.roc
validation_name = args.validation_dataset
is_lasso = args.lasso

if validation_name is None:
    msi_name = input_dir.replace("models", "trainingsets") + os.path.sep + "train.tif"
else:
    msi_name = validation_name


image_itk = sitk.ReadImage(msi_name)
images = sitk.GetArrayFromImage(image_itk).T
x = images.reshape(images.shape[1:])

region_names = os.path.dirname(msi_name) + os.path.sep + "regions" + os.path.sep + "*.tif"
regions = []
names = []
files = glob.glob(region_names)
files = natsorted(files)
for region_name in files:
    names.append(os.path.splitext(os.path.basename(region_name))[0])
    region = read_image(region_name)
    regions.append(region)

y = np.concatenate(regions).T


if roc_name is not None:
    roc_values = pd.read_excel(roc_name, header=0, index_col=0)
    roc_names = roc_values.index.values

    print(roc_values.loc["Casein"])

nb = []
vals = []
for root, dirs, files in os.walk(input_dir):
    for f in files:
        card = len(f.split("_"))
        if card < 4 and f.endswith(".joblib"):
            input_name = root + os.path.sep + f
            number = os.path.splitext(f)[0].split("_")[-1]
            if is_lasso and "lasso" in f:
                currvals = analyze_model(input_name, x, y, names)
                nb.append(float(number))
                vals.append(currvals)

            elif not is_lasso and "pls" in f:
                currvals= analyze_model(input_name, x, y, names)
                nb.append(float(number))
                vals.append(currvals)

vals = np.array(vals)

fig, ax = plt.subplots(1, 5)
ax[0].scatter(nb, vals[:, 0])
ax[1].scatter(nb, vals[:, 1])
ax[2].scatter(nb, vals[:, 2])
ax[3].scatter(nb, vals[:, 3])
ax[4].scatter(nb, vals[:, 4])

plt.show()
