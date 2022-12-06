import numpy as np
import joblib
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.metrics import mean_squared_error

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
    print(values)
    return np.mean(values)



def analyze_model(input_name, roc, x):
    names_name = os.path.splitext(input_name)[0] + "_names.csv"
    mzs_name = os.path.splitext(input_name)[0] + "_mzs.csv"
    y_train = os.path.splitext(input_name)[0] + "_y.csv"
    regression = joblib.load(input_name)


    coef = regression.coef_
    if coef.shape[0] < coef.shape[1]:
        coef = coef.T

    peaks = np.loadtxt(mzs_name)
    names = np.loadtxt(names_name, dtype=str)
    y = np.loadtxt(y_train, dtype=float, delimiter=",")
    y_predict = regression.predict(x)

    return mean_squared_error(y, y_predict)
    all_mzs = []
    total_length = 0
    print(input_name)
    for i, name in enumerate(names):
        score = coef[..., i]
        indices = np.argsort(score)[::-1]
        indices = indices[:10]
        mzs = peaks[indices].flatten()
        auc_av = average_value_from_roc(mzs, roc, name)
        print(auc_av)
        all_mzs.append(mzs)
        total_length += len(mzs)
    all_mzs = np.concatenate(all_mzs)
    dup_ratio = duplicate_ratio(all_mzs, total_length)
    print(input_name, dup_ratio)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input dir containing models")
parser.add_argument("--roc", help="ROC values")
parser.add_argument("--lasso", help="Is Lasso", action="store_true")
args = parser.parse_args()

input_dir = args.input
roc_name = args.roc
is_lasso = args.lasso

msi_name = input_dir.replace("models", "trainingsets") + os.path.sep + "train.tif"
image_itk = sitk.ReadImage(msi_name)
images = sitk.GetArrayFromImage(image_itk).T
x = images.reshape(images.shape[1:])

roc_values = pd.read_excel(roc_name, header=0, index_col=0)
roc_names = roc_values.index.values

print(roc_values.loc["Casein"])

nb = []
mse = []
for root, dirs, files in os.walk(input_dir):
    for f in files:
        if f.endswith(".joblib"):
            input_name = root + os.path.sep + f
            number = os.path.splitext(f)[0].split("_")[-1]
            if is_lasso and "lasso" in f:
                val = analyze_model(input_name, roc_values, x)
                nb.append(float(number))
                mse.append(val)
            elif not is_lasso and "pls" in f:
                val = analyze_model(input_name, roc_values, x)
                nb.append(float(number))
                mse.append(val)

plt.scatter(nb, mse)
plt.show()
