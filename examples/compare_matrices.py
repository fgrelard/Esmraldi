import xlsxwriter
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import os

def max_lowerthan(array, elem):
    curr_elem = 0
    for elem2 in array:
        if elem2 > elem:
            break
        curr_elem = elem2

    return curr_elem

def extract_info_per_dataset(data, names):
    first_row = data[0, :]
    nan_indices = np.where(pd.isna(first_row))[0]
    all_values = []
    for name in names:
        col_indices = np.where(first_row == name)[0]
        values = []
        for col_ind in col_indices:
            nan_ind = max_lowerthan(nan_indices, col_ind)
            row_indices = np.where(data[:, nan_ind] == name)[0]
            if row_indices.size:
                row_index = row_indices[0]
                value = data[row_index, col_ind]
                values.append(value)
        all_values.append(values)
    return all_values

def extract_all_info_per_dataset(data, names):
    first_row = data[0, :]
    nan_indices = np.where(pd.isna(first_row))[0]
    all_values = []
    for ind, name in enumerate(names):
        col_indices = np.where(first_row == name)[0]
        name_values = []
        for j, col_ind in enumerate(col_indices):
            nan_ind = max_lowerthan(nan_indices, col_ind)
            row_indices = np.where(data[:, nan_ind] == name)[0]
            if row_indices.size:
                row_index = row_indices[0]
                if j == col_indices.size - 1:
                    next_nan_ind = None
                else:
                    next_nan_ind = max_lowerthan(nan_indices, col_indices[j+1])-1
                values = np.array(data[row_index, nan_ind+1:next_nan_ind], dtype=np.float128)
                name_values.append(values)
        all_values.append(name_values)
    all_values = np.array(all_values, dtype=object)
    return all_values


def data_summary(values, names, legend_names=None):
    caption_names = np.array(legend_names)
    if legend_names is None:
        caption_names = np.array(names)
    means = []
    stds = []
    for i, all_values in enumerate(values):
        mean = np.mean(all_values, axis=0)
        std = np.std(all_values, axis=0)
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)
    y_offset = np.zeros(len(names))
    cm = plt.get_cmap("Dark2")

    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(names))
    array_colors = np.array(cm.colors)
    names_array = np.array(names)
    ind_mat = np.where((names_array == "Matrix") | (names_array == "Tape"))[0]
    black = np.array([0.15, 0.15, 0.15])
    gray = np.array([0.3, 0.3, 0.3])
    array_colors[len(names), :] = gray
    array_colors[ind_mat, :] = black
    cm = colors.ListedColormap(array_colors)

    tmp = means[-1, :].copy()
    means[-1, :] = means[ind_mat, :]
    means[ind_mat, :] = tmp
    caption_names[-1], caption_names[ind_mat] = caption_names[ind_mat[0]], caption_names[-1]

    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    plt.figure()
    for i in range(len(means[0])):
        m = means[:, i]
        plt.bar(caption_names, m, width=0.4, bottom=y_offset, color=array_colors[i])
        y_offset += m



def diffs(values_ref, values_target):
    for i, target in enumerate(values_target):
        reference = values_ref[i]
        diff = np.array(target) - np.array(reference)
        av_diff = np.mean(diff)
        std_diff = np.std(diff)
        print(av_diff, std_diff)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--reference", help="Input xlsx")
parser.add_argument("-t", "--target", help="Target xlsx")
parser.add_argument("--names", help="Names", nargs="+")
parser.add_argument("--legend_names", help="Legend names", nargs="+")
parser.add_argument("-o", "--output", help="Output directory")
args = parser.parse_args()

reference_name = args.reference
target_name = args.target
region_names = args.names
legend_names = args.legend_names
outdir = args.output

data_reference = pd.read_excel(reference_name, sheet_name=None)
data_target = pd.read_excel(target_name, sheet_name=None)

values_ref = data_reference.values()
values_target = data_target.values()

for i, value_ref in enumerate(values_ref):
    name = list(data_reference.keys())[i]
    if "Sensibility" not in name and "Specificity" not in name:
        continue
    print(name)
    ref = np.array(value_ref)
    target = np.array(list(values_target)[i])
    allinfo_ref = extract_all_info_per_dataset(ref, region_names)
    allinfo_target = extract_all_info_per_dataset(target, region_names)
    info_ref = extract_info_per_dataset(ref, region_names)
    info_target = extract_info_per_dataset(target, region_names)
    data_summary(allinfo_ref, region_names, legend_names)
    filename = "barplot_" + name.lower().replace(" ", "_")
    savename = outdir + os.path.sep + filename
    plt.savefig(savename + "_gmm.pdf")
    data_summary(allinfo_target, region_names, legend_names)
    plt.savefig(savename + "_nogmm.pdf")
    diffs(info_ref, info_target)
