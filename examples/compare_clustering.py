import esmraldi.imzmlio as io
import SimpleITK as sitk
import os
import scipy.spatial.distance as distance
from sklearn.metrics import confusion_matrix, roc_auc_score
import esmraldi.imageutils as imageutils
import esmraldi.fusion as fusion
import esmraldi.segmentation as seg
from esmraldi.sliceviewer import SliceViewer
import numpy as np
import matplotlib.pyplot as plt
import argparse
import xlsxwriter
import re
import pandas as pd
from natsort import natsorted
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input clusters", nargs="+")
parser.add_argument("-t", "--target", help="Target clusters", nargs="+")
parser.add_argument("-v", "--value", help="Value", default=0)
parser.add_argument("-o", "--output", help="Output xlsx file")
args = parser.parse_args()

input_names = args.input
target_names = args.target
output_name = args.output
value = float(args.value)

inputs = []
targets = []

assess_missing = False
if input_names[0].endswith(".csv"):
    assess_missing = True

root, ext = os.path.splitext(output_name)
missing = root + os.path.sep + "missing"
common = root + os.path.sep + "common"
if assess_missing:
    os.makedirs(root, exist_ok=True)
    os.makedirs(common, exist_ok=True)
    os.makedirs(missing, exist_ok=True)
    print(missing)
    for f in os.listdir(missing):
        os.remove(os.path.join(missing, f))
    for f in os.listdir(common):
        os.remove(os.path.join(common, f))
    input_name = input_names[0]
    target_name = target_names[0]
    image_name = os.path.dirname(target_name) + os.path.sep + "av_image"
    input_array = np.array(pd.read_csv(input_name, delimiter=" ")).astype(np.float64)
    target_array = np.array(pd.read_csv(target_name, delimiter=" ")).astype(np.float64)
    mzs_input = input_array[:, 0]
    cluster_input = input_array[:, 3]
    mzs_target = target_array[:, 0]
    cluster_target = target_array[:, 3]

    unique_input = np.unique(cluster_input)
    unique_target = np.unique(cluster_target)
    number_in_common = 1

    workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
    worksheet = workbook.add_worksheet("Missing")
    worksheet.write_row(0, 0, ["Clusters", "Missing", "Total miss", "Total common"])
    worksheet.write_column(1, 0, np.arange(unique_target.size, dtype=int))
    cpt = 0
    cpt_common = 0
    for i in unique_target:
        indices = cluster_target == i
        mzs_cluster = mzs_target[indices]
        found = False
        for j in unique_input:
            indices_j = cluster_input == j
            mzs_cluster_other = mzs_input[indices_j]
            intersect = np.intersect1d(mzs_cluster, mzs_cluster_other)
            if intersect.size >= number_in_common:
                found = True
                break
        if found:
            cpt_common += 1
            shutil.copy(image_name + str(int(i-1)) + ".tif", common)
            worksheet.write(i, 1, "False")
        else:
            cpt+=1
            shutil.copy(image_name + str(int(i-1)) + ".tif", missing)
            worksheet.write(i, 1, "True")
    worksheet.write(1, 2, cpt)
    worksheet.write(1, 3, cpt_common)
    workbook.close()
else:
    input_names = natsorted(input_names)
    target_names = natsorted(target_names)

    for input_name in input_names:
        input_im = sitk.GetArrayFromImage(sitk.ReadImage(input_name)).T
        inputs.append(input_im)

    for target_name in target_names:
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_name)).T
        targets.append(target)

    input_array = np.transpose(inputs, (1, 2, 0))
    target_array = np.transpose(targets, (1, 2, 0))
    print(input_array.shape)

    X = fusion.flatten(input_array, is_spectral=True)
    y = fusion.flatten(target_array, is_spectral=True)

    similar_images, distances, indices_same_input, all_distances = seg.find_similar_image_distance_map_percentile(input_array, targets, value, quantiles=[0], add_otsu_thresholds=True, return_indices=True, reverse=False, is_mean=True, return_distances=True)

    conditions = (all_distances < value).astype(int)

    workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
    worksheet = workbook.add_worksheet("Comparison")
    worksheet2 = workbook.add_worksheet("Threshold_" + str(value))
    worksheets = [worksheet, worksheet2]

    input_names_trimmed = [os.path.splitext(os.path.basename(input_name))[0] for input_name in input_names]
    target_names_trimmed = [os.path.splitext(os.path.basename(target_name))[0] for target_name in target_names]

    for w in worksheets:
        w.write_column(1, 0, input_names_trimmed)
        w.write_row(0, 1, target_names_trimmed)

    for row, data in enumerate(all_distances):
        worksheet.write_row(row+1, 1, data)

    end = 1
    for row, data in enumerate(conditions):
        worksheet2.write_row(end, 1, data)
        end += 1

    common_element_target = np.max(conditions, axis=0)
    common_element_input = np.max(conditions, axis=1)
    total_input = conditions.shape[0]
    common_input = np.sum(common_element_input)
    missing_input = total_input - common_input
    total_target = conditions.shape[1]
    common_target = np.sum(common_element_target)
    missing_target = total_target - common_target
    stats_input = [total_input, common_input, missing_input]
    stats_target = [total_target, common_target, missing_target]

    for w in worksheets:
        w.write(0, total_target+1, "Found")
        w.write(end, 0, "Found")
        w.write_column(1, total_target+1, common_element_input)
        w.write_row(end, 1, common_element_target)
    end += 2
    for w in worksheets:
        w.write_row(end, 1, ["Total", "Common", "Missing"])
        w.write(end+1, 0, re.sub('\d+', '', input_names_trimmed[0]))
        w.write_row(end+1, 1, stats_input)
        w.write(end+2, 0, re.sub('\d+', '', target_names_trimmed[0]))
        w.write_row(end+2, 1, stats_target)
    workbook.close()

    exit()

    np.set_printoptions(suppress=True)
    fig, ax = plt.subplots(1)
    label = distances.T
    tracker = SliceViewer(ax, np.transpose(input_array, (2, 1, 0)), labels=label)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

    similar_images, distances, indices_same_cluster = seg.find_similar_image_distance_map_percentile(input_array, targets, value, quantiles=[0], add_otsu_thresholds=True, return_indices=True, reverse=True, is_mean=True)
    np.set_printoptions(suppress=True)
    fig, ax = plt.subplots(1)
    label = distances.T
    tracker = SliceViewer(ax, np.transpose(target_array, (2, 1, 0)), labels=label)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    print(similar_images.shape)


    indices_inputs_all = np.arange(X.shape[0])
    indices_targets_all = np.arange(y.shape[0])
    indices_same_input = np.argwhere(indices_same_input).ravel()
    indices_same_cluster = np.argwhere(indices_same_cluster).ravel()
    not_indices_inputs = np.setdiff1d(indices_inputs_all, indices_same_input)
    not_indices_targets = np.setdiff1d(indices_targets_all, indices_same_cluster)

    print(indices_inputs_all, indices_same_input, not_indices_inputs)

    not_inputs = not_indices_inputs.size
    not_targets = not_indices_targets.size

    print(not_inputs, not_targets, indices_inputs_all.size, indices_targets_all.size)

    for i in not_indices_inputs:
        plt.imshow(np.reshape(X[i, ...], input_array.shape[:-1]))
        plt.show()

    for i in not_indices_targets:
        plt.imshow(np.reshape(y[i, ...], target_array.shape[:-1]))
        plt.show()




    for i in range(X.shape[0]):
        for j in range(y.shape[0]):
            correlation = distance.correlation(X[i, ...], y[j, ...])
            if correlation < 0.6:
                indices_inputs.append(i)
                indices_targets.append(j)
                print(correlation)
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(np.reshape(X[i, ...], inputs.shape[:-1]))
                ax[1].imshow(np.reshape(y[j, ...], inputs.shape[:-1]))
                plt.show()

    indices_inputs = np.unique(indices_inputs)
    indices_targets = np.unique(indices_targets)

    indices_inputs_all = np.arange(X.shape[0])
    indices_targets_all = np.arange(y.shape[0])
    not_indices_inputs = np.setdiff1d(indices_inputs_all, indices_inputs)
    not_indices_targets = np.setdiff1d(indices_targets_all, indices_targets)


    # for i in not_indices_inputs:
    #     plt.imshow(np.reshape(X[i, ...], inputs.shape[:-1]))
    #     plt.show()

    # for i in not_indices_targets:
    #     plt.imshow(np.reshape(y[i, ...], inputs.shape[:-1]))
    #     plt.show()

    not_inputs = not_indices_inputs.size
    not_targets = not_indices_targets.size

    print(not_inputs, not_targets, indices_inputs.size, indices_targets.size)
