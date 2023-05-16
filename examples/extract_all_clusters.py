import numpy as np
import SimpleITK as sitk
import argparse
import os
import esmraldi.registration as reg
import matplotlib.pyplot as plt


def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    print(image_name)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input cluster file")
parser.add_argument("-o", "--output", help="Output directory")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str, default=None)
parser.add_argument("--names", help="Mask names", nargs="+", type=str)
args = parser.parse_args()

input_name = args.input
output_name = args.output
region_names = args.regions
names = args.names

if region_names is not None:
    regions = []
    print("Read image")
    for region_name in region_names:
        region = read_image(region_name)
        print(region.shape)
        regions.append(region.T)


image = sitk.GetArrayFromImage(sitk.ReadImage(input_name))
colors = ["#dfaa01", "#9901ff", "#ff0000", "#0070c0", "#00b050"]

print(image.shape)
for i, value in enumerate(np.unique(image)):
    cluster = np.where(image == value, 255, 0).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(cluster), output_name + os.path.sep + "cluster_" + str(i) + ".tif")
    if region_names is not None:
        fmeasures = []
        for region in regions:
            cluster_bin = np.where(cluster > 0, 1, 0)
            region_bin = np.where(region > 0, 1, 0)
            precision = reg.precision(cluster_bin, region_bin)
            recall = reg.recall(cluster_bin, region_bin)
            f = reg.fmeasure(precision, recall)
            fmeasures.append(f)
        fig, ax = plt.subplots(figsize=(8,4), ncols=1)
        ax.bar(names, fmeasures, color=colors)
        ax.spines[['top', 'right']].set_visible(False)
        ax.autoscale()
        plt.autoscale()
        ax.set_ylim([0, 1])
        plt.savefig(output_name + os.path.sep + "fmeasure_" + str(i) + ".pdf")
        plt.close()
