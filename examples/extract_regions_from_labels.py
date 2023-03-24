import numpy as np
from skimage.morphology import opening, closing, disk
import SimpleITK as sitk
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("--size", help="Size SE", default=1)
parser.add_argument("--correspondences", help="Correspondence list")
parser.add_argument("--merge", action="store_true")
args = parser.parse_args()

inputname = args.input
outputname = args.output
size_se = float(args.size)
correspondence_names = args.correspondences
is_merge = args.merge

os.makedirs(outputname, exist_ok=True)
correspondences = pd.read_csv(correspondence_names, delimiter=",")
indices = np.array(correspondences)[:, 0]
correspondences = np.array(correspondences)[:, 1]
correspondences = [s.replace(" ", "_") for s in correspondences]

image = sitk.GetArrayFromImage(sitk.ReadImage(inputname)).T
max_im = int(image.max())
if is_merge:
    max_im //= 2

print(max_im, correspondences, indices)

for ind, intensity in enumerate(indices):
    if is_merge:
        condition = (image == intensity) | (image == intensity+20)
    else:
        condition = (image == intensity)
    mask = np.where(condition, 255, 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        continue
    outname = outputname + os.path.sep + correspondences[ind] + ".tif"
    sitk.WriteImage(sitk.GetImageFromArray(mask.T), outname)



    # plt.imshow(out)
    # plt.show()
