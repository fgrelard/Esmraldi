import numpy as np
from skimage.morphology import opening, closing, disk
import SimpleITK as sitk
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

def max_region(image):
    image_copy = image.copy().astype(int)
    for (x,y) in np.ndindex(image.shape):
        values = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i == 0 and j == 0) or x+i < 0 or y+j < 0 or x+i >= image.shape[0] or y+j >= image.shape[1]:
                    continue
                values.append(image[x+i, y+j])
        unique, counts = np.unique(values, return_counts=True)
        if unique.size > 1:
            image_copy[x, y] = unique[np.argmax(counts)]
        else:
            image_copy[x, y] = image[x, y]
    return image_copy

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



image = sitk.GetArrayFromImage(sitk.ReadImage(inputname)).T
max_im = int(image.max())
if is_merge:
    max_im //= 2


image[image > max_im] -= max_im
image_copy = max_region(image.astype(int)).astype(np.uint8)
sitk.WriteImage(sitk.GetImageFromArray(image_copy.T), "test.png")
# plt.imshow(image_copy.T, cmap="Set3", interpolation="none", vmax=max_im)
# plt.axis("off")
# plt.savefig("test.png", bbox_inches="tight")
exit()

os.makedirs(outputname, exist_ok=True)
correspondences = pd.read_csv(correspondence_names, delimiter=",")
indices = np.array(correspondences)[:, 0]
correspondences = np.array(correspondences)[:, 1]
correspondences = [s.replace(" ", "_") for s in correspondences]

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
