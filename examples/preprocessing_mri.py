"""
Remove tube and pericarp from MRI image
To ease the registration process
"""

import src.segmentation as seg
import numpy as np
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="MRI image (any ITK format)")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("-s", "--radius_se", help="Radius for structuring element to remove pericarp", default=1)
parser.add_argument("-r", "--minr", help="Small radius for tube detection", default=15)
parser.add_argument("-R", "--maxr", help="Big radius for tube detection", default=25)
parser.add_argument("-t", "--threshold", help="Threshold for tube detection", default=150)
args = parser.parse_args()

filename_in = args.input
filename_out = args.output
radius_selem = int(args.radius_se)
minr = int(args.minr)
maxr = int(args.maxr)
threshold = int(args.threshold)

image = sitk.ReadImage(filename_in, sitk.sitkUInt8)
start_size = image.GetSize()[0]
image = seg.resize(image, start_size*2)
image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)

array_image = sitk.GetArrayFromImage(image)

dim_image = len(image.GetSize())
if dim_image == 2:
    center_x, center_y, radius = seg.detect_circle(array_image, threshold=threshold, min_radius=minr, max_radius=maxr)
    array_image = seg.fill_circle(center_x, center_y, maxr, array_image)

if dim_image == 3:
    center_x, center_y, radius = seg.detect_tube(array_image, min_radius=minr, max_radius=maxr, threshold=threshold)
    array_image = seg.fill_circle(center_x, center_y, maxr, array_image)

array_image = seg.remove_pericarp(array_image, radius_selem)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(sitk.GetArrayFromImage(image))
ax[1].imshow(array_image)
plt.show()

image = sitk.GetImageFromArray(array_image)
image = seg.resize(image, start_size)
sitk.WriteImage(image, filename_out)
