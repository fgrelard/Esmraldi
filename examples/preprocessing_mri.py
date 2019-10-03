"""
Remove tube and pericarp from MRI image
To ease the registration process
"""

import src.registration as reg
import numpy as np
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="MRI image (any ITK format)")
parser.add_argument("-o", "--output", help="Output")
parser.add_argument("-r", "--minr", help="Small radius", default=15)
parser.add_argument("-R", "--maxr", help="Big radius", default=25)
parser.add_argument("-t", "--threshold", help="Threshold", default=150)
args = parser.parse_args()

filename_in = args.input
filename_out = args.output
minr = int(args.minr)
maxr = int(args.maxr)
threshold = int(args.threshold)

image = sitk.ReadImage(filename_in, sitk.sitkFloat32)
image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)

array_image = sitk.GetArrayFromImage(image)

dim_image = len(image.GetSize())
if dim_image == 2:
    center_x, center_y, radius = reg.detect_circle(array_image, threshold=threshold, min_radius=minr, max_radius=maxr)
    array_image = reg.fill_circle(center_x, center_y, maxr, array_image)

if dim_image == 3:
    center_x, center_y, radius = reg.detect_tube(array_image, min_radius=minr, max_radius=maxr, threshold=threshold)
    array_image = reg.fill_circle(center_x, center_y, maxr, array_image)

plt.imshow(array_image[8,...])
plt.show()
sitk.WriteImage(sitk.GetImageFromArray(array_image), filename_out)
