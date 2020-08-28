import argparse
import sys
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import esmraldi.imageutils as utils

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input image")
parser.add_argument("-o", "--output", help="Output distance transform image")
parser.add_argument("-t", "--threshold", help="Threshold for binarization of the image", default=0)
args = parser.parse_args()

inputname = args.input
outputname = args.output
threshold = args.threshold

fixed_image = sitk.ReadImage(inputname,sitk.sitkFloat32)

fixed_array = sitk.GetArrayFromImage(fixed_image)
fixed_bin = np.where(fixed_array > threshold, 255, 0)

sampling = [1, 1]
if fixed_image.GetDimension() > 2:
    #Consider each slice independently
    sampling.insert(0, sys.maxsize)

fixed_bin = distance_transform_edt(fixed_bin, sampling=sampling)
fixed_image = sitk.GetImageFromArray(fixed_bin.astype("float32"))

sitk.WriteImage(fixed_image, outputname)
