import SimpleITK as sitk
import numpy as np
import argparse
import os
from skimage.color import rgb2gray

def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))[..., 0]
    mask = mask.T
    return mask

def invert(image):
    ones = np.count_nonzero(image)
    zeroes = image.size - ones
    if ones > zeroes:
        return 255 - image
    return image


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input dir", nargs="+", type=str)
args = parser.parse_args()

input_names = args.input
output_dir = os.path.dirname(input_names[0])  + os.path.sep + "converted" + os.path.sep

os.makedirs(output_dir, exist_ok=True)

for input_name in input_names:
    name = os.path.basename(input_name)
    image = read_image(input_name)
    image = invert(image)
    sitk.WriteImage(sitk.GetImageFromArray(image.T), output_dir + name)
