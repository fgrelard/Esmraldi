import numpy as np
import SimpleITK as sitk
import argparse
import os
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

def compute_complementary(dirname):
    print(dirname)
    out_name = dircur + os.path.sep + "Matrix_Tape.tif"
    if os.path.exists(out_name):
        os.remove(out_name)
    for imagename in os.listdir(dirname):
        if (imagename.endswith(".tif")):
            image = sitk.GetArrayFromImage(sitk.ReadImage(dirname + os.path.sep + imagename))
            shape = image.shape
            break

    summed_image = np.zeros(shape)

    for imagename in os.listdir(dirname):
        if (imagename.endswith(".tif")):
            image = sitk.GetArrayFromImage(sitk.ReadImage(dirname + os.path.sep + imagename))
            summed_image += image

    summed_image = np.where(summed_image > 0, 255, 0)
    summed_image = binary_fill_holes(summed_image)

    out_image = ((1 - summed_image)*255).astype(np.uint8)

    plt.imshow(out_image)
    plt.show()

    sitk.WriteImage(sitk.GetImageFromArray(out_image), out_name)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input image directory")
args = parser.parse_args()

inputname = args.input

dirname = os.path.dirname(inputname)

for root, dirs, files in os.walk(dirname):
    if "resized" in dirs:
        dircur = root + os.path.sep + "resized"
        compute_complementary(dircur)
