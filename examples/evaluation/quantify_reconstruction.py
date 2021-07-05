import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import esmraldi.imageutils as utils
import esmraldi.imzmlio as imzmlio

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MRI image")
parser.add_argument("-r", "--reconstruction", help="Reconstructed MRI image")
parser.add_argument("--number_slice", help="Number of the slice to process", default=-1)

args = parser.parse_args()

inputname = args.input
reconstructedname = args.reconstruction
number_slice = int(args.number_slice)

input_mri = sitk.ReadImage(inputname)
reconstructed_mri = sitk.ReadImage(reconstructedname)


if number_slice >= 0:
    if input_mri.GetDimension() >= 3 and number_slice < input_mri.GetSize()[-1]:
        input_mri = input_mri[:,:,number_slice]
    if reconstructed_mri.GetDimension() >= 3 and number_slice < reconstructed_mri.GetSize()[-1]:
        reconstructed_mri = reconstructed_mri[:,:,number_slice]


input_mri_array = sitk.GetArrayFromImage(input_mri)
input_mri_array = imzmlio.normalize(input_mri_array)
input_mri = sitk.GetImageFromArray(input_mri_array)

reconstructed_mri_array = sitk.GetArrayFromImage(reconstructed_mri)
reconstructed_mri_array = imzmlio.normalize(reconstructed_mri_array)
reconstructed_mri = sitk.GetImageFromArray(reconstructed_mri_array)
print(reconstructed_mri_array.shape, input_mri_array.shape)

mse_stddev = utils.mse_stddev(input_mri, reconstructed_mri)
print("MSE from std images=", np.sqrt(mse_stddev))
