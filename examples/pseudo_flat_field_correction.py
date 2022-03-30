import esmraldi.imageutils as utils
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input ITK format")
parser.add_argument("-s", "--sigma", help="Blur sigma")
parser.add_argument("-o", "--output", help="Output ITK format")
args = parser.parse_args()

input_name = args.input
sigma = float(args.sigma)
output_name = args.output

image = sitk.GetArrayFromImage(sitk.ReadImage(input_name))
corrected_image = utils.pseudo_flat_field_correction(image, sigma)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[1].imshow(corrected_image)
plt.show()
