import esmraldi.segmentation as seg
import esmraldi.imageutils as utils
import esmraldi.imzmlio as io
import SimpleITK as sitk
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input spectrum")
parser.add_argument("-o", "--output", help="Output spectrum")
parser.add_argument("-f", "--factor", help="Scale factor")
args = parser.parse_args()

input_name = args.input
outname = args.output
factor = float(args.factor)

imzml = io.open_imzml(input_name)
original_spectra = io.get_spectra(imzml)
mzs = original_spectra[:, 0, :]

spectra = imzmlio.get_full_spectra(imzml)
max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))
itk_image = sitk.GetImageFromArray(image)

size = image.shape[:2]

new_size = [d*factor for d in size]
new_size = new_size[::-1]
resized_itk = utils.resize(itk_image, new_size)

resized_image = sitk.GetArrayFromImage(resized_itk)
intensities, coordinates = io.get_spectra_from_images(resized_image)

mzs = mzs[:len(intensities)]

resized_spectra = np.array((mzs, intensities))
resized_spectra = np.transpose(resized_spectra, (1, 0, 2))

mzs = resized_spectra[:, 0, :]
intensities = resized_spectra[:, 1, :]

io.write_imzml(mzs, intensities, coordinates, outname)
