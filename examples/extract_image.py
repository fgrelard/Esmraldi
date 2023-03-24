import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.utils as utils
import esmraldi.imageutils as imageutils
from esmraldi.msimagefly import MSImageOnTheFly
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
output_name = args.output

if input_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
    spectra = io.get_spectra(imzml)
    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]
    shape = (max_x, max_y, max_z)

    root = os.path.splitext(input_name)[0]
    npy_path = root + ".npy"
    npy_indexing_path = root + "_indexing.npy"
    mean_spectra = None
    indexing = None
    if os.path.isfile(npy_path):
        mean_spectra = np.load(npy_path)
    if os.path.isfile(npy_indexing_path):
        indexing = np.load(npy_indexing_path, mmap_mode="r")
    msimage = MSImageOnTheFly(spectra=spectra, coords=coordinates, tolerance=14, mean_spectra=mean_spectra, indexing=indexing, spectral_axis=-1)
    msimage.spectral_axis = 0
    msimage = msimage.transpose((2, 1, 0))

mz = 746.711
tol = utils.tolerance(mz, msimage.tolerance, msimage.is_ppm)
image = msimage.get_ion_image_mzs(mz, tl=tol, tr=tol)
logimage = np.log(image, where=image>0)
plt.imshow(logimage)
plt.show()
sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(logimage.T), sitk.sitkFloat32), output_name)
