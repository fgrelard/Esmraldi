import argparse

import numpy as np
import esmraldi.imzmlio as io

from esmraldi.msimage import MSImage


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML file")
parser.add_argument("-o", "--output", help="Output imzML file")
parser.add_argument("-t", "--tolerance", help="m/z tolerance", default=0.001)
parser.add_argument("-p", "--percentage", help="Threshold for the proportion of pixels", default=0.01)

args = parser.parse_args()


inputname = args.input
outputname = args.output
tolerance = float(args.tolerance)
percentage = float(args.percentage)
inputname="/mnt/d/CouplageMSI-Immunofluo/Scan rate 37° line/20210112_107x25_20um_Mouse_Spleen_DAN_Neg_mode_200-2000mz_70K_Laser37_6p5kV_350C_Slens90_Line_centroid.imzml"
imzml = io.open_imzml(inputname)
mz, I = imzml.getspectrum(0)
spectra = io.get_full_spectra(imzml)
msimage = MSImage(spectra, image=None, coordinates=imzml.coordinates, tolerance=tolerance)
new_image = msimage.copy()
new_image.tolerance = 0
new_image.set_densify(False)

# indices = []
# N = msimage.shape[-1]
# for i in range(N):
#     if i%10000 == 0: print(i)
#     current_image = msimage.get_ion_image_index(i)
#     n = np.count_nonzero(current_image)
#     proportion = n / np.prod(msimage.shape[:-1])
#     if proportion < percentage:
#         indices.append(i)
#         break

indices = np.loadtxt("list_ind.txt", dtype=np.int64, delimiter=",")

new_image = np.delete(new_image, indices, axis=-1)
new_image.set_densify(True)
# spectra = np.ones((100, 2, 1000))
# spectra[:, 0, ...] = np.arange(1000)+1
# image = np.ones((10, 10, 1000))
# new_image = MSImage(spectra, image, tolerance=tolerance)
# print(new_image.spectra.shape, len(new_image.mzs))

intensities, coordinates = io.get_spectra_from_images(new_image)
print(len(intensities), len(intensities[0]))
print("Spectra")
s = new_image.spectra[:, 0, ...]
print(s.shape)
print(new_image.nnz, msimage.nnz)
io.write_imzml(s, intensities, coordinates, outputname)
