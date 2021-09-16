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

imzml = io.open_imzml(inputname)
mz, I = imzml.getspectrum(0)
spectra = io.get_full_spectra(imzml)
msimage = MSImage(spectra, image=None, coordinates=imzml.coordinates, tolerance=tolerance)
new_image = msimage.copy()
new_image.is_maybe_densify = False

indices = []
N = msimage.shape[-1]
for i in range(N):
    if i%10000 == 0: print(i)
    current_image = msimage.get_ion_image_index(i)
    n = np.count_nonzero(current_image)
    proportion = n / np.prod(msimage.shape[:-1])
    if proportion < percentage:
        indices.append(i)

print(indices)
new_image = np.delete(new_image, indices, axis=-1)

# spectra = np.ones((100, 2, 1000))
# spectra[:, 0, ...] = np.arange(1000)+1
# image = np.ones((10, 10, 1000))
new_image = MSImage(spectra, image, tolerance=tolerance)
print(len(new_image.mzs))

intensities, coordinates = io.get_spectra_from_images(new_image)
print(len(intensities), len(intensities[0]))
print("Spectra")
s = new_image.spectra[:, 0, ...]
print(s.shape)
io.write_imzml(s, intensities, coordinates, outputname)
