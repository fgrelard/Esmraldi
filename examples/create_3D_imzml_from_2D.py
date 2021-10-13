"""
Convert several 2D imzML files from a 3D imzML file
"""

import argparse
import random
import esmraldi.imzmlio as io

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 2D imzml")
parser.add_argument("-o", "--output", help="Output 3D imzml")
parser.add_argument("-d", "--dimension", help="Size of 3rd dimension", default=10)
args = parser.parse_args()

inputname = args.input
outputname = args.output
size_3D = int(args.dimension)

imzml2D = io.open_imzml(inputname)
coordinates = imzml2D.coordinates

new_coordinates = coordinates.copy()
mz, I = imzml2D.getspectrum(0)

mzs = [mz] * (size_3D * len(coordinates))
intensities = list(io.get_spectra_intensities(imzml2D))

for i in range(2,size_3D+1):
    for elem in coordinates:
        current_elem = (elem[0], elem[1], i)
        new_coordinates.append(current_elem)
        rand = random.randrange(0, len(coordinates))
        _, I = imzml2D.getspectrum(rand)
        intensities.append(I)



io.write_imzml(mzs, intensities, new_coordinates, outputname)
