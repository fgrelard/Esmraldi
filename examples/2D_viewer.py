"""
Very basic 2D viewer, allowing to pick pixels
and select m/z
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import esmraldi.imzmlio as imzmlio
from esmraldi.spectralviewer import SpectralViewer

def onclick(event):
    x,y = int(event.xdata), int(event.ydata)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input ITK image or imzML file")
parser.add_argument("--memmap", help="Create and read a memmap file", action="store_true")

args = parser.parse_args()

inputname = args.input
is_memmap = args.memmap

if inputname.lower().endswith(".imzml"):
    memmap_dir = os.path.dirname(inputname) + os.path.sep + "mmap" + os.path.sep
    memmap_basename = os.path.splitext(os.path.basename(inputname))[0]
    memmap_image_filename = memmap_dir + memmap_basename + ".npy"
    memmap_spectra_filename = memmap_dir + memmap_basename + "_spectra.npy"
    memmap_files_exist = (os.path.exists(memmap_dir)
                          and os.path.exists(memmap_image_filename)
                          and os.path.exists(memmap_spectra_filename))

    if is_memmap and  memmap_files_exist:
        print("Reading from memmap")
        spectra = np.load(memmap_spectra_filename, mmap_mode="r")
        image = np.load(memmap_image_filename, mmap_mode="r")
    else:
        imzml = imzmlio.open_imzml(inputname)
        mz, I = imzml.getspectrum(0)
        spectra = imzmlio.get_full_spectra(imzml)
        max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
        max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
        max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
        image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))

        if is_memmap:
            os.makedirs(memmap_dir, exist_ok=True)
            np.save(memmap_image_filename, image)
            np.save(memmap_spectra_filename, spectra)

if len(image.shape) == 4:
    image = image[0, ...]

image = image.transpose((1, 0, 2))
# print(spectra.shape)
fig, ax = plt.subplots(3, 1)
tracker = SpectralViewer(ax, image, spectra)
fig.canvas.mpl_connect('button_press_event', tracker.onclick)
plt.show()
