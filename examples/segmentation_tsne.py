"""
t-SNE segmentation of MS images
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import esmraldi.fusion as fusion
import esmraldi.imzmlio as imzmlio

import SimpleITK as sitk

from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold (optional)", default=0)

args = parser.parse_args()

inputname = args.input
outputname = args.output
threshold = int(args.threshold)

if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    spectra = imzmlio.get_full_spectra(imzml)
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))
    mzs, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(inputname)).T
    mzs = [i for i in range(image.shape[2])]
    mzs = np.asarray(mzs)

image = image[..., mzs >= threshold]
print(image.shape)
mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

shape = image.shape[:-1]

image_norm = imzmlio.normalize(image)
image_norm = fusion.flatten(image_norm, is_spectral=True).T

n=3
X_tsne = TSNE(n_components=n, random_state=0).fit_transform(image_norm)
X_tsne = imzmlio.normalize(X_tsne)
X_tsne = X_tsne.reshape(shape + (3,))

tsne_itk = sitk.GetImageFromArray(X_tsne, isVector=True)
sitk.WriteImage(tsne_itk, outputname)
