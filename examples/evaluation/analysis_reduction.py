"""
Extracts loadings and scores
For a given dimension reduction method
(NMF or PCA)
"""
import csv
import argparse
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import esmraldi.imzmlio as imzmlio
import esmraldi.segmentation as seg
import esmraldi.speciesrule as sr
import esmraldi.spectrainterpretation as si
import esmraldi.fusion as fusion

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF, PCA
from esmraldi.theoreticalspectrum import TheoreticalSpectrum

from esmraldi.sliceviewer import SliceViewer


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image")
parser.add_argument("-m", "--mzs", help="MZS corresponding to MALDI image")
parser.add_argument("-t", "--theoretical", help="Theoretical spectrum")
parser.add_argument("-o", "--output", help="Output csv file(s)")
parser.add_argument("-n", "--n", help="Number of components for dimension reduction method")
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold (optional)", default=0)

parser.add_argument("-p", "--preprocess", help="Whether to normalize or not", action="store_true")
parser.add_argument("-f", "--nmf", help="Use NMF instead of PCA", action="store_true")
args = parser.parse_args()


inputname = args.input
mzsname = args.mzs
theoretical_name = args.theoretical
outname = args.output
threshold = int(args.threshold)
n = int(args.n)
is_normalized = args.preprocess
is_nmf = args.nmf

outroot, outext = os.path.splitext(outname)
print(inputname, mzsname, theoretical_name, outname, is_normalized, is_nmf)
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
    if mzsname:
        with open(mzsname) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            mzs = [float(row[0]) for row in csv_reader]
    else:
        mzs = [i for i in range(image.shape[2])]
    mzs = np.asarray(mzs)

image = image[..., mzs >= threshold]
mzs = mzs[mzs>=threshold]
mzs = np.around(mzs, decimals=2)

if theoretical_name:
    species = sr.json_to_species(theoretical_name)
    ions = [mol for mol in species if mol.category=="Ion"]
    adducts = [mol for mol in species if mol.category=="Adduct"]
    modifications = [mol for mol in species if mol.category=="Modification"]
    theoretical_spectrum = TheoreticalSpectrum(ions, adducts, modifications)
    print(len(ions), "ions,", len(adducts), "adducts")

else:
    theoretical_spectrum = TheoreticalSpectrum([],[],[])

if is_normalized:
    image = imzmlio.normalize(image)
else:
    image = np.uint8(image)

image_shape = image.shape[:-1]
image_norm = fusion.flatten(image, is_spectral=True)
M = image_norm.T
print(M.shape)


if is_nmf:
    nmf = NMF(n_components=n, init='nndsvda', solver='cd', random_state=0)
    fit_nmf = nmf.fit(M)
    eigenvectors = fit_nmf.components_ #H
    eigenvalues = nmf.fit_transform(M); #W
    inverse_transform = nmf.inverse_transform(eigenvalues)
    eigenvectors_transposed = eigenvalues.T
else:
    # p, n = M.shape
    pca = PCA(n)
    fit_pca = pca.fit(M)
    eigenvectors = fit_pca.components_
    eigenvalues = fit_pca.transform(M)
    inverse_transform = pca.inverse_transform(eigenvalues)
    eigenvectors_transposed = eigenvalues.T


mse = mean_squared_error(M, inverse_transform, multioutput='raw_values')
outlier_indices = [i for i in range(len(mse))]
outlier_indices.sort(key=lambda x:mse[x], reverse=True)

number_outliers = 10
# for i in range(number_outliers):
#     outlier_index = outlier_indices[i]
#     image_original = M[..., outlier_index]
#     image_original = np.reshape(image_original, (np.prod(image_shape[:-1]), image_shape[-1]))
#     image_reconstructed = inverse_transform[..., outlier_index]
#     image_reconstructed = np.reshape(image_reconstructed, (np.prod(image_shape[:-1]), image_shape[-1]))

#     mse_slice = mean_squared_error(image_original, image_reconstructed, multioutput='raw_values')
#     mse_slice = ((image_original-image_reconstructed)**2).sum(axis=1)
#     outlier_slice_indices = [i for i in range(len(mse_slice))]
#     outlier_slice_indices.sort(key=lambda x:mse_slice[x], reverse=True)
#     print(mzs[outlier_index], " slices=", outlier_slice_indices)



image_eigenvectors = eigenvectors_transposed.T
new_shape = image_shape + (image_eigenvectors.shape[-1],)
image_eigenvectors = image_eigenvectors.reshape(new_shape)
print(image_eigenvectors.shape)

weights = eigenvectors[..., 0] / np.sum(eigenvectors[..., 0])
image_0 = fusion.reconstruct_image_from_components(image_eigenvectors, weights)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image[..., 0])
ax[1].imshow(image_0)
plt.show()

print(image_eigenvectors.shape)

tables = []
for i in range(eigenvectors.shape[0]):
    current_name = outroot + "_" + str(i)
    current_image = image_eigenvectors[..., i].T
    itk_image = sitk.GetImageFromArray(current_image)
    if itk_image.GetPixelID() >= sitk.sitkFloat32:
        itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)
    sitk.WriteImage(itk_image, current_name + ".tif")
    plt.stem(mzs, eigenvectors[i], use_line_collection=True)
    plt.savefig(current_name + "_eigenvectors.png", bbox_inches="tight")
    plt.close()
    descending_indices = eigenvectors[i].argsort()[::-1]
    descending_scores = eigenvectors[i, descending_indices]
    descending_mzs = mzs[descending_indices]
    descending_names = si.annotation(descending_mzs, theoretical_spectrum.spectrum, 0.1)
    descending_names = {k:(v if len(v) > 0 else "?") for k,v in descending_names.items()}
    table = np.column_stack(np.array([list(descending_names.keys()), list(descending_names.values()), descending_scores.tolist()], dtype=object))
    tables.append(table)

tables_array = np.array(tables).transpose((1, 0, 2))
tables_reshaped = tables_array.reshape(tables_array.shape[0],-1)
header = "".join([str(i)+(";")*tables_array.shape[2] for i in range(tables_array.shape[1])]) + "\n" + ("mz;name;score;") * tables_array.shape[1]
np.savetxt(outname, tables_reshaped, delimiter=";", fmt="%s", header=header, comments="")
