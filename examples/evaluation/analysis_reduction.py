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
from sklearn.decomposition import NMF
from src.theoreticalspectrum import TheoreticalSpectrum


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image")
parser.add_argument("-m", "--mzs", help="MZS corresponding to MALDI image")
parser.add_argument("-t", "--theoretical", help="Theoretical spectrum")
parser.add_argument("-o", "--output", help="Output csv file(s)")
parser.add_argument("-n", "--n", help="Number of components for dimension reduction method")
parser.add_argument("-p", "--preprocess", help="Whether to normalize or not", action="store_true")
parser.add_argument("-f", "--nmf", help="Use NMF instead of PCA", action="store_true")
args = parser.parse_args()


inputname = args.input
mzsname = args.mzs
theoretical_name = args.theoretical
outname = args.output
n = int(args.n)
is_normalized = args.preprocess
is_nmf = args.nmf

outroot, outext = os.path.splitext(outname)

if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    image = imzmlio.to_image_array(imzml)
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

mzs = np.around(mzs, decimals=2)
species = sr.json_to_species(theoretical_name)
ions = [mol for mol in species if mol.category=="Ion"]
adducts = [mol for mol in species if mol.category=="Adduct"]
modifications = [mol for mol in species if mol.category=="Modification"]
theoretical_spectrum = TheoreticalSpectrum(ions, adducts, modifications)
print(len(ions), "ions,", len(adducts), "adducts")

if is_normalized:
    image = imzmlio.normalize(image)
else:
    image = np.uint8(image)

image_shape = (image.shape[0], image.shape[1])
image_norm = seg.preprocess_pca(image)
M = image_norm.T
print(M.shape)

if is_nmf:
    nmf = NMF(n_components=n, init='nndsvda', solver='cd', random_state=0)
    fit_nmf = nmf.fit(M)
    eigenvectors = fit_nmf.components_ #H
    eigenvalues = nmf.fit_transform(M); #W
    eigenvectors_transposed = eigenvalues.T
else:
    # p, n = M.shape
    fit_pca = fusion.pca(M, n)

    eigenvectors = fit_pca.components_
    eigenvalues = fit_pca.transform(M)
    eigenvectors_transposed = eigenvalues.T

print(eigenvectors.shape, " ", eigenvalues.shape)

image_eigenvectors = eigenvectors_transposed.T
new_shape = image_shape + (image_eigenvectors.shape[-1],)
image_eigenvectors = image_eigenvectors.reshape(new_shape)

tables = []
for i in range(eigenvectors.shape[0]):
    current_name = outroot + "_" + str(i)
    current_image = image_eigenvectors[..., i].T
    im = plt.imshow(current_image, cmap="gray")
    clim = im.properties()['clim']
    plt.colorbar()
    plt.clim(clim)
    plt.axis("off")
    plt.savefig(current_name + ".png", bbox_inches="tight")
    plt.close()
    plt.stem(mzs, eigenvectors[i], use_line_collection=True)
    plt.savefig(current_name + "_eigenvectors.png", bbox_inches="tight")
    plt.close()
    descending_indices = eigenvectors[i].argsort()[::-1]
    descending_scores = eigenvectors[i, descending_indices]
    descending_mzs = mzs[descending_indices]
    descending_names = si.annotation(descending_mzs, theoretical_spectrum.spectrum, 0.1)
    descending_names = {k:(v if len(v) > 0 else "?") for k,v in descending_names.items()}
    table = np.column_stack([list(descending_names.keys()), list(descending_names.values()), descending_scores.tolist()])
    tables.append(table)
    # outfile.write("Eigenvector " + str(i))
    # np.savetxt(outfile, np.array(table), delimiter=";", fmt="%s", header="mz;name;score", comments="")

tables_array = np.array(tables).transpose((1, 0, 2))
tables_reshaped = tables_array.reshape(tables_array.shape[0],-1)
header = "".join([str(i)+(";")*tables_array.shape[2] for i in range(tables_array.shape[1])]) + "\n" + ("mz;name;score;") * tables_array.shape[1]
np.savetxt(outname, tables_reshaped, delimiter=";", fmt="%s", header=header, comments="")

# print(image_eigenvectors.shape)
# plt.imshow(image_eigenvectors[..., 0])
# plt.show()
