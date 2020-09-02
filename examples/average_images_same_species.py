"""
Generates average images of species
with the same modifications

Example:
Species= Mol1, Modification1, Modification2

Average images generated are:
Mol1+Modification1 and Mol1+Modification2 and
Mol1+Modification1+Modification2
"""
import re
import argparse
import os
import csv

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import esmraldi.imzmlio as imzmlio
import esmraldi.segmentation as seg
import esmraldi.speciesrule as sr
import esmraldi.spectrainterpretation as si
import esmraldi.fusion as fusion
from esmraldi.theoreticalspectrum import TheoreticalSpectrum


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image")
parser.add_argument("-m", "--mzs", help="MZS corresponding to MALDI image")
parser.add_argument("-t", "--theoretical", help="Theoretical spectrum")
parser.add_argument("-o", "--output", help="Output directory file(s)")
args = parser.parse_args()


inputname = args.input
mzsname = args.mzs
theoretical_name = args.theoretical
outdir = args.output

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

mzs = np.around(mzs, decimals=2)
species = sr.json_to_species(theoretical_name)
ions = [mol for mol in species if mol.category=="Ion"]
adducts = [mol for mol in species if mol.category=="Adduct"]
modifications = [mol for mol in species if mol.category=="Modification"]

theoretical_spectrum = TheoreticalSpectrum(ions, adducts, modifications)
annotation = si.annotation(mzs, theoretical_spectrum.spectrum, 0.5)

list_names = [v[0] for v in annotation.values() if len(v) > 0]
list_names_str = '\n'.join(list_names)
pattern_species = re.compile(r"(.*?)[0-9]*_.*")
pattern_adducts = re.compile(r".*_(.*)")
unique_matches = set(pattern_species.findall(list_names_str))
unique_matches = unique_matches.union(set(pattern_adducts.findall(list_names_str)))
for add in unique_matches:
    s = r".*" + add + ".*"
    pattern = re.compile(s)
    matches = pattern.findall(list_names_str)
    mz_matches = [k for k, v in annotation.items() for m in matches if len(v) > 0 and v[0] == m]
    condition = np.in1d(mzs, np.array(mz_matches))
    image_reduced = np.mean(image[..., condition], axis=2)
    name = outdir + os.path.sep + add + ".png"
    plt.imshow(image_reduced, cmap="gray")
    plt.axis("off")
    plt.savefig(name, bbox_inches='tight')
    plt.close()
