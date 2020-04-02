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

import src.imzmlio as imzmlio
import src.segmentation as seg
import src.speciesrule as sr
import src.spectrainterpretation as si
import src.fusion as fusion
from src.theoreticalspectrum import TheoreticalSpectrum


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
annotation = si.annotation(mzs, theoretical_spectrum.spectrum, 0.5)

list_names = [v for v in annotation.values() if v is not None]
list_names_str = '\n'.join(list_names)
pattern = re.compile(r".*_(.*)")
unique_matches = set(pattern.findall(list_names_str))
unique_matches.add("AX")
for add in unique_matches:
    s = r".*_" + add
    if add == "AX":
        s = r"AX[0-9]+\b(?!_)"
    pattern = re.compile(s)
    matches = pattern.findall(list_names_str)
    mz_matches = [k for k, v in annotation.items() for m in matches if v == m]
    condition = np.in1d(mzs, np.array(mz_matches))
    image_reduced = np.mean(image[..., condition], axis=2)
    name = outdir + os.path.sep + add + ".png"
    plt.imshow(image_reduced, cmap="gray")
    plt.axis("off")
    plt.savefig(name, bbox_inches='tight')
    plt.close()
