import esmraldi.imzmlio as io
import argparse
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-a", "--annotation", help="Annotation file provided by METASPACE (.csv)")
args = parser.parse_args()

input_name = args.input
annotation_name = args.annotation

data = pd.read_csv(annotation_name, header=2, delimiter=",")
mzs_annotated = data.mz
msm = data.msm

mzs_annotated, ind = np.unique(mzs_annotated, return_index=True)
msm = msm[ind]

print("Theoretical", mzs_annotated.shape, data.mz.shape)


if input_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
    spectra = io.get_spectra(imzml)
    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
    print(len(mzs))
else:
    mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv")

print("Observed", mzs.shape, mzs[:100])

diffs = np.argmin(np.abs(mzs - mzs_annotated[:, None]), axis=-1)
intersection = np.abs(mzs[diffs] - mzs_annotated) < 1e-2
percentage = np.count_nonzero(intersection)/len(mzs_annotated)
percentage *= 100
missing = mzs_annotated[~intersection]
missing_msm = msm[~intersection]
print("Found", percentage, "% peaks from Metaspace")
print("Missing", missing)
np.set_printoptions(suppress=True)
print("Missing MSM", np.vstack((missing, missing_msm)).T)
