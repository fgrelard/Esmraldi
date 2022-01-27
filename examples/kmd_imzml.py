# -*- coding: utf-8 -*-

# Copyright 2021 Florent Grélard
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import argparse
import numpy as np
import matplotlib
#to get blending effects
matplotlib.use("module://mplcairo.qt")
import matplotlib.pyplot as plt
from mplcairo import operator_t
from mpldatacursor import datacursor
import esmraldi.imzmlio as imzmlio
import esmraldi.spectraprocessing as sp

def display_onclick(**kwargs):
    label = kwargs["label"]
    label = label.strip("_collection")
    return "mz:" + str(kwargs["x"])+ ", label:" + label

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-r", "--r_exact_mass", help="Exact mass of group (default m_CH2=14.0156)", default=14.0156)
parser.add_argument("-a", "--annotation", help="Annotation file provided by METASPACE (.csv)")
args = parser.parse_args()

inputname = args.input
r = float(args.r_exact_mass)
annotation = args.annotation

if not os.path.isfile(inputname + ".npy"):
    imzml = imzmlio.open_imzml(inputname)
    spectra = imzmlio.get_spectra(imzml)
    all_mzs = np.hstack(spectra[:, 0])
    I = np.hstack(spectra[:, 1])
    mzs, unique_indices = np.unique(all_mzs, return_inverse=True)
    indices_mzs = np.searchsorted(mzs, all_mzs)
    counts = np.zeros(len(mzs))
    mean_spectra = np.zeros(len(mzs))
    N = spectra.shape[0]
    print(N)
    for i, ind in enumerate(indices_mzs):
        mean_spectra[ind] += I[i]
        # counts[ind] += 1

    mean_spectra /= N
    mz_spectra = np.array([mzs, mean_spectra])
    np.save(inputname, mz_spectra)
else:
    mzs, mean_spectra = np.load(inputname+".npy")


step_ppm = 14
n_pieces = 100
factor_prominence = 100
diff = mzs[1] - mzs[0]
step = step_ppm / 1e6
wlen = int(step * mzs[-1] / diff)
size = mean_spectra.shape[0]
stddev_piecewise = np.median([np.nanstd(mean_spectra[max(0,i-wlen) : min(i+wlen, size-1)]) for i in range(wlen*n_pieces)])
median_piecewise = np.median([np.median(mean_spectra[max(0,i-wlen) : min(i+wlen, size-1)]) for i in range(wlen*n_pieces)])
threshold_prominence = (median_piecewise + stddev_piecewise) * factor_prominence
peak_indices = sp.peak_indices(mean_spectra, prominence=threshold_prominence, wlen=wlen, distance=wlen)

plt.plot(mzs, mean_spectra)
plt.plot(mzs[peak_indices], mean_spectra[peak_indices], "o")
plt.show()
print("wlen", wlen, "peak len", len(peak_indices))
peaks, peak_intensities = [], []
for i in peak_indices:
    peak = mzs[i]
    tolerance = step_ppm / 1e6 * peak
    begin = peak-tolerance
    end = peak+tolerance
    indices = np.where((mzs > begin) & (mzs < end))[0]
    intensity = np.sum(mean_spectra[indices])
    peaks += [peak]
    peak_intensities += [intensity]

peaks = np.array(peaks)
peak_intensities = np.array(peak_intensities)

annotation_dict = {}
if annotation:
    import pandas as pd
    data = pd.read_csv(annotation, header=2, delimiter=",")
    mzs_annotated = data.mz
    molecule_names = data.moleculeNames
    off_sample = data.offSample
    print(off_sample)
    off_peaks, in_peaks = [], []
    for i in range(len(off_sample)):
        molecule_name = molecule_names[i].split(", ")
        off = off_sample[i]
        print(off)
        if off:
            off_peaks.append({"mz": mzs_annotated[i], "name": molecule_name[:3]})
        else:
            in_peaks.append({"mz": mzs_annotated[i], "name": molecule_name[:3]})



print(off_peaks, in_peaks)

np.array([peaks, peak_intensities])

plt.plot(peaks, peak_intensities)
plt.show()

kendrick_mass = peaks * round(r) / r
kendrick_mass_defect = kendrick_mass - np.floor(kendrick_mass)

#Colors for different ROIs
colors = ["r", "g", "b"]

fig, ax = plt.subplots()
ax.set_xlabel("m/z")
ax.set_ylabel("KMD")

fig.patch.set(alpha=0)
ax.patch.set(alpha=0)

#Arbitrary threshold to make low intensity points not apparent in the resulting KMD plot
threshold = 1000
log_intensities = np.log10(peak_intensities/threshold)

#Min clipping to avoid 0 valued sizes
size_intensities = 20*np.clip(log_intensities, np.finfo(float).eps, None)

print(kendrick_mass_defect.shape, len(annotation_dict.keys()))
for i in range(peaks.shape[-1]):
    pc = ax.scatter(peaks, kendrick_mass_defect, s=size_intensities, c=colors[0], picker=True, ec=None)
    #blending colors
    operator_t.EXCLUSION.patch_artist(pc)
    #picking events
    datacursor(pc, formatter=display_onclick)

plt.show()
