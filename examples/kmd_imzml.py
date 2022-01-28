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
import matplotlib
#to get blending effects
matplotlib.use("module://mplcairo.qt")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import esmraldi.imzmlio as imzmlio
import esmraldi.spectraprocessing as sp

from mplcairo import operator_t
from mpldatacursor import datacursor

def display_onclick(x=None, y=None, s=None, z=None, label=None, **kwargs):
    annotation = kwargs["point_label"]
    try:
        annotation = '\n'.join(str(name) for a in annotation for name in a)
    except TypeError:
        annotation = '\n'.join(str(a) for a in annotation)
    label = ""
    if x != annotation:
        label += str(x) + "\n"
    label += str(annotation)
    return label

def compute_kendrick_mass_defect(peak_maps, r):
    mzs = np.array([peak_map["mz"] for peak_map in peak_maps])
    kendrick_mass = mzs * round(r) / r
    kendrick_mass_defect = kendrick_mass - np.floor(kendrick_mass)
    return mzs, kendrick_mass_defect

def kendricks_plot(ax, peak_maps, r, cut_off, **kwargs):
    color =  kwargs["color"] if "color" in kwargs else "r"
    label = kwargs["label"] if "label" in kwargs else None
    size_factor = kwargs["size_factor"] if "size_factor" in kwargs else 1
    intensities = np.array([peak_map["intensity"] for peak_map in peak_maps])
    if intensities.size == 0:
        return

    mzs, kendrick_mass_defect = compute_kendrick_mass_defect(peak_maps, r)
    threshold = np.max(intensities) * cut_off
    log_intensities = np.log10(intensities/threshold)

    annotation = np.array([peak_map["name"] for peak_map in peak_maps], dtype=object)

    #Min clipping to avoid 0 valued sizes
    size_intensities = size_factor*np.clip(log_intensities, np.finfo(float).eps, None)

    pc = ax.scatter(mzs, kendrick_mass_defect, s=size_intensities, c=color, picker=True, ec=None, label=label)
    #blending colors
    operator_t.EXCLUSION.patch_artist(pc)
    #picking events
    dc = datacursor(pc, formatter=display_onclick, display="single", point_labels=annotation, bbox=None)
    return dc

def onclick(event, datacursors):
    for dc in datacursors:
        if dc:
            dc.hide()

def off_in_sample(annotation_files):
    annotation = []
    off_peaks, in_peaks = [], []
    for annotation_file in annotation_files:
        data = pd.read_csv(annotation_file, header=2, delimiter=",")
        mzs_annotated = data.mz
        molecule_names = data.moleculeNames
        off_sample = data.offSample
        intensities_annotated = data.totalIntensity
        for i in range(len(off_sample)):
            mz = mzs_annotated[i]
            if mz in annotation:
                continue
            molecule_name = molecule_names[i].split(", ")
            off = off_sample[i]
            if off:
                off_peaks.append({"mz": mzs_annotated[i], "name": molecule_name[:3], "intensity": intensities_annotated[i]})
            else:
                in_peaks.append({"mz": mzs_annotated[i], "name": molecule_name[:3], "intensity": intensities_annotated[i]})
            annotation.append(mz)
    return annotation, off_peaks, in_peaks



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-r", "--r_exact_mass", help="Exact mass of group (default m_CH2=14.0156)", default=14.0156)
parser.add_argument("-a", "--annotation", help="Annotation file provided by METASPACE (.csv)", nargs="+", type=str, default=[])
parser.add_argument("-c", "--cut_off", help="Cut-off to discard lowest intensities in resulting Kendrick's plot. Expressed as a percentage to compute a threshold = cut_off * max_intensity.", default=0)
parser.add_argument("-s", "--size_factor", help="Size factor for points in the Kendricks plot", default=20)
args = parser.parse_args()

inputname = args.input
r = float(args.r_exact_mass)
annotation = args.annotation
cut_off = float(args.cut_off)
size_factor = float(args.size_factor)

if not os.path.isfile(inputname + ".npy"):
    imzml = imzmlio.open_imzml(inputname)
    spectra = imzmlio.get_spectra(imzml)
    imzml_mzs = np.hstack(spectra[:, 0])
    I = np.hstack(spectra[:, 1])
    mzs, unique_indices = np.unique(imzml_mzs, return_inverse=True)
    indices_mzs = np.searchsorted(mzs, imzml_mzs)
    mean_spectra = np.zeros(len(mzs))
    N = spectra.shape[0]
    for i, ind in enumerate(indices_mzs):
        mean_spectra[ind] += I[i]

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

# plt.plot(mzs, mean_spectra)
# plt.plot(mzs[peak_indices], mean_spectra[peak_indices], "o")
# plt.show()
print("wlen", wlen, "peak len", len(peak_indices))
peaks, peak_intensities = [], []
# for i in peak_indices:
#     peak = mzs[i]
#     tolerance = step_ppm / 1e6 * peak
#     begin = peak-tolerance
#     end = peak+tolerance
#     indices = np.where((mzs > begin) & (mzs < end))[0]
#     intensity = np.sum(mean_spectra[indices])
#     peaks += [peak]
#     peak_intensities += [intensity]

peaks = np.array(peaks)
peak_intensities = np.array(peak_intensities)

peaks = mzs[peak_indices]
peak_intensities = mean_spectra[peak_indices]

imzml_peaks = []
for i, peak in enumerate(peaks):
    imzml_peaks.append({"mz": peak, "name": peak, "intensity": peak_intensities[i]})

if annotation:
    annotation_mzs, off_peaks, in_peaks = off_in_sample(annotation)
    for mz in annotation_mzs:
        tol = mz * step
        idx = np.where((peaks > mz - tol) & (peaks < mz + tol))
        peaks = np.delete(peaks, idx)
        peak_intensities = np.delete(peak_intensities, idx)


fig, ax = plt.subplots()
ax.set_xlabel("m/z")
ax.set_ylabel("KMD")

fig.patch.set(alpha=0)
ax.patch.set(alpha=0)

dc_imzml = kendricks_plot(ax, imzml_peaks, r, cut_off, color="r", label="imzML", size_factor=size_factor)
dc_in = kendricks_plot(ax, in_peaks, r, cut_off, color="g", label="In-sample", size_factor=size_factor)
dc_off = kendricks_plot(ax, off_peaks, r, cut_off, color="b", label="Off-sample", size_factor=size_factor)

cid = fig.canvas.mpl_connect('pick_event', lambda event: onclick(event, [dc_imzml, dc_in, dc_off]))

plt.show()
