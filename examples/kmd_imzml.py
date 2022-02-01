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
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyimzml.ImzMLParser as imzmlparser
import scipy.signal as signal

from mpldatacursor import datacursor

def display_onclick(x=None, y=None, s=None, z=None, label=None, **kwargs):
    """
    On-click function called
    to display annotations
    """
    annotation = kwargs["point_label"]
    try:
        annotation = '\n'.join(str(name) for a in annotation for name in a)
    except TypeError:
        annotation = '\n'.join(str(a) for a in annotation)
    label = ""
    if str(x) != annotation:
        label += str(x) + "\n"
    label += annotation
    return label

def onclick(event, datacursors):
    """
    On-click function which clears
    datacursors and prevents
    annotations from lingering
    """
    for dc in datacursors:
        if dc:
            dc.hide()

def compute_kendrick_mass_defect(peak_maps, r):
    """
    Mass defect function
    """
    mzs = np.array([peak_map["mz"] for peak_map in peak_maps])
    kendrick_mass = mzs * round(r) / r
    kendrick_mass_defect = kendrick_mass - np.floor(kendrick_mass)
    return mzs, kendrick_mass_defect

def open_imzml(filename):
    """
    Opening an .imzML file
    with the pyimzml library
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ibd_file = imzmlparser.INFER_IBD_FROM_IMZML
        return imzmlparser.ImzMLParser(filename, ibd_file=ibd_file)

def get_spectra(imzml, pixel_numbers=[]):
    """
    Get spectra from pyimzml.ImzMLParser
    """
    spectra = []
    coordinates = []
    for i in pixel_numbers:
        coordinates.append(imzml.coordinates[i])

    if len(pixel_numbers) == 0:
        coordinates = imzml.coordinates.copy()

    for i, (x, y, z) in enumerate(coordinates):
        mz, ints = imzml.getspectrum(i)
        spectra.append([mz, ints])
    if spectra and not all(len(l[0]) == len(spectra[0][0]) for l in spectra):
        return np.array(spectra, dtype=object)
    return np.array(spectra)

def find_peak_indices(data, prominence, wlen, distance=1):
    """
    Find peak indices based on prominence,
    i.e. the height of the peak relative to
    the nearest higher peak
    """
    intensities = np.ma.masked_invalid(data)
    peak_indices, _ = signal.find_peaks(tuple(data),
                                         prominence=prominence,
                                         wlen=wlen,
                                         distance=distance)
    return peak_indices


def kendricks_plot(ax, peak_maps, r, cut_off, **kwargs):
    """
    Kendrick's plot display
    With interactivity
    """
    global color_index
    color =  kwargs["color"] if "color" in kwargs else None
    label = kwargs["label"] if "label" in kwargs else None
    alpha =  kwargs["alpha"] if "alpha" in kwargs else 1
    size_factor = kwargs["size_factor"] if "size_factor" in kwargs else 1
    size_intensities = kwargs["size_intensities"] if "size_intensities" in kwargs else None
    if not color:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][color_index]

    intensities = np.array([peak_map["intensity"] for peak_map in peak_maps])
    if intensities.size == 0:
        return
    intensities[intensities==0] = np.finfo(float).eps

    mzs, kendrick_mass_defect = compute_kendrick_mass_defect(peak_maps, r)
    threshold = np.max(intensities) * cut_off if cut_off > 0 else 1
    log_intensities = np.log10(intensities/threshold)

    annotation = np.array([peak_map["name"] for peak_map in peak_maps], dtype=object)

    if not size_intensities:
        #Min clipping to avoid 0 valued sizes
        size_intensities = np.clip(log_intensities, np.finfo(float).eps, None)

        #Mask for low intensities
        mask = size_intensities != np.finfo(float).eps
        mzs = mzs[mask]
        kendrick_mass_defect = kendrick_mass_defect[mask]
        size_intensities = np.log10(intensities)[mask]
        annotation = annotation[mask]

        size_intensities *= size_factor

    pc = ax.scatter(mzs, kendrick_mass_defect, s=size_intensities, c=color, picker=True, ec="k", label=label, alpha=alpha)
    #picking events
    dc = datacursor(pc, tolerance=2, formatter=display_onclick, display="single", point_labels=annotation, bbox=None)
    color_index += 1
    return dc


def peaks_to_maps(peaks, peak_names, peak_intensities):
    """
    Arrays of values to arrays of maps (dictionaries)
    containing mz values, ion name (annotation),
    and intensities
    """
    peak_maps = []
    for i, peak in enumerate(peaks):
        molecule_name = str(peak_names[i]).split(", ")
        peak_maps.append({"mz": peaks[i], "name": molecule_name[:3], "intensity": peak_intensities[i]})
    return peak_maps

def off_in_sample(annotation_files):
    """
    Separate the annotation from METASPACE
    between off- and in-sample.

    Remove duplicate annotation
    from different databases
    """
    annotation = []
    off_peaks, in_peaks = [], []
    for annotation_file in annotation_files:
        data = pd.read_csv(annotation_file, header=2, delimiter=",")
        mzs_annotated = data.mz
        molecule_names = data.moleculeNames
        off_sample = data.offSample
        intensities_annotated = data.totalIntensity
        for i, off in enumerate(off_sample):
            mz = mzs_annotated[i]
            if mz in annotation:
                continue
            molecule_name = molecule_names[i].split(", ")
            if off:
                off_peaks.append({"mz": mzs_annotated[i], "name": molecule_name[:3], "intensity": intensities_annotated[i]})
            else:
                in_peaks.append({"mz": mzs_annotated[i], "name": molecule_name[:3], "intensity": intensities_annotated[i]})
            annotation.append(mz)
    return annotation, off_peaks, in_peaks


def imzml_to_mean_spectra(inputname, is_save):
    """
    Opens imzML file and outputs the mean spectrum
    and associated unique m/z ratios
    """
    if not os.path.isfile(inputname + ".npy"):
        imzml = open_imzml(inputname)
        spectra = get_spectra(imzml)
        imzml_mzs = np.hstack(spectra[:, 0])
        I = np.hstack(spectra[:, 1])
        mzs, unique_indices = np.unique(imzml_mzs, return_inverse=True)
        indices_mzs = np.searchsorted(mzs, imzml_mzs)
        mean_spectra = np.zeros(len(mzs))
        N = spectra.shape[0]
        for i, ind in enumerate(indices_mzs):
            mean_spectra[ind] += I[i]

        mean_spectra /= N
        if is_save:
            mz_spectra = np.array([mzs, mean_spectra])
            np.save(inputname, mz_spectra)
    else:
        mzs, mean_spectra = np.load(inputname+".npy")
    return mzs, mean_spectra

def extract_peaks(mzs, mean_spectra, step_ppm, factor_prominence, n_pieces=20):
    """
    Extracts the peaks in the mean spectrum
    Based on the prominence from the baseline signal
    The baseline signal is estimated as median+stddev
    of the intensities over several computation windows
    """
    diff = np.amin(np.diff(mzs))
    step = step_ppm / 1e6
    wlen = int(step * mzs[0] / diff)
    size = mean_spectra.shape[0]
    piece = wlen*10
    stddev_piecewise = np.median([np.nanstd(mean_spectra[max(0,i-piece) : min(i+piece, size-1)]) for i in range(piece*n_pieces)])
    median_piecewise = np.median([np.median(mean_spectra[max(0,i-piece) : min(i+piece, size-1)]) for i in range(piece*n_pieces)])
    threshold_prominence = (median_piecewise + stddev_piecewise) * factor_prominence
    peak_indices = find_peak_indices(mean_spectra, prominence=threshold_prominence, wlen=wlen, distance=wlen)
    print("wlen", wlen, "peak len", len(peak_indices))
    return peak_indices

def not_indices(indices, length):
    """
    Compute the complementary of
    the indices in a range of size "length"
    """
    mask = np.ones(length, dtype=bool)
    mask[indices] = False
    full_indices = np.arange(length, dtype=int)
    return full_indices[mask]

def align_peaks(mzs, intensities, reference_peaks, step_ppm, keep_mzs=True):
    """
    Align mzs in the spectra within a tolerance window
    (defined by the step in ppm) on detected peaks

    The alignment can be done without keeping the "mzs"
    which are not in within a tolerance window of
    "reference_peaks" (keep_mzs=False),
    or by keeping them (keep_mzs=True)
    """
    peaks, peak_intensities = [], []
    indices_peaks_found = np.array([], dtype=int)
    diffs = np.zeros_like(mzs)
    for peak in reference_peaks:
        tolerance = step_ppm / 1e6 * peak
        begin = peak-tolerance
        end = peak+tolerance
        indices = np.where((mzs > begin) & (mzs < end))[0]
        diffs[indices] = peak - mzs[indices]
        intensity = np.sum(intensities[indices])
        peaks += [peak]
        peak_intensities += [intensity]
        indices_peaks_found = np.concatenate((indices_peaks_found, indices))
    if keep_mzs:
        keep_indices = not_indices(indices_peaks_found, len(mzs))
        if indices_peaks_found.size > 0:
            #Find the closest shift and apply it to each remaining peak
            indices_closest = np.searchsorted(mzs[indices_peaks_found], mzs[keep_indices])
            diffs[keep_indices] = diffs[indices_peaks_found[indices_closest]]
            shift_mzs = mzs[keep_indices] + diffs[keep_indices]
        else:
            shift_mzs = mzs[keep_indices]
        peaks = np.concatenate((peaks, shift_mzs))
        peak_intensities = np.concatenate((peak_intensities, intensities[keep_indices]))
    else:
        peaks = np.array(peaks)
        peak_intensities = np.array(peak_intensities)
    return peaks, peak_intensities

def delete_duplicate_peaks(peaks, peak_intensities, annotation_mzs, step_ppm):
    """
    Remove peaks in "peaks" and "peak_intensities"
    that are already in annotation_mzs
    """
    peak_curated = peaks.copy()
    peak_intensities_curated = peak_intensities.copy()
    for mz in annotation_mzs:
        tol = mz * step_ppm / 1e6
        idx = np.where((peaks > mz - tol) & (peaks < mz + tol))
        peaks = np.delete(peaks, idx)
        peak_intensities = np.delete(peak_intensities, idx)
    return peaks, peak_intensities

def display_peaks(mzs, mean_spectra, peak_indices):
    """
    Display peaks over the mean spectrum
    """
    fig1, ax1 = plt.subplots()
    ax1.plot(mzs, mean_spectra)
    ax1.plot(mzs[peak_indices], mean_spectra[peak_indices], "o")
    fig1.show()


def kendricks_plot_with_annotation(ax, inputname, annotation, previous_peak_maps, step_ppm=14, factor_prominence=100, **kwargs):
    """
    Kendrick's plot from the imzML file to the final display:
    1. Peak detection
    2. Peak alignment
    3. Annotation handling
    4. Kendricks plot
    """
    r = kwargs["r"] if "r" in kwargs else 14.0156
    cut_off = kwargs["cut_off"] if "cut_off" in kwargs else 0
    label = kwargs["label"] if "label" in kwargs else ""
    display = kwargs["display"] if "display" in kwargs else False
    is_save = kwargs["is_save"] if "is_save" in kwargs else False

    mzs, mean_spectra = imzml_to_mean_spectra(inputname, is_save)
    peak_indices = extract_peaks(mzs, mean_spectra, step_ppm=step_ppm, factor_prominence=factor_prominence)
    peaks, peak_intensities = align_peaks(mzs, mean_spectra, mzs[peak_indices], step_ppm, keep_mzs=False)

    #Align with peaks from other datasets, if they exist
    previous_mzs = np.array([peak_map["mz"] for peak_map in previous_peak_maps])
    peaks, peak_intensities = align_peaks(peaks, peak_intensities, previous_mzs, step_ppm)

    if display:
        display_peaks(mzs, mean_spectra, peak_indices)

    imzml_peaks = peaks_to_maps(peaks, peaks, peak_intensities)

    off_peaks, in_peaks = [], []
    if annotation:
        annotation_mzs, off_peaks, in_peaks = off_in_sample(annotation)

    dc_imzml = kendricks_plot(ax, imzml_peaks, r, cut_off, label=label, size_factor=size_factor, alpha=0.5)

    return dc_imzml, imzml_peaks, off_peaks, in_peaks



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML", action="append", type=str, default=[])
parser.add_argument("-r", "--r_exact_mass", help="Exact mass of group (default m_CH2=14.0156)", default=14.0156)
parser.add_argument("-a", "--annotation", help="Annotation file provided by METASPACE (.csv)", action="append", nargs="*", type=str, default=[])
parser.add_argument("-c", "--cut_off", help="Cut-off to discard lowest intensities in resulting Kendrick's plot. Expressed as a percentage to compute a threshold = cut_off * max_intensity.", default=0)
parser.add_argument("-s", "--size_factor", help="Size factor for points in the Kendricks plot", default=20)
parser.add_argument("-f", "--factor_prominence", help="Prominence factor to determine peaks, i.e. the prominence threshold is this value multiplied by the signal baseline", default=100)
parser.add_argument("-p", "--step_ppm", help="Tolerance step in ppm", default=14)
parser.add_argument("--save", help="Save mean spectrum as .npy file to avoid parsing the full imzML file", action="store_true")
args = parser.parse_args()

input_names = args.input
r = float(args.r_exact_mass)
annotation_names = args.annotation
cut_off = float(args.cut_off)
size_factor = float(args.size_factor)
factor_prominence = float(args.factor_prominence)
is_save = args.save
step_ppm = float(args.step_ppm)

color_index = 0

fig, ax = plt.subplots()
ax.set_xlabel("m/z")
ax.set_ylabel("KMD")

fig.patch.set(alpha=0)
ax.patch.set(alpha=0)

datacursors = []
off_peaks, in_peaks = [], []

previous_peaks = []
for i, input_name in enumerate(input_names):
    name = os.path.splitext(os.path.basename(input_name))[0]
    annotation_name = annotation_names if len(annotation_names) == 0 or type(annotation_names[i]) != list else annotation_names[i]
    dc_imzml, imzml_current, off_current, in_current = kendricks_plot_with_annotation(ax, input_name, annotation_name, previous_peaks, step_ppm=step_ppm, factor_prominence=factor_prominence, r=r, cut_off=cut_off, label=name, is_save=is_save, display=True)
    previous_peaks += imzml_current
    off_peaks += off_current
    in_peaks += in_current
    datacursors.append(dc_imzml)

dc_off = kendricks_plot(ax, off_peaks, r, cut_off, label="Off-sample",  color="tab:gray", size_intensities=3**2)
dc_in = kendricks_plot(ax, in_peaks, r, cut_off, label="In-sample", color="k", size_intensities=3**2)

datacursors += [dc_off, dc_in]

cid = fig.canvas.mpl_connect('pick_event', lambda event: onclick(event, datacursors))

ax.legend()
plt.show()
