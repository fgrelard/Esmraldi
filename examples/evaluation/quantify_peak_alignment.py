"""
Evaluation of alignment on real data
"""
import csv
import argparse
import os
import csv
import esmraldi.spectraprocessing as sp
import numpy as np
import matplotlib.pyplot as plt
from itertools import *
import re
import timeit
import time
import esmraldi.imzmlio as io

def intersection_spectra(theoretical, observed, tol):
    I = []
    o = observed
    t = np.array(theoretical)
    intersection_th = t[(np.abs(o[:, None] - t) < tol).any(0)]
    fraction_intersection = len(intersection_th) * 1.0 / len(t)
    I.append(fraction_intersection)
    return I

def missing(theoretical, observed, tol):
    o = observed
    t = np.array(theoretical)
    missing_th = t[(np.abs(o[:, None] - t) > tol).all(0)]
    return missing_th

def average_diffs_common_peaks(theoretical, observed, tol):
    D = {k:[] for k in theoretical}
    o = observed
    t = np.array(theoretical)
    intersection_th = t[(np.abs(o[:, None] - t) < tol).any(0)]
    intersection_observed = o[(np.abs(intersection_th[:, None] - o).argmin(axis=1))]
    # print(o)
    # print(intersection_th)
    # print(intersection_observed)
    for j in range(len(intersection_th)):
        k = intersection_th[j]
        v = intersection_observed[j]
        D[k] = v
    diffs = []
    for k,v in D.items():
        diffs.append(np.abs(v - k))
    return diffs

def imzml_to_npy(inputname):
    p = io.open_imzml(inputname)
    size = len(p.coordinates)
    spectra = io.get_spectra_intensities(p)
    np.save("data/intensities_250DJ.npy", spectra)
    spectra = io.get_spectra_mzs(p)
    np.save("data/mzs_250DJ.npy", spectra)

def spectra_to_peaks(peak_name, spectra_name, prominence, wlen):
    spectra =  np.load(spectra_name, mmap_mode='r')
    peak_selected = sp.spectra_peak_mzs_adaptative(spectra, factor=prominence, wlen=wlen)
    np.save(peak_name, peak_selected)

def realign_peaks(realign_name, peak_name, spectra_name, step, occurrence):
    spectra =  np.load(spectra_name, mmap_mode='r')
    peak_selected = np.load(peak_name, allow_pickle=True)
    realigned_spectra = sp.realign_mzs(spectra, peak_selected, reference="frequence", nb_occurrence=occurrence, step=step)
    np.save(realign_name, realigned_spectra)




timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input spectra")
parser.add_argument("-t", "--theoretical", help="Theoretical spectra")
args = parser.parse_args()

inputname = args.input
theoreticalname = args.theoretical

with open(theoreticalname, "r") as f:
    reader = csv.reader(f, delimiter=",")
    theoretical = [float(r[1]) for r in reader]


prominence = 50
wlen = 60
occurrence = 4
step = 0.053
tolerance = 0.2
mean_name = "data/spectra_mean_250DJ.npy"
spectra_name = "data/full_spectra_250DJ.npy"
peak_name = "data/peakselected_250DJ_f"+str(prominence)+"_wlen"+str(wlen)+".npy"
realign_name = "data/realigned_250DJ.npy"

# spectra_to_peaks(peak_name, spectra_name, prominence, wlen)
# realign_peaks(realign_name, peak_name, spectra_name, step, occurrence)
# exit(0)

spectra =  np.load(spectra_name, mmap_mode='r')
peak_selected = np.load(peak_name, allow_pickle=True)
full_mzs = np.hstack(peak_selected)
groups = sp.index_groups(full_mzs, step)
groups  = [group for group in groups if len(group) > occurrence]
realigned_spectra = np.load(realign_name, allow_pickle=True)
refs, I = realigned_spectra[0]
diffs = [np.array(groups[i]) - refs[i] for i in range(len(groups))]
medians = [np.median(d) for d in diffs]
maxs = [np.amax(np.abs(d)) for d in diffs]
mins = [np.amin(np.abs(d)) for d in diffs]
inters_ours = intersection_spectra(theoretical, refs, tolerance)
missing_ours = missing(theoretical, refs, tolerance)
diff_ours = average_diffs_common_peaks(theoretical, refs, tolerance)
mean_spectra = np.load(mean_name)
peak_selected_mean = sp.peak_indices(mean_spectra, prominence=6, wlen=wlen)
mz, I = spectra[0]
# plt.plot(mz, mean_spectra)
# plt.show()
mz_mean = mz[peak_selected_mean]
print(len(refs), len(mz_mean))
inters_mean = intersection_spectra(theoretical, mz_mean, tolerance)
diff_mean = average_diffs_common_peaks(theoretical, mz_mean, tolerance)
print(inters_mean)
print(inters_ours)
print(np.mean(diff_ours))
print(np.mean(diff_mean))
# print(mz_mean)
# print(medians)
# print(maxs)
# print(mins)
# print(full_mzs.shape)
# print(realigned_spectra.shape)
