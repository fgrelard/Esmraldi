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
    for i in range(observed.shape[0]):
        o = observed[i]
        o = o[~np.isnan(o)]
        t = np.array(theoretical)
        intersection_th = t[(np.abs(o[:, None] - t) < tol).any(0)]
        fraction_intersection = len(intersection_th) * 1.0 / len(t)
        I.append(fraction_intersection)
    return I

def average_diffs_common_peaks(theoretical, observed, tol):
    D = {k:[] for k in theoretical}
    # D = {k:[] for k
    for i in range(observed.shape[0]):
        o = observed[i]
        o = o[~np.isnan(o)]
        t = np.array(theoretical)
        intersection_th = t[(np.abs(o[:, None] - t) < tol).any(0)]

        intersection_observed = o[(np.abs(intersection_th[:, None] - o).argmin(axis=1))]
        for j in range(len(intersection_th)):
            k = intersection_th[j]
            v = intersection_observed[j]
            D[k].append(v)
    diffs = []
    for k,v in D.items():
        diffs.append(np.abs(np.mean(v) - k))
    return diffs


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


p = io.open_imzml(inputname)
size = len(p.coordinates)
# r = range(size//2 - 50, size//2 + 50)
r = range(size//2 - 50, size//2 + 50)
spectra = io.get_spectra(p, r)
tolerance = 0.5

time_prominence, mzs = timeit.timeit(lambda: sp.spectra_peak_mzs_adaptative(spectra, 12, 200), number=1)
diffs = average_diffs_common_peaks(theoretical, mzs, tolerance)
inters_prominence = intersection_spectra(theoretical, mzs, tolerance)
size_prominence = [len(m) for m in mzs]
recall_prominence = np.mean(inters_prominence)
precision_prominence = len(theoretical) / np.mean(size_prominence)
print(diffs)
print(np.mean(diffs))
print(inters_prominence, size_prominence)
print(time_prominence, precision_prominence, recall_prominence)


time_cwt, mzs_cwt = timeit.timeit(lambda: sp.spectra_peak_mzs_cwt(spectra, 0.95, [10, 20, 30, 40, 50, 100]), number=1)
inters_cwt = intersection_spectra(theoretical, mzs_cwt, tolerance)
diffs_cwt = average_diffs_common_peaks(theoretical, mzs_cwt, tolerance)
size_cwt = [len(m) for m in mzs_cwt]
recall_cwt = np.mean(inters_cwt)
precision_cwt = len(theoretical) / np.mean(size_cwt)
print(diffs_cwt)
print(np.mean(diffs_cwt))
print(inters_cwt, size_cwt)
print(time_cwt, precision_cwt, recall_cwt)
# print(s.shape)
