import csv
import argparse
import os
import csv
import src.spectraprocessing as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import white_tophat
from scipy.ndimage import gaussian_filter
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from itertools import *
import re
import timeit
import time

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def intersection_spectra(theoretical, observed, tol):
    I = []
    for i in range(observed.shape[0]):
        o = np.array(observed[i])
        t = theoretical[i][0]
        o = o[~np.isnan(o)]
        t = np.array(t)
        intersection_th = t[(np.abs(o[:, None] - t) < tol).any(0)]
        fraction_intersection = len(intersection_th) * 1.0 / len(t)
        I.append(fraction_intersection)
    return I

def average_diffs_common_peaks(theoretical, observed, tol):
    keys = np.unique([elem for t in theoretical for elem in t[0]])
    D = {k:[] for k in keys}
    # D = {k:[] for k
    for i in range(observed.shape[0]):
        o = observed[i]
        t = theoretical[i][0]
        o = o[~np.isnan(o)]
        t = np.array(t)
        intersection_observed = o[(np.abs(t[:, None] - o).argmin(axis=1))]
        for j in range(len(t)):
            k = t[j]
            v = intersection_observed[j]
            D[k].append(v)
    diffs = []
    for k,v in D.items():
        diffs.append(np.abs(np.mean(v) - k))
    return diffs

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def build_spectra(inputdir):
    spectra = []
    i = 0
    for filename in sorted_alphanumeric(os.listdir(inputdir)):
        with open(inputdir + os.path.sep + filename) as f:
            data = list(csv.reader(f, delimiter=" "))
            masses = [float(data[i][0]) for i in range(1, len(data))]
            intensities = [float(data[i][1]) for i in range(1, len(data))]
            spectra.append([masses, intensities])
        i+=1
    return np.array(spectra)

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

inputdir = args.input
theoreticaldir = args.theoretical


spectra = build_spectra(inputdir)
spectra = sp.same_mz_axis(spectra)
spectra_bc = []
index = 0
for x,y in spectra:
    str_el = np.repeat([1], 20)
    I = gaussian_filter(y, 1)
    I = white_tophat(I, None, str_el)
    # plt.plot(x, I)
    # plt.show()
    spectra_bc.append([x, I])
    index += 1


spectra_bc = np.array(spectra_bc)
theoretical = build_spectra(theoreticaldir)
time_prominence, mzs = timeit.timeit(lambda: sp.spectra_peak_mzs_adaptative(spectra_bc, 1.15, 3000), number=1)

diffs = average_diffs_common_peaks(theoretical, mzs, 70)
inters_prominence = intersection_spectra(theoretical, mzs, 70)
size_prominence = [len(m) for m in mzs]
recall_prominence = np.mean(inters_prominence)
precision_prominence = len(theoretical) / np.mean(size_prominence)
print(diffs)
print(np.mean(diffs))
print(inters_prominence, size_prominence)
print(time_prominence, precision_prominence, recall_prominence)

time_cwt, mzs_cwt = timeit.timeit(lambda: sp.spectra_peak_mzs_cwt(spectra_bc, 1.45, [1, 2, 5, 10, 20, 50]), number=1)
inters_cwt = intersection_spectra(theoretical, mzs_cwt, 70)
diffs_cwt = average_diffs_common_peaks(theoretical, mzs_cwt, 70)
size_cwt = [len(m) for m in mzs_cwt]
recall_cwt = np.mean(inters_cwt)
precision_cwt = len(theoretical) / np.mean(size_cwt)
print(diffs_cwt)
print(np.mean(diffs_cwt))
print(inters_cwt, size_cwt)
print(time_cwt, precision_cwt, recall_cwt)

# print(mzs)
