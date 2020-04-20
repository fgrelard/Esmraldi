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


def intersection_spectra(theoretical, observed, tol):
    intersection_th = theoretical[(np.abs(observed[:, None] - theoretical) < tol).any(0)]
    return intersection_th

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
    for filename in os.listdir(inputdir):
        with open(inputdir + os.path.sep + filename) as f:
            data = list(csv.reader(f, delimiter=" "))
            masses = [float(data[i][0]) for i in range(1, len(data))]
            intensities = [float(data[i][1]) for i in range(1, len(data))]
            spectra.append([masses, intensities])
        break
    return np.array(spectra)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input spectra")
parser.add_argument("-t", "--theoretical", help="Theoretical spectra")
args = parser.parse_args()

inputdir = args.input
theoreticaldir = args.theoretical


spectra = build_spectra(inputdir)
spectra_bc = []
for x,y in spectra:
    str_el = np.repeat([1], 100)
    I = gaussian_filter(y, 4)
    I = white_tophat(I, None, str_el)
    # plt.plot(x, I)
    # plt.show()
    spectra_bc.append([x, I])

spectra_bc = np.array(spectra_bc)
theoretical = build_spectra(theoreticaldir)
mzs = sp.spectra_peak_mzs_adaptative(spectra_bc, 0.62, 50)


mzs_cwt = sp.spectra_peak_mzs_cwt(spectra, 2.35, [1, 2, 5, 10, 20, 50])

inters_prominence = intersection_spectra(theoretical[0, 0], mzs[0], 70)
inters_cwt = intersection_spectra(theoretical[0, 0], mzs_cwt[0], 70)

print(mzs_cwt.shape, mzs.shape)
print(inters_cwt.shape, inters_prominence.shape, theoretical.shape)
# print(mzs)
