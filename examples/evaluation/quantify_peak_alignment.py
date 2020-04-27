import csv
import argparse
import os
import csv
import src.spectraprocessing as sp
import numpy as np
import matplotlib.pyplot as plt
from itertools import *
import re
import timeit
import time
import src.imzmlio as io


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


# p = io.open_imzml(inputname)
# size = len(p.coordinates)
# spectra = io.get_spectra_intensities(p)
# np.save("data/intensities_250DJ.npy", spectra)
# spectra = io.get_spectra_mzs(p)
# np.save("data/mzs_250DJ.npy", spectra)


spectra =  np.load("data/full_spectra_250DJ.npy", mmap_mode='r')
# peak_selected = sp.spectra_peak_indices(spectra, prominence=75, wlen=60)
# peak_selected = spectra[0][peak_selected]
# np.save("data/peakselected_250DJ_p75_wlen60.npy", peak_selected)
# exit(0)
peak_selected = np.load("data/peakselected_250DJ_f100_wlen60.npy", allow_pickle=True)
mz, I = spectra[0]
diff = mz[1] - mz[0]
realigned_spectra = sp.realign_mzs(spectra, peak_selected, reference="frequence", nb_occurrence=4, step=0.053)
np.save("data/realigned_250DJ.npy", realigned_spectra)
print(realigned_spectra.shape)
print(spectra.shape)
print(peak_selected.shape)
# realigned_spectra = sp.realign_median(spectra, factor=5, nb_occurrence=4, step=0.05)
# print(realigned_spectra.shape)
