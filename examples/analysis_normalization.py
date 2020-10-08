import argparse
import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import numpy as np
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI imzML")
parser.add_argument("-p", "--prominence", help="Prominence factor for peak selection")

args = parser.parse_args()
inputname = args.input
prominence = float(args.prominence)

p = io.open_imzml(inputname)
spectra = io.get_spectra(p)

mz, I = spectra[0]
min_diff = mz[1] - mz[0]
wlen = max(10, int(50.0 / min_diff))
indices_mzs = sp.spectra_peak_indices_adaptative(spectra, factor=prominence, wlen=wlen)

width_peak_indices = 10

tic = []
medians = []
sic = []
sic_comp = []

for i, (x,y) in enumerate(spectra):
    indices = indices_mzs[i]
    indices = np.unique(np.array([int(max(0, min(ind+i, y.shape[0]-1))) for ind in indices for i in range(-width_peak_indices//2, width_peak_indices//2)], dtype=np.int64))
    mask = np.zeros(y.shape, dtype=bool)
    mask[indices] = True
    y_without_indices = y[~mask]
    y_indices = y[mask]
    tic.append(np.sum(y))
    sic.append(np.sum(y_without_indices))
    sic_comp.append(np.sum(y_indices))
    medians.append(np.median(y))

p_tic = pearsonr(np.array(tic), np.array(medians))[0]
p_sic = pearsonr(np.array(sic), np.array(medians))[0]
p_sic_comp = pearsonr(np.array(sic_comp), np.array(medians))[0]

print("Pearson correlation TIC", p_tic)
print("Pearson correlation SIC (only baseline)", p_sic)
print("Pearson correlation SIC complementary (only peaks)", p_sic_comp)
