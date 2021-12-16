import numpy as np
import sys
import argparse
import esmraldi.imzmlio as imzmlio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuitCV
import io
import matplotlib.pyplot as plt
from mmappickle.dict import mmapdict
from mmappickle.stubs import EmptyNDArray
from esmraldi.sparsematrix import SparseMatrix
import esmraldi.spectraprocessing as sp
import time
import scipy.signal as signal
import scipy.ndimage as ndimage

def create_groups(mzs, indices):
    groups = []
    for i in range(len(indices)):
        if i == 0:
            first = 0
        second = indices[i]
        if second != len(mzs):
            previous_second = mzs[second-1]
            next_second = mzs[second+1]
            current_second = mzs[second]
            is_closest_previous = abs(previous_second-current_second) < abs(next_second-current_second)
            if is_closest_previous:
                second += 1
        groups.append(mzs[first:second])
        first = second
    return groups


def associated_groups_sublevel(groups, level, current_group_index):
    group_level = groups[level]
    group_sublevel = groups[level-1]
    print(len(group_level), len(group_sublevel), group_level[:5], group_sublevel[:5])
    current_group = group_level[current_group_index]
    start = sum(len(g) for g in group_level[:current_group_index])
    end = start+len(current_group)
    print(start, end)
    associated_groups = group_sublevel[start:end]
    print(current_group, associated_groups)


def find_peaks(peak_hierarchy, group_hierarchy, threshold_tolerance):
    levels = len(group_hierarchy)
    for level in range(levels-1, 0, -1):
        current_group = group_hierarchy[level]
        current_peaks = peak_hierarchy[level]
        diff_resolution = np.array([np.mean(abs(g - np.median(g))) for g in current_group])
        print("diff resolution", diff_resolution)
        condition_resolution = np.where(diff_resolution < threshold_tolerance)[0]
        print(current_peaks, current_group)
        selected_peaks = current_peaks[condition_resolution]
        print(selected_peaks)
        print(condition_resolution)
    return selected_peaks



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input mmap")
parser.add_argument("-t", "--threshold", help="Threshold in spectral resolution")

args = parser.parse_args()
inputname = args.input

print("Open imzML")
mdict = mmapdict(inputname)
mean_spectra = mdict["mean_spectra"]
mzs = np.unique(mdict["unique"])

plt.plot(mzs, mean_spectra)
plt.show()

peak_indices = sp.peak_indices(mean_spectra, prominence=0, wlen=1000)
m, I = mzs[peak_indices], mean_spectra[peak_indices]
print(m.shape, I.shape)

m = np.arange(100)+0.1
I = np.arange(100)%10
I[::4] *= 2
print(I)
super_groups = []
super_peaks = []
while len(I) > 0:
    ind = signal.argrelextrema(I, np.greater)[0]
    if I[0] > I[1]:
        ind = np.insert(ind, 0, 0)
    if I[-1] > I[-2]:
        ind = np.append(ind, len(I)-1)
    ind_min = signal.argrelextrema(I, np.less)[0]
    ind_min = np.append(ind_min, len(m))
    groups = create_groups(m, ind_min)

    print(len(groups), len(ind))

    super_groups.append(groups)
    super_peaks.append(m)
    diff_groups = [np.mean(abs(g - np.median(g))) for g in groups]
    m, I = m[ind], I[ind]

peaks = find_peaks(super_peaks, super_groups, 3)
# associated_groups_sublevel(super_groups, 1, 0)

# print(mzs.shape, mean_spectra.shape)
# plt.plot(mzs, mean_spectra)

# plt.plot(m, I, "o")
# plt.show()
