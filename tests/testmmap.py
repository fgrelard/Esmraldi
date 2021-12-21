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
from treelib import Node, Tree

def create_groups(mzs, intensities, indices):
    new_mzs, new_intensities = [], []
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
        new_mzs.append(mzs[first:second])
        new_intensities.append(np.mean(intensities[first:second]))
        first = second
    groups = [new_mzs, new_intensities]
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


def update_delete_indices(indices, group):
    cumsum_len = np.concatenate(([0], np.cumsum([len(g) for g in group])))
    current_to_delete = [slice(cumsum_len[i], cumsum_len[i+1]) for i in indices]
    indices_to_delete = [i for item in current_to_delete for i in range(item.start, item.stop)]
    return indices_to_delete

def create_tree(group_hierarchy):
    levels = len(group_hierarchy)
    tree = Tree()
    tree.create_node(None, identifier="-1,0")
    cumsumlen = np.array([])
    for level in range(levels):
        print(level)
        G = group_hierarchy[levels-level-1]
        current_group, I = G
        arange = np.arange(len(current_group))
        parents = np.searchsorted(cumsumlen, arange)
        print(current_group, parents)
        incr = 0
        for i, group in enumerate(current_group):
            for j, elem in enumerate(group):
                tree.create_node(identifier=str(level)+","+str(incr), parent=str(level-1)+","+str(i), data=elem)
                incr += 1
        tree.show()
        cumsumlen = np.concatenate(([0], np.cumsum([len(g) for g in current_group])))
    tree.show()




def find_peaks(peak_hierarchy, group_hierarchy, threshold_tolerance):
    levels = len(group_hierarchy)
    indices_to_delete = []
    new_mzs, new_intensities = np.array([]), np.array([])
    for level in range(levels-1)[::-1]:
        print(level)
        G = group_hierarchy[level]
        current_group, I = G
        current_group = [current_group[i] for i in range(len(current_group)) if i not in indices_to_delete]
        current_peaks = peak_hierarchy[level+1]
        current_peaks = np.array([current_peaks[i] for i in range(len(current_peaks)) if i not in indices_to_delete])
        I = np.array([I[i] for i in range(len(I)) if i not in indices_to_delete])
        diff_resolution = np.array([np.sum(abs(g - g[len(g)//2]))//(len(g)-1) for g in current_group])
        condition_resolution = np.where(diff_resolution < threshold_tolerance)[0]
        new_mzs = np.concatenate((new_mzs, current_peaks[condition_resolution]))
        new_intensities = np.concatenate((new_intensities, I[condition_resolution]))
        indices_to_delete = update_delete_indices(indices_to_delete, original_group)
        indices_to_delete = update_delete_indices(condition_resolution, current_group)

        print(indices_to_delete)
    # plt.plot(new_mzs, new_intensities)
    # plt.show()
    return new_mzs, new_intensities


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input mmap")
parser.add_argument("-t", "--threshold", help="Threshold in spectral resolution")

args = parser.parse_args()
inputname = args.input

print("Open imzML")
mdict = mmapdict(inputname)
mean_spectra = mdict["mean_spectra"]
mzs = np.unique(mdict["unique"])

peak_indices = sp.peak_indices(mean_spectra, prominence=0, wlen=1000)
m, I = mzs[peak_indices], mean_spectra[peak_indices]
print(m.shape, I.shape)

original_m = np.arange(100)+0.1
original_I = np.arange(100)%10
original_I[::4] *= 2
# original_m = mzs
# original_I = mean_spectra
m, I = original_m, original_I
print(I)
super_groups = []
super_peaks = []
while len(I) > 1:
    ind = signal.argrelextrema(I, np.greater)[0]
    if I[0] > I[1]:
        ind = np.insert(ind, 0, 0)
    if I[-1] > I[-2]:
        ind = np.append(ind, len(I)-1)
    ind_min = signal.argrelextrema(I, np.less)[0]
    ind_min = np.append(ind_min, len(m))
    groups = create_groups(m, I, ind_min)
    super_groups.append(groups)
    super_peaks.append(m)
    m, I = m[ind], I[ind]
    # plt.plot(original_m, original_I)
    # plt.plot(m, I, "o")
    # plt.show()

create_tree(super_groups)
new_mzs, new_intensities = find_peaks(super_peaks, super_groups, 3.5)
plt.plot(original_m, original_I)
plt.plot(new_mzs, new_intensities, "o")
plt.show()
# associated_groups_sublevel(super_groups, 1, 0)

# print(mzs.shape, mean_spectra.shape)
# plt.plot(mzs, mean_spectra)

# plt.plot(m, I, "o")
# plt.show()
