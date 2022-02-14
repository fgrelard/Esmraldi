import numpy as np
import scipy.signal as signal

from treelib import Tree, Node

class PeakDetectionTree:
    def __init__(self, mzs, mean_spectrum, step_ppm):
        self.mzs = mzs
        self.mean_spectrum = mean_spectrum
        self.step_ppm = step_ppm

    def create_groups(self, mzs, intensities, indices):
        new_mzs, new_intensities = [], []
        first = indices[0]
        for i in range(len(indices)-1):
            second = indices[i+1]
            if second < len(mzs)-1:
                previous_second = mzs[second-1]
                next_second = mzs[second+1]
                current_second = mzs[second]
                is_closest_previous = abs(previous_second-current_second) < abs(next_second-current_second)
                if is_closest_previous:
                    second += 1
            new_mzs.append(mzs[first:second])
            new_intensities.append(intensities[first:second])
            first = second
        groups = [np.array(new_mzs, dtype=object), np.array(new_intensities, dtype=object)]
        return groups

    def local_minima(self, intensities):
        return ((intensities <= np.roll(intensities, -1)) &
                (intensities < np.roll(intensities, 1)))

    def create_group_hierarchy(self):
        group_hierarchy = []
        m, I = self.mzs, self.mean_spectrum
        while len(I) > 1:
            ind = signal.argrelextrema(I, np.greater)[0]
            ind_min = np.where(self.local_minima(I))[0]
            if I[0] > I[1]:
                ind = np.insert(ind, 0, 0)
                # ind_min = np.insert(ind_min, 0, 0)
            if I[-1] > I[-2]:
                ind = np.append(ind, len(m)-1)
                # ind_min = np.append(ind_min, len(I))
            if ind_min[0] != 0:
                ind_min = np.insert(ind_min, 0, 0)
            if ind_min[-1] == len(m) - 1:
                ind_min[-1] = len(m)
            else:
                ind_min = np.append(ind_min, len(m))
            # if len(I) <= 2:
            #     ind_min = np.array([0, len(I)])
            groups = self.create_groups(m, I, ind_min)
            group_hierarchy.append(groups)
            m, I = m[ind], I[ind]
        return np.array(group_hierarchy, dtype=object)

    def create_tree(self, group_hierarchy):
        print(np.array(group_hierarchy)[:, 0])
        levels = len(group_hierarchy)
        tree = Tree()
        data = type('DataElement', (object,), {'mz':None, 'I':0})()
        tree.create_node(None, identifier="-1,0", data=data)
        cumsumlen = np.array([])
        for level in range(levels-1):
            G = group_hierarchy[levels-level-1]
            current_group, I = G
            arange = np.arange(len(current_group))
            parents = np.searchsorted(cumsumlen, arange)
            incr = 0
            for i, group in enumerate(current_group):
                for j, elem in enumerate(group):
                    data = type('DataElement', (object,), {'mz':elem, 'I': I[i][j]})()
                    tree.create_node(identifier=str(level)+","+str(incr), parent=str(level-1)+","+str(i), data=data)
                    incr += 1
            cumsumlen = np.concatenate(([0], np.cumsum([len(g) for g in current_group])))
        # tree.show(data_property="mz")
        return tree

    def find_levels_threshold(self, group_hierarchy):
        levels = len(group_hierarchy)
        min_diff = np.zeros(levels)
        diffs = {}
        for i, (current_group, _) in enumerate(group_hierarchy):
            current_level = levels-i-1
            current_diffs = self.update_diffs(current_group)
            if min(current_diffs) <= self.step_ppm:
                print(min(current_diffs))
                diffs[current_level] = current_diffs
        if not diffs:
            diffs[levels-1] = current_diffs
            diffs = dict(sorted(diffs.items()))
            print(diffs)
        return diffs

    def update_diffs(self, current_group):
        current_diffs = []
        for mz in current_group:
            average_diff = np.mean(np.diff(mz))
            current_diffs.append(average_diff)
        return current_diffs

    def update_counts(self, current_group):
        return np.concatenate(([int(0)], np.cumsum([len(g) for g in current_group])))


    def not_indices(self, indices, length):
        mask = np.ones(length, dtype=bool)
        mask[indices] = False
        full_indices = np.arange(length, dtype=int)
        return full_indices[mask]

    def update_hierarchy(self, group_hierarchy, level, counts, indices_to_remove):
        levels = len(group_hierarchy)
        if len(indices_to_remove) == 0:
            return group_hierarchy
        new_hierarchy = group_hierarchy.copy()
        next_indices = indices_to_remove.copy()
        for i, (current_group, I) in enumerate(new_hierarchy[:0:-1]):
            current_level = i
            index_hierarchy = levels-1-current_level
            if current_level >= level:
                N = np.sum([len(g) for g in current_group])
                keep_indices = self.not_indices(next_indices, N)
                next_group, next_I = new_hierarchy[index_hierarchy-1]
                new_mzs = next_group[keep_indices]
                new_I = next_I[keep_indices]
                new_hierarchy[index_hierarchy-1] = [new_mzs, new_I]
                if current_level+1 in counts:
                    current_counts = counts[current_level+1]
                    next_indices = [list(range(current_counts[ind], current_counts[ind+1])) for ind in next_indices]
                    next_indices = [item for sublist in next_indices for item in sublist]
        return new_hierarchy


    def find_peaks_group_hierarchy(self, group_hierarchy, diffs_threshold):
        new_hierarchy = group_hierarchy.copy()
        levels = len(group_hierarchy)
        peaks = []
        for i in range(len(new_hierarchy[::-1])):
            level  = levels-1-i
            current_group, I = new_hierarchy[level]
            counts = self.update_counts(current_group)
            diffs = self.update_diffs(current_group)
            to_remove = []
            for j, elem in enumerate(diffs):
                if elem <= self.step_ppm:
                    peaks += current_group[j].tolist()
                    to_remove += list(range(counts[j], counts[j+1]))
            new_hierarchy = self.update_hierarchy(new_hierarchy, i, counts, to_remove)
        print("PEAKS", peaks)
        return peaks

    def extract_peaks(self):
        print("Creating group hierarchy")
        group_hierarchy = self.create_group_hierarchy()
        print("Min diff")
        diff_threshold = self.find_levels_threshold(group_hierarchy)
        print("Finding peaks")
        peaks = self.find_peaks_group_hierarchy(group_hierarchy, diff_threshold)
        return peaks
