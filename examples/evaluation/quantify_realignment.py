"""
Quantify realignement of spectra
with various measures
"""
import numpy as np
import skimage.restoration as restoration
import src.spectraprocessing as sp
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
import scipy.signal as signal

def complete_sum(spectra):
    """
    Intensity sum
    across all spectra

    Parameters
    ----------
    spectra: np.ndarray
        spectra

    Returns
    ----------
    float
        intensity sum

    """
    sum = 0
    for x, y in spectra:
        sum = np.add(sum, np.sum(y))
    return sum

def full_ratio(reduced, full):
    """
    Measure of lost information
    between reduced spectra and full
    spectra

    Parameters
    ----------
    reduced: float
        measure on reduced spectra
    full: float
        measure on full spectra

    Returns
    ----------
    float
        ratio
    """
    return reduced * 1.0 / full

def estimate_noise_ratio(spectra):
    """
    Estimates the SNR
    in MALDI spectra

    Parameters
    ----------
    spectra: np.ndarray
        spectra

    Returns
    ----------
    tuple
        noise level, signal median
    """
    sigma = restoration.estimate_sigma(spectra)
    median = np.median(spectra)
    return sigma, median

def estimate_noise_proportion(y, median, sigma):
    """
    Proportion of points in a spectrum
    that are considered noise
    that is to say their intensity
    is lower than signal median + noise

    Parameters
    ----------
    y: list
        intensities
    median: float
        signal median
    sigma: float
        noise level
    """
    total_count = 0
    for elem in y:
        if elem <= median + sigma:
            total_count += 1
    total_count /= 1.0 * len(y)
    return total_count


def distance_indices(indices1, indices2):
    """
    Distances between closest indices in
    two lists

    Parameters
    ----------
    indices1: np.ndarray
        first list of indices
    indices2: np.ndarray
        second list of indices

    Returns
    ----------
    list
        lowest distances between indices
    """
    l = []
    for val in indices1:
        idx = (np.abs(indices2 - val)).argmin()
        closest_value = indices2[idx]
        d = abs(val - closest_value)
        l.append(d)
    return l

def realign_close_peaks(array1, array2):
    """
    Get a list of close peaks in two arrays

    Parameters
    ----------
    array1: np.ndarray
        first array
    array2: np.ndarray
        second array

    Returns
    ----------
    list
        close peaks

    """
    l = []
    for val in array1:
        idx = (np.abs(array2 - val)).argmin()
        peak_index = array2[idx]
        if abs(peak_index - val) < 10:
            l.append(peak_index)
    return l


def precision(array1, array2):
    """
    Precision : |a1 \cap a2|/|a2|

    Parameters
    ----------
    array1: np.ndarray
        first array
    array2: np.ndarray
        second array

    Returns
    ----------
    float
        precision

    """
    dset_a1 = OrderedSet(array1)
    set_a2 = OrderedSet(array2)
    inters = set_a1.intersection(set_a2)
    return len(inters) * 1.0 / len(set_a1)

def recall(array1, array2):
     """
    Recall : |a1 \cap a2|/|a1|

    Parameters
    ----------
    array1: np.ndarray
        first array
    array2: np.ndarray
        second array

    Returns
    ----------
    float
        recall

    """
    set_a1 = OrderedSet(array1)
    set_a2 = OrderedSet(array2)
    inters = set_a1.intersection(set_a2)
    print(len(set_a1), " ", len(inters), " ", len(set_a2))
    return len(inters) * 1.0 / len(set_a2)

def missing_indices(array1, array2):
    """
    Set difference between array2 and array1

    Parameters
    ----------
    array1: np.ndarray
        first array
    array2: np.ndarray
        second array

    Returns
    ----------
    list
        set difference

    """
    set_a1 = OrderedSet(array1)
    set_a2 = OrderedSet(array2)
    diff = set_a2.difference(set_a1)
    return list(diff)


def extract_indices_from_mz(mzs, x):
    """
    From a list of mzs, extract
    corresponding indices in x

    Parameters
    ----------
    mzs: np.ndarray
        reference mass to charge list
    x: list
        observed mass to charge list

    Returns
    ----------
    list
        indices of x associated to mzs
    """
    l = []
    for i in range(len(x)):
        if x[i] in mzs:
            l.append(i)
    return l

averagine = {'C': 7.0, 'H': 11.8333, 'N': 0.5, 'O': 5.16666}
realigned_spectra = np.load("data/peaksel_250DJ_prominence75.npy")
realigned_spectra = sp.deisotoping_simple(realigned_spectra, nb_charges=2, average_distribution={})
print(realigned_spectra.shape)
sum_realigned = complete_sum(realigned_spectra)
snoise_realigned, median_realigned = estimate_noise_ratio(realigned_spectra[:, 1, :])

full_spectra =  np.load("data/old/spectra.npy")
indices_realigned = extract_indices_from_mz(realigned_spectra[0, 0, :], full_spectra[0,0,:])

distance_between_mz = full_spectra[0,0,1] - full_spectra[0,0,0]
prominence = 50

indices_full_spectra = np.load("data/old/indices_full.npy")
unique_indices = np.unique(indices_full_spectra)

spectra_max_before = np.load("data/old/spectra_max.npy")
indices_spectra_max = sp.peak_indices(spectra_max_before, 5)

print(len(spectra_max_before), " ", len(indices_spectra_max))
plt.plot(full_spectra[0][0], spectra_max_before, full_spectra[0,0, indices_realigned], np.array(spectra_max_before[indices_realigned]), ".")
plt.show()
exit(0)
ui_realigned = realign_close_peaks(indices_realigned, indices_spectra_max)
ism_realigned = realign_close_peaks(indices_spectra_max, indices_realigned)
p = precision(ui_realigned, ism_realigned)
r = recall(ui_realigned, ism_realigned)
missing = missing_indices(indices_realigned, indices_spectra_max)
peaks_missing = spectra_max_before[missing]
print("FullSpectra min=", full_spectra[:, 1, :].min(), ", max=", full_spectra[:, 1, :].max(), ", mean=", np.mean(full_spectra[:, 1, :]), ", median=", np.median(full_spectra[:, 1, :]))
print("Precision=", p, " recall=", r)
print("Missing indices=", len(missing), " max=", peaks_missing.max(), " median=", np.median(peaks_missing), " mean=", np.mean(peaks_missing), " stddev=", np.std(peaks_missing))



#print(indices_spectra_max)
#realign = sp.realign(full_spectra, prominence)
#print(len(realign[0][0]))

# d_unique_max = distance_indices(unique_indices, indices_spectra_max)
# d_max_unique = distance_indices(indices_spectra_max, unique_indices)

# print("Length unique, max", len(unique_indices), " ", len(indices_spectra_max))
# print("Distance indices = ", max(d_unique_max), " maxunique, ", max(d_max_unique))


snoise_full, median_full = estimate_noise_ratio(full_spectra[:, 1, :])
sum_full = complete_sum(full_spectra)
snoise_max, median_max = estimate_noise_ratio(spectra_max_before)

print("Std noise max=",snoise_full, " Median max=", median_full)
noise_proportion = estimate_noise_proportion(peaks_missing, median_full, 3*median_full)

print("Noise ratio=", noise_proportion)
r = full_ratio(sum_realigned, sum_full)
print("Sum full=", sum_full, " sum_realigned=", sum_realigned, " ratio=", r)
