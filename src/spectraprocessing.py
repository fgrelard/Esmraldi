import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


def spectra_sum(spectra):
    spectra_sum = [0 for i in range(len(spectra[0][0]))]
    for x, y in spectra:
        spectra_sum = np.add(spectra_sum, y)
    return spectra_sum

def spectra_mean(spectra):
    spectra_mean = [0 for i in range(len(spectra[0][0]))]
    for x, y in spectra:
        spectra_mean = np.add(spectra_mean, y)
    spectra_mean /= len(spectra)
    return spectra_mean

def spectra_max(spectra):
    spectra_max = [0 for i in range(len(spectra[0][0]))]
    for x, y in spectra:
        spectra_max = np.maximum(spectra_max, y)
    return spectra_max

def spectra_peak_indices(spectra, prominence=50):
    indices = []
    for spectrum in spectra:
        x, y = spectrum
        indices_current = peak_indices(y, prominence)
        indices = indices + indices_current.tolist()
    return indices

def peak_indices(data, prominence=50):
    peak_indices, _ =  signal.find_peaks(tuple(data),
                                         prominence=prominence,
                                         wlen=10,
                                         distance=1)
    return peak_indices

def peak_reference_indices(indices):
    counts = np.bincount(indices)
    max_value = counts.max()
    indices_second = peak_indices(counts,4)
    return indices_second

def peak_selection(x, y):
    i = peak_indices(y)
    return x[i], y[i]

def normalization_tic(y):
    sum_intensity = 0
    for intensity in y:
        sum_intensity += intensity
    return y / sum_intensity


def width_peak_indices(indices, full_indices):
    indices_to_width = {}
    for index in indices:
        right = index
        left = index
        while (right in full_indices or right+1 in full_indices) and (right == index or right not in indices):
            right += 1
        while (left in full_indices or left-1 in full_indices) and (left == index or left not in indices):
            left -=1
        if left in indices:
            right = left
        elif right in indices:
            left = right
        width = (right - left)/2
        indices_to_width[index] = width
    return indices_to_width


def closest_peak(num, indices_to_width):
    mz = min(indices_to_width.keys(), key=lambda k: abs(k-num))
    width = indices_to_width[mz]
    return mz, width

def realign(spectra, prominence=50):
    full_indices = spectra_peak_indices(spectra, prominence)
    unique_indices = np.unique(full_indices)
    aligned_indices = peak_reference_indices(full_indices)
    indices_to_width = width_peak_indices(aligned_indices, full_indices)
    realigned_spectra = []
    for spectrum in spectra:
        x, y = spectrum
        x_realigned = np.copy(x)
        y_realigned = np.copy(y)
        indices = peak_indices(y, prominence)
        for i in indices:
            mz, width = closest_peak(i, indices_to_width)
            if (i != mz and i >= mz - width and i <= mz + width):
                y_realigned[mz] = max(y[i], y_realigned[mz])
        x_realigned = x_realigned[aligned_indices]
        y_realigned = y_realigned[aligned_indices]
        realigned_spectra.append((x_realigned, y_realigned))
    return realigned_spectra
