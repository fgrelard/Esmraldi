import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

def spectra_sum(spectra):
    spectra_sum = [0 for i in range(len(spectra[1][0]))]
    for x, y in spectra:
        spectra_sum = np.add(spectra_sum, y)
    return spectra_sum

def spectra_mean(spectra):
    spectra_mean = [0 for i in range(len(spectra[1][0]))]
    for x, y in spectra:
        spectra_mean = np.add(spectra_mean, y)
    spectra_mean /= len(spectra)
    return spectra_mean

def peak_indices(data):
    peak_indices, _ =  signal.find_peaks(tuple(data),
                                         prominence=50,
                                         wlen=10,
                                         distance=1)
    return peak_indices

def peak_reference_indices(indices):
    counts = np.bincount(indices)
    indices_second = peak_indices(counts)
    return indices_second

def peak_selection(x, y):
    i = peak_indices(y)
    return x[i], y[i]

def normalization_tic(y):
    sum_intensity = 0
    for intensity in y:
        sum_intensity += intensity
    return y / sum_intensity


def closest_peak(peak, references):
    for peak_reference in references:
        cnp.amin(references - peak)
