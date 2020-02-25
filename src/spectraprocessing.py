import ms_peak_picker
import ms_deisotope
import math
import re
import sys
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from pyopenms import *

class Peak:
    def __init__(self, mz, intensity):
        self.mz = mz
        self.intensity = intensity


def array_to_peaks(array):
    L = []
    mzs = array[0, :]
    intensities = array[1, :]
    for j in range(len(mzs)):
        p = Peak(mzs[j], intensities[j])
        L.append(p)
    return np.array(L)

def peaks_to_array(peaks):
    L = [ [peak.mz for peak in peaks],
          [peak.intensity for peak in peaks] ]
    return np.array(L)

def spectra_to_peaklist(spectra):
    L = [ array_to_peaks(array) for array in spectra ]
    return np.array(L)

def peaklist_to_spectra(peaklist):
    L = [ peaks_to_array(peaks) for peaks in peaklist ]
    return np.array(L)

def spectra_sum(spectra):
    """
    Computes the spectrum from the sum of all spectra

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array

    Returns
    ----------
    np.ndarray
        Sum of spectra

    """
    spectra_sum = [0 for i in range(len(spectra[0][0]))]
    for x, y in spectra:
        spectra_sum = np.add(spectra_sum, y)
    return spectra_sum

def spectra_mean(spectra):
    """
    Computes the average spectrum

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array

    Returns
    ----------
    np.ndarray
        Mean spectrum

    """
    spectra_mean = [0 for i in range(len(spectra[0][0]))]
    for x, y in spectra:
        spectra_mean = np.add(spectra_mean, y)
    spectra_mean /= len(spectra)
    return spectra_mean

def spectra_max(spectra):
    """
    Computes the maximum intensity for each abscissa

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array

    Returns
    ----------
    np.ndarray
        Max spectrum

    """
    spectra_max = [0 for i in range(len(spectra[0][0]))]
    for x, y in spectra:
        spectra_max = np.maximum(spectra_max, y)
    return spectra_max

def spectra_min(spectra):
    """
    Computes the minimum intensity for each abscissa

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array

    Returns
    ----------
    np.ndarray
        Max spectrum

    """
    spectra_min = [sys.maxsize for i in range(len(spectra[0][0]))]
    for x, y in spectra:
        spectra_min = np.minimum(spectra_min, y)
    return spectra_min

def spectra_peak_indices(spectra, prominence=50, wlen=10):
    """
    Estimates and extracts significant peaks in the spectra
    By using the prominence (height of the peak relative to the nearest
    higher peak)

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    prominence: int
        threshold on prominence


    Returns
    ----------
    np.ndarray
        Peak indices relative to spectra
    """
    indices = []
    for spectrum in spectra:
        x, y = spectrum
        indices_current = peak_indices(y, prominence, wlen)
        indices.append(indices_current)
    return np.array(indices)


def spectra_peak_indices_adaptative(spectra, factor=1, wlen=10):
    indices = []
    min_spectra = spectra_min(spectra)
    size = min_spectra.shape[0]
    stddev = np.array([np.std(min_spectra[max(0,i-wlen) : min(i+wlen, size-1)]) for i in range(min_spectra.shape[0])])
    for spectrum in spectra:
        x, y = spectrum
        indices_current = peak_indices(y, stddev * factor, wlen)
        indices.append(indices_current)
    return np.array(indices)

def peak_indices(data, prominence, wlen):
    """
    Estimates and extracts significant peaks in the spectrum
    By using the prominence (height of the peak relative to the nearest
    higher peak)

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    prominence: int
        threshold on prominence


    Returns
    ----------
    np.ndarray
        Peak indices relative to spectrum
    """
    peak_indices, _ = signal.find_peaks(tuple(data),
                                         prominence=prominence,
                                         wlen=wlen,
                                         distance=1)
    return peak_indices


def peak_indices_cwt(data, widths):
    peak_indices, _ = signal.find_peaks_cwt(tuple(data),
                                            widths=widths,
                                            min_snr=3)
    return peak_indices


def peak_selection_shape(spectra):
    """
    Peak selection based on shape
    Uses ms_peak_picker module

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array

    Returns
    ----------
    list
        list of peaks for all spectra
    """
    spectra_peak_list = []
    for spectrum in spectra:
        x, y = spectrum
        peak_list = ms_peak_picker.pick_peaks(x, y, fit_type="quadratic", signal_to_noise_threshold=2)
        print([p.mz for p in peak_list.peaks.peaks])
        break
        spectra_peak_list = spectra_peak_list + list(peak_list.peaks.peaks)
    return spectra_peak_list

def peak_reference_indices(indices):
    """
    Extracts the peaks on the histogram of peaks
    for realignment purposes

    Parameters
    ----------
    indices: np.ndarray
        indices corresponding to peaks

    Returns
    ----------
    np.ndarray
        indices of peaks on the histogram

    """
    counts = np.bincount(indices)
    indices_second = peak_indices(counts, prominence=4, wlen=10)
    return indices_second

def normalization_tic(y):
    """
    TIC (total ion count) normalization
    Divides each intensity in a spectrum by the sum of all its intensities

    Parameters
    ----------
    y: np.ndarray
        spectrum

    Returns
    ----------
    np.ndarray
        normalized spectrum
    """
    sum_intensity = 0
    for intensity in y:
        sum_intensity += intensity
    return y / sum_intensity


def index_groups(indices, step=1):
    """
    Makes groups of indices
    For realignment and spatial selection

    Parameters
    ----------
    indices: list
        list of peak indices
    step: int
        threshold in indices to create groups

    Returns
    ----------
    list
        groups=list of list of peak indices
    """
    indices.sort()
    groups = []
    index = 0
    L = []
    while index < len(indices) - 1:
        value = indices[index]
        next = indices[index+1]
        L.append(value)
        if abs(value - next) > step:
            groups.append(L)
            if index == len(indices) - 2:
                groups.append([next])
            L = []
        elif index == len(indices) - 2:
            L.append(next)
            groups.append(L)
        index += 1
    return groups

def peak_reference_indices_group(group):
    """
    Extracts the reference peak in a group
    i.e. the most frequent in a group

    Parameters
    ----------
    group: list
        list of peak indices
    """
    return max(set(group), key=group.count)

def peak_reference_indices_groups(groups):
    """
    Extracts the reference peaks for several groups

    Parameters
    ----------
    groups: list
        groups=list of list of peak indices

    Returns
    ----------
    list
        list of reference peak indices
    """
    indices = []
    for group in groups:
        index = peak_reference_indices_group(group)
        indices.append(index)
    return indices

def peak_reference_indices_median(groups):
    indices = []
    for group in groups:
        group_copy = sorted(group.copy())
        index = int(np.median(group_copy))
        indices.append(index)
    return indices



def width_peak_indices(indices, full_indices):
    """
    Computes the width of a peak
    by checking neighbor indices

    Parameters
    ----------
    indices: np.ndarray
        peak indices on the histogram of peaks
    full_indices: np.ndarray
        peak indices on the spectra

    Returns
    ----------
    dict:
        peak indices to corresponding width
    """
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
    """
    Extracts the closest peak of index num

    Parameters
    ----------
    num: int
        index
    indices_to_width: dict
        dictionary mapping indices to width

    Returns
    ----------
    tuple
        mz and associated width

    """
    mz = min(indices_to_width.keys(), key=lambda k: abs(k-num))
    width = indices_to_width[mz]
    return mz, width


def realign_wrt_peaks(spectra, aligned_peaks, full_peaks, indices_to_width):
    realigned_spectra = []
    for i in range(spectra.shape[0]):
        spectrum = spectra[i]
        x, y = spectrum
        x_realigned = np.copy(x)
        y_realigned = np.copy(y)
        indices = full_peaks[i]
        for i in indices:
            mz, width = closest_peak(i, indices_to_width)
            if (i != mz and i >= mz - width and i <= mz + width):
                y_realigned[mz] = max(y[i], y_realigned[mz])
        x_realigned = x_realigned[aligned_peaks]
        y_realigned = y_realigned[aligned_peaks]
        realigned_spectra.append((x_realigned, y_realigned))
    return realigned_spectra

def realign(spectra, prominence=50, nb_occurrence=4, step=0.02):
    """
    Main function allowing to realign the spectra
    First extracts the peaks on all spectra,
    then extracts the reference peaks
    and maps each peak to its closest reference peak

    Parameters
    ----------
    spectra: np.ndarray
        spectra
    prominence: int
        threshold on prominence

    Returns
    ----------
    list
        realigned spectra

    """
    mz, I = spectra[0]
    min_diff = mz[1] - mz[0]
    step_index = math.ceil(step / min_diff)
    wlen = max(10, int(0.2 / min_diff))
    full_indices = spectra_peak_indices(spectra, prominence, wlen)
    flat_full_indices = np.hstack(full_indices)
    unique_indices = np.unique(flat_full_indices)
    groups = index_groups(flat_full_indices, step)
    groups = [group for group in groups if len(group) > nb_occurrence]
    aligned_indices = peak_reference_indices_groups(groups)
    indices_to_width = width_peak_indices(aligned_indices, full_indices)
    realigned_spectra = realign_wrt_peaks(spectra, aligned_indices, full_indices, indices_to_width)
    return np.array(realigned_spectra)


def realign_median(spectra, factor=1, nb_occurrence=4, step=0.02):
    mz, I = spectra[0]
    min_diff = mz[1] - mz[0]
    step_index = math.ceil(step / min_diff)
    wlen = max(10, int(1.0 / min_diff))
    full_indices = spectra_peak_indices_adaptative(spectra, factor, wlen)
    flat_full_indices = np.hstack(full_indices)
    unique_indices = np.unique(flat_full_indices)
    groups = index_groups(flat_full_indices, step)
    groups = [group for group in groups if len(group) > nb_occurrence]
    aligned_indices = peak_reference_indices_median(groups)
    indices_to_width = width_peak_indices(aligned_indices, full_indices)
    realigned_spectra = realign_wrt_peaks(spectra, aligned_indices, full_indices, indices_to_width)
    return np.array(realigned_spectra)


def neighbours(index, n, spectra):
    s = index
    e = index + n if index + n < spectra.shape[0] else spectra.shape[0]
    return spectra[s:e, ...]

def forward_derivatives(peaks):
    derivatives = []
    for i in range(len(peaks)-1):
        p = peaks[i]
        p_next = peaks[i+1]
        d = (p_next[1] - p[1]) / (p_next[0] - p[0])
        derivatives.append(d)
    return derivatives

def find_isotopic_pattern(neighbours, tolerance, nb_charges):
    pattern = [neighbours[0]]
    for j in range(1, neighbours.shape[0]):
        n = neighbours[j]
        d = n[0] - neighbours[0][0]
        eps = abs(d - round(d))

        previous = pattern[-1]
        d_previous = n[0] - previous[0]
        eps_previous = abs(d_previous - round(d_previous))
        if eps < tolerance and d_previous-eps_previous < nb_charges:
            pattern.append(n)
    return pattern

def peaks_max_intensity_isotopic_pattern(pattern):
    index_max = np.argmax(np.array(pattern), axis=0)[1]
    return [pattern[index_max]]


def peaks_derivative_isotopic_pattern(pattern):
    peaks = [pattern[0]]
    derivatives = forward_derivatives(pattern)
    for j in range(1, len(derivatives)):
        d = derivatives[j]
        d_previous = derivatives[j - 1]
        previous_peak = pattern[j]
        associated_peak = pattern[j+1]
        ratio = associated_peak[1]/previous_peak[1]
        if (d > 0 and d_previous <= 0) or ratio > 5:
            peaks.append(associated_peak)
    return peaks


def isotopes_from_pattern(pattern, peaks_in_pattern):
    isotopes = []
    for i in range(len(pattern)):
        close = False
        peak = pattern[i]
        for other_peak in peaks_in_pattern:
            close |= np.isclose(peak[0], other_peak[0])
        if not close:
            isotopes.append(peak)
    return np.array(isotopes)

def mz_second_isotope_most_abundant(average_distribution):
    masses = {"H": 1.0078, "C": 12, "N": 14.00307, "O": 15.99491, "S": 31.97207, "H2": 2.014, "C13": 13.00335, "N15":15.00011, "O17": 16.999913, "O18": 17.99916, "S33":32.97146, "S34": 33.9678}
    distribution = {"H": 0.99985, "C": 0.9889, "N": 0.9964, "O": 0.9976, "S": 0.95, "H2": 0.00015, "C13": 0.0111, "N15": 0.0036, "O17": 0.0004, "O18": 0.002, "S33": 0.0076, "S34": 0.0422}
    without_isotopes = 0
    with_isotopes = 0
    list_names = '\n'.join(list(masses.keys()))
    for k, v in average_distribution.items():
        if k in masses:
            pattern = re.compile(k + ".*")
            without_isotopes += v*masses[k]
            matches = pattern.findall(list_names)
            with_isotopes += sum([v*masses[m]*distribution[m] for m in matches])
    if with_isotopes == without_isotopes:
        return 0
    n = 1.0 / (with_isotopes - without_isotopes)
    mz = n * without_isotopes
    return mz

def peak_to_index(peak, pattern):
    for i in range(len(pattern)):
        p = pattern[i]
        if np.isclose(peak[0], p[0]):
            return i
    return 0

def deisotoping_simple(spectra, tolerance=0.1, nb_neighbours=8, nb_charges=5, average_distribution={}):
    deisotoped_spectra = []
    deisotoped_indices = []
    ignore_indices = []
    mz_second_isotope = mz_second_isotope_most_abundant(average_distribution)
    mzs = spectra[0][0]
    peaks = spectra_max(spectra)
    peaks = np.array([mzs, peaks])
    # peaks = peaks[...,(peaks[0] > 860) & (peaks[0] < 880)]
    x = peaks.shape[-1]
    for i in range(x):
        if np.any([np.isclose(ignore_indices[j][0], peaks[0, i]) for j in range(len(ignore_indices))]):
            continue
        peak = peaks[..., i]
        N = neighbours(i, nb_neighbours, peaks.T)
        pattern = find_isotopic_pattern(N, tolerance, nb_charges)
        if peak[0] < mz_second_isotope:
            peaks_pattern = peaks_max_intensity_isotopic_pattern(pattern)
        else:
            peaks_pattern = peaks_derivative_isotopic_pattern(pattern)
        isotopes = isotopes_from_pattern(pattern, peaks_pattern)
        ignore_indices.extend(isotopes)
        indices = [peak_to_index(peak, pattern) for peak in peaks_pattern]
        deisotoped_indices.extend([i+j for j in indices if i+j not in deisotoped_indices])
        # print("Neighbours=", N)
        # print("Pattern=", pattern)
        # print("Other peaks=", peaks_pattern)
        # print("Isotopes=", isotopes)

    deisotoped_indices = np.array(deisotoped_indices)
    for spectrum in spectra:
        mzs, intensities = spectrum
        new_mzs = mzs[deisotoped_indices]
        new_intensities = intensities[deisotoped_indices]
        deisotoped_spectra.append((new_mzs, new_intensities))
    return np.array(deisotoped_spectra)


def deisotoping(spectra):
    """
    Removes isotopes from a collection of spectra
    Computes deisotoping for each spectrum and
    keeps array of mzs of highest length

    Based on pyopenms module
    Looks at height of neighbouring peaks
    not accurate for patterns where peak with max intensity
    does not have the lowest m/z in the pattern

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array

    Returns
    ----------
    np.ndarray
        Deisotoped spectra
    """
    max_length = 0
    mzs = spectra[0, 0]
    x = spectra.shape[0]
    for i in range(x):
        s = MSSpectrum()
        s.set_peaks(spectra[i, ...].tolist())
        Deisotoper.deisotopeAndSingleCharge(s, fragment_tolerance=0.1, fragment_unit_ppm=False, min_charge=1, max_charge=8, keep_only_deisotoped=True, min_isopeaks=2, max_isopeaks=3, make_single_charged=True, annotate_charge=True)
        if s.size() > max_length:
            max_length = s.size()
            mzs = np.array(s.get_peaks())[0, ...]
    condition = np.isin(spectra[0,0], mzs)
    deisotoped_spectra = spectra[..., condition]
    return deisotoped_spectra


def peak_to_ms_peak(peak, index):
    """
    Converts a peak from [mz,I] to
    ms_peak_picker.FittedPeak

    Parameters
    ----------
    peak: list
        [mz,I]
    index: int
        index in spectrum

    Returns
    ----------
    ms_peak_picker.FittedPeak
        adjusted peak
    """
    peak = ms_peak_picker.FittedPeak(mz=peak[0],
                                     intensity=peak[1],
                                     signal_to_noise=max(peak[1]*0.1, 50),
                                     peak_count=index,
                                     index=index,
                                     full_width_at_half_max=0.1,
                                     area=peak[1]*0.1,
                                     left_width=0.05,
                                     right_width=0.05)
    return peak





def deisotoping_deconvolution(spectra):
    """
    Deisotoping by deconvolution
    Computes deisotoping on max spectrum

    Based on ms_deisotope module
    """
    max_length = 0
    x, y = spectra[0]
    y = spectra_max(spectra)
    peak_list = [peak_to_ms_peak([x[i], y[i]], i) for i in range(len(x))]
    deconvoluted_peaks, _ = ms_deisotope.deconvolute_peaks(peak_list,
                                                           averagine=ms_deisotope.glycan,
                                                           scorer=ms_deisotope.MSDeconVFitter(10.0),
                                                           charge_range=(1,3),
                                                           truncate_after=0.8)
    mzs = [elem.mz for elem in list(deconvoluted_peaks.peaks)]
    condition = np.isin(spectra[0,0], mzs)
    deisotoped_spectra = spectra[..., condition]
    return deisotoped_spectra
