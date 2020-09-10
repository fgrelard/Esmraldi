"""
Module for the preprocessing of spectra
specifically designed for MALDI images

  - Peak picking
  - Local realignment procedures
  - Deisotoping
"""

import math
import re
import sys
import scipy.signal as signal
import numpy as np
import bisect
from functools import reduce

def spectra_sum(spectra):
    """
    Computes the spectrum from the sum of all spectra.

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
    Computes the average spectrum.

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
    Computes the maximum intensity for each abscissa.

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
        spectra_max = np.fmax(spectra_max, y)
    return spectra_max

def spectra_min(spectra):
    """
    Computes the minimum intensity for each abscissa.

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
        spectra_min = np.fmin(spectra_min, y)
    return spectra_min

def spectra_peak_indices(spectra, prominence=50, wlen=10):
    """
    Estimates and extracts significant peaks in the spectra
    by using the prominence (height of the peak
    relative to the nearest higher peak).

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
    """
    Estimates and extracts significant peaks in the spectra
    by using the local prominence
    (height of the peak relative to the background noise).

    Background noise is estimated as the
    standard deviation of the
    signal over a window of size wlen.

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    factor: float
        prominence factor
    wlen: int
        size of the window

    Returns
    ----------
    np.ndarray
        Peak indices relative to spectra
    """
    indices = []
    min_spectra = spectra_min(spectra)
    size = min_spectra.shape[0]
    stddev = np.array([np.std(min_spectra[max(0,i-wlen) : min(i+wlen, size-1)]) for i in range(min_spectra.shape[0])])
    for spectrum in spectra:
        x, y = spectrum
        indices_current = peak_indices(y, stddev * factor, wlen)
        indices.append(indices_current)
    return np.array(indices)

def spectra_peak_mzs_adaptative(spectra, factor=1, wlen=10):
    """
    Estimates and extracts significant peaks in the spectra
    by using the local prominence
    (height of the peak relative to the background noise).

    Background noise is estimated as the
    standard deviation of the
    signal over a window of size wlen.

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    factor: float
        prominence factor
    wlen: int
        size of the window

    Returns
    ----------
    np.ndarray
        Peaks m/z
    """
    mzs = []
    min_spectra = spectra_min(spectra)
    size = min_spectra.shape[0]
    stddev = np.array([np.nanstd(min_spectra[max(0,i-wlen) : min(i+wlen, size-1)]) for i in range(min_spectra.shape[0])])
    index = 0
    for spectrum in spectra:
        x, y = spectrum
        indices_current = peak_indices(y, stddev * factor, wlen)
        mzs_current = x[indices_current]
        mzs.append(mzs_current)
        index += 1
    return np.array(mzs, dtype=object)

def spectra_peak_mzs_adaptative_noiselevel(spectra, factor=1, noise_level=1, wlen=10):
    """
    Estimates and extracts significant peaks in the spectra
    with specified noise level(s),
    by using the local prominence
    (height of the peak relative to
    the background noise)

    Background noise is estimated as the
    standard deviation of the
    signal over a window of size wlen.

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    factor: float
        prominence factor
    noise_level: float or list
        noise level
    wlen: int
        size of the window

    Returns
    ----------
    np.ndarray
        Peaks m/z
    """
    mzs = []
    index = 0
    for spectrum in spectra:
        x, y = spectrum
        indices_current = peak_indices(y, noise_level * factor, wlen)
        mzs_current = x[indices_current]
        mzs.append(mzs_current)
        index += 1
    return np.array(mzs, dtype=object)


def peak_indices(data, prominence, wlen):
    """
    Estimates and extracts significant peaks
    in the spectrum,
    by using the prominence (height of the peak
    relative to the nearest
    higher peak).

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
    intensities = np.ma.masked_invalid(data)
    peak_indices, _ = signal.find_peaks(tuple(data),
                                         prominence=prominence,
                                         wlen=wlen,
                                         distance=1)
    return peak_indices

def peak_indices_cwt(data, factor, widths):
    """
    Peak indices using continuous wavelet
    transform.

    Parameters
    ----------
    data: np.ndarray
        Spectra as [mz*I] array
    factor: float
        Threshold SNR
    widths: list
        scales

    Returns
    ----------
    np.ndarray
        Peak indices relative to spectrum

    """
    peak_indices = signal.find_peaks_cwt(tuple(data),
                                            widths=widths,
                                            min_snr=factor)
    return peak_indices

def spectra_peak_mzs_cwt(spectra, factor, widths):
    """
    Peak detection using the Continuous Wavelet Transform.

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    factor: float
        CWT threshold
    widths: list
        Wavelet widths

    Returns
    ----------
    np.ndarray
        Detected peak m/z ratios

    """
    mzs = []
    for spectrum in spectra:
        x, y = spectrum
        indices_current = peak_indices_cwt(y, factor, widths)
        if len(indices_current) > 0:
            mzs.append(x[indices_current])
        else:
            mzs.append([])
    return np.array(mzs, dtype=object)

def same_mz_axis(spectra, tol=0):
    """
    Generates spectra with common m/z values.

    Missing intensity values are added as np.nan.

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    tol: float
        Tolerance to consider when two species are the same

    Returns
    ----------
    np.ndarray
        Spectra as [mz*I] array
    """
    masses = spectra[..., 0]
    o = np.array(masses[0])
    if tol == 0:
        masses_union = np.array(list(set().union(*masses)))
    else:
        i = 0
        for mass_list in masses:
            m = np.array(mass_list)
            c = m[(np.abs(o[:,None] - m) >= tol).all(0)]
            o = np.concatenate((o, c), axis=None)
        masses_union = np.array(o)
    new_matrix = np.zeros(shape=(spectra.shape[0], spectra.shape[1], masses_union.shape[0]))
    new_matrix.fill(np.nan)
    index = 0
    for x, y in spectra:
        x_array = np.array(x)
        y_array = np.array(y)
        indices = np.nonzero(np.isin(masses_union, x_array))[0]
        new_matrix[index, 0] = masses_union
        np.put(new_matrix[index, 1], indices, y_array)
        index += 1
    return new_matrix



def normalization_tic(spectra):
    """
    TIC (total ion count) normalization.

    Divides each intensity in a spectrum by
    the sum of all its intensities.

    Parameters
    ----------
    spectra: np.ndarray
        spectra as [mz*I] array

    Returns
    ----------
    np.ndarray
        normalized spectrum
    """
    spectra_normalized = spectra.copy()
    for i, (x, y) in enumerate(spectra):
        spectra_sum = np.sum(y)
        median = np.median(y)
        new_y = y / spectra_sum * median
        spectra_normalized[i, 1, :] = new_y
    return spectra_normalized



def index_groups(indices, step=1):
    """
    Makes groups of indices.

    For realignment and spatial selection.

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
    Extracts the reference peak in a group,
    i.e. the most frequent in a group.

    Parameters
    ----------
    group: list
        list of peak indices
    """
    unique, counts = np.unique(group, return_counts=True)
    most_frequent = unique[counts == counts.max()][0]
    return most_frequent

def peak_reference_indices_groups(groups):
    """
    Extracts the reference peaks for several groups.

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
    """
    Extracts the reference peak in a group
    as the median peak.

    Parameters
    ----------
    groups: list
        list of peak indices

    Returns
    ----------
    list
        list of reference peak indices

    """
    indices = []
    for group in groups:
        group_copy = sorted(group.copy())
        index = np.median(group_copy)
        indices.append(index)
    return indices

def width_peak_mzs(aligned_mzs, groups, default=0.001):
    """
    Computes the width of a peak
    by computing the difference in m/z
    between the upper and lower bounds in the group.

    Parameters
    ----------
    aligned_mzs: list
        list of m/z
    groups: list
        list of peak indices
    default: float
        default width of the peak

    Returns
    ----------
    dict
        maps m/z to widths

    """
    indices_to_width = {}
    aligned_array = np.array(aligned_mzs)
    for group in groups:
        diff = max(group) - min(group) + default
        group_array = np.array(group)
        index = aligned_array[np.abs(np.mean(group_array)- aligned_array[:, None]).argmin()]
        indices_to_width[index] = diff
    return indices_to_width


def width_peak_indices(indices, full_indices):
    """
    Computes the width of a peak
    by checking neighboring indices.

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
    Extracts the closest peak of index num.

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


def realign_wrt_peaks_mzs(spectra, aligned_mzs, full_mzs, indices_to_width):
    """
    Realign spectra to reference peaks.

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    aligned_mzs: list
        reference mz peaklist
    full_mzs: np.ndarray
        complete mz peaklist over all spectra
    indices_to_width: dict
        mz to width

    Returns
    ----------
    list
        realigned spectra
    """
    realigned_spectra = []
    for i in range(spectra.shape[0]):
        spectrum = spectra[i]
        x, y = spectrum
        matching_indices = [bisect.bisect_left(x, ii) for ii in aligned_mzs]
        matching_indices = [elem if elem < len(x) else len(x) - 1 for elem in matching_indices]
        y_realigned = np.array(y)[matching_indices]
        indices = full_mzs[i]
        for i in indices:
            mz, width = closest_peak(i, indices_to_width)
            mz_index = np.abs(aligned_mzs - mz).argmin()
            i_index = np.abs(x - i).argmin()
            if (i != mz and i >= mz - width and i <= mz + width):
                y_realigned[mz_index] = max(y[i_index], y_realigned[mz_index])
        realigned_spectra.append((aligned_mzs, y_realigned))
    return realigned_spectra


def realign_wrt_peaks(spectra, aligned_peaks, full_peaks, indices_to_width):
    """
    Realign spectra to reference peaks
    from indices.

    Parameters
    ----------
    spectra: np.ndarray
        Spectra as [mz*I] array
    aligned_mzs: list
        reference mz peaklist
    full_mzs: np.ndarray
        complete mz peaklist over all spectra
    indices_to_width: dict
        indices to width

    Returns
    ----------
    list
        realigned spectra indices
    """
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

def realign_indices(spectra, indices, reference="frequence", nb_occurrence=4, step=0.02):
    """
    Alignment function.

    First extracts the peaks on all spectra,
    then extracts the reference peaks
    and maps each peak to its closest reference peak.

    Parameters
    ----------
    spectra: np.ndarray
        spectra
    indices: np.ndarray
        indices of peaks relative to spectra

    Returns
    ----------
    list
        realigned spectra

    """
    mz, I = spectra[0]
    min_diff = mz[1] - mz[0]
    step_index = math.ceil(step / min_diff)
    flat_full_indices = np.hstack(indices)
    unique_indices = np.unique(flat_full_indices)
    groups = index_groups(flat_full_indices, step_index)
    groups = [group for group in groups if len(group) > nb_occurrence]
    if reference == "frequence":
        aligned_indices = peak_reference_indices_groups(groups)
    else:
        aligned_indices = peak_reference_indices_median(groups)
    indices_to_width = width_peak_indices(aligned_indices, indices)
    realigned_spectra = realign_wrt_peaks(spectra, aligned_indices, indices, indices_to_width)
    return np.array(realigned_spectra)


def realign_mzs(spectra, mzs, reference="frequence", nb_occurrence=4, step=0.02):
    """
    Alignment function.

    First extracts the peaks on all spectra based on local prominence,
    then extracts the reference peaks
    and maps each peak to its closest reference peak.

    Parameters
    ----------
    spectra: np.ndarray
        spectra
    mzs: np.ndarray
        peaklist of mzs ratio

    Returns
    ----------
    list
        realigned spectra

    """
    mz, I = spectra[0]
    flat_full_mzs = np.hstack(mzs)
    groups = index_groups(flat_full_mzs, step)
    groups = [group for group in groups if len(group) > nb_occurrence]
    print(len(groups))
    if reference == "frequence":
        aligned_mzs = peak_reference_indices_groups(groups)
    else:
        aligned_mzs = peak_reference_indices_median(groups)
    indices_to_width = width_peak_mzs(aligned_mzs, groups)
    realigned_spectra = realign_wrt_peaks_mzs(spectra, aligned_mzs, mzs, indices_to_width)
    return np.array(realigned_spectra)



def neighbours(index, n, spectra):
    """
    Right-sided neighbours of a point in a spectrum.

    Parameters
    ----------
    index: int
        index to search neighbours from
    n: int
        number of neighbours
    spectra: np.ndarray
        spectrum

    Returns
    ----------
    np.ndarray
        neighbours

    """
    s = index
    e = index + n if index + n < spectra.shape[0] else spectra.shape[0]
    return spectra[s:e, ...]

def forward_derivatives(peaks):
    """
    Forward derivatives from peak value distribution.

    Parameters
    ----------
    peaks: list
        peaklist

    Returns
    ----------
    list
        derivatives
    """
    derivatives = []
    for i in range(len(peaks)-1):
        p = peaks[i]
        p_next = peaks[i+1]
        d = (p_next[1] - p[1]) / (p_next[0] - p[0])
        derivatives.append(d)
    return derivatives

def find_isotopic_pattern(neighbours, tolerance, nb_charges):
    """
    Extracts isotopic pattern based on mz similarity
    and max number of charges.

    Parameters
    ----------
    neighbours: np.ndarray
        neighbouring mz in spectra
    tolerance: float
        acceptable mz delta for a peak to be considered isotopic
    nb_charges: int
        maximum number of charges


    Returns
    ----------
    list
        pattern peaklist

    """
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
    """
    Finds the peak with maximum intensity in
    the isotopic pattern.

    Parameters
    ----------
    pattern: list
        pattern peaklist

    Returns
    ----------
    list
        peak with maximum intensity
    """
    index_max = np.argmax(np.array(pattern), axis=0)[1]
    return [pattern[index_max]]


def peaks_derivative_isotopic_pattern(pattern):
    """
    Finds the peak where the sign of the derivative changes
    from negative to positive
    from the pattern intensities.

    Parameters
    ----------
    pattern: list
        pattern peaklist

    Returns
    ----------
    list
        peaks where the derivative sign changes
    """
    peaks = [pattern[0]]
    derivatives = forward_derivatives(pattern)
    for j in range(1, len(derivatives)):
        d = derivatives[j]
        d_previous = derivatives[j - 1]
        previous_peak = pattern[j]
        associated_peak = pattern[j+1]
        if previous_peak[1] <= 0:
            continue

        ratio = associated_peak[1]/previous_peak[1]
        if (d > 0 and d_previous <= 0) or ratio > 5:
            peaks.append(associated_peak)
    return peaks


def isotopes_from_pattern(pattern, peaks_in_pattern):
    """
    Find isotopes from a pattern.

    Parameters
    ----------
    pattern: list
         pattern peaklist
    peaks_in_pattern: list
         peaks that correspond to other species in the pattern

    Returns
    ----------
    np.ndarray
        isotopes in the pattern
    """
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
    """
    Determines where the second isotope becomes the most abundant
    from a given distribution.

    Parameters
    ----------
    average_distribution: dict
        maps atom mass to its average abundance

    Returns
    ----------
    float
        mass at which the second isotope is the most abundant

    """
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
    """
    Gets the index of a peak in a pattern

    Parameters
    ----------
    peak: float
        mass ratio
    pattern: list
        pattern peaklist

    Returns
    ----------
    int
        index of peak

    """
    for i in range(len(pattern)):
        p = pattern[i]
        if np.isclose(peak[0], p[0]):
            return i
    return 0

def deisotoping_simple(spectra, tolerance=0.1, nb_neighbours=8, nb_charges=5, average_distribution={}):
    """
    Simple deisotoping depending on the mass of the
    secondmost abundant isotope:

      - Before this mass: uses the peak with max intensity
        as reference

      - After this mass: use the peak where the sign of the
        derivative changes

    Parameters
    ----------
    spectra: np.ndarray
        peaklist
    tolerance: float
        acceptable mz delta
    nb_neighbours: int
        size of patterns
    nb_charges: int
        maximum number of charges in isotopic pattern
    average_distribution: dict
        maps atom mass to its average abundance

    Returns
    ----------
    np.ndarray
        deisotoped spectra

    """
    deisotoped_spectra = []
    deisotoped_indices = []
    ignore_indices = []
    mz_second_isotope = mz_second_isotope_most_abundant(average_distribution)
    mzs = spectra[0][0]
    peaks = spectra_max(spectra)
    peaks = np.array([mzs, peaks])
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
    deisotoped_indices = np.array(deisotoped_indices)
    for spectrum in spectra:
        mzs, intensities = spectrum
        new_mzs = mzs[deisotoped_indices]
        new_intensities = intensities[deisotoped_indices]
        deisotoped_spectra.append((new_mzs, new_intensities))
    return np.array(deisotoped_spectra)
