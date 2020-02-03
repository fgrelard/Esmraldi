import ms_peak_picker
import ms_deisotope
import math
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

def spectra_peak_indices(spectra, prominence=50):
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
        indices_current = peak_indices(y, prominence)
        indices = indices + indices_current.tolist()
    return indices

def peak_indices(data, prominence=50):
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
                                         wlen=10,
                                         distance=1)
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
        peak_list = ms_peak_picker.pick_peaks(x, y, fit_type="quadratic", signal_to_noise_threshold=10)
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
    indices_second = peak_indices(counts, 4)
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


def realign(spectra, prominence=50, nb_occurrence=4, step=2):
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
    full_indices = spectra_peak_indices(spectra, prominence)
    unique_indices = np.unique(full_indices)
    groups = index_groups(full_indices, step)
    groups = [group for group in groups if len(group) > nb_occurrence]
    aligned_indices = peak_reference_indices_groups(groups)
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
    pattern = []
    for j in range(neighbours.shape[0]):
        n = neighbours[j]
        d = n[0] - neighbours[0][0]
        eps = abs(d - round(d))
        if eps < tolerance and d-eps < nb_charges:
            pattern.append(n)
    return pattern

def peaks_isotopic_pattern(pattern):
    peaks = []
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


def isotope_indices(pattern, peaks_in_pattern):
    indices_isotopes = []
    for i in range(len(pattern)):
        close = False
        peak = pattern[i]
        for other_peak in peaks_in_pattern:
            close |= np.isclose(peak[0], other_peak[0])
        if not close:
            indices_isotopes.append(i)
    return indices_isotopes

def isotopes_from_pattern(pattern, peaks_in_pattern):
    isotopes = []
    for i in range(len(pattern)):
        close = False
        peak = pattern[i]
        for other_peak in peaks_in_pattern:
            close |= np.isclose(peak[0], other_peak[0])
        if not close and i > 0:
            isotopes.append(peak)
    return np.array(isotopes)

def deisotoping_simple(spectra, tolerance=0.1, nb_neighbours=8, nb_charges=5):
    deisotoped_spectra = []
    deisotoped_indices = []
    ignore_indices = []
    mzs = spectra[0][0]
    peaks = spectra_max(spectra)
    peaks = np.array([mzs, peaks])
    peaks = peaks[...,(peaks[0] > 604) & (peaks[0] < 622)]
    x = peaks.shape[-1]
    # print(peaks)
    for i in range(x):
        if np.any([np.isclose(ignore_indices[j][0], peaks[0, i]) for j in range(len(ignore_indices))]):
            continue
        peak = peaks[..., i]
        N = neighbours(i, nb_neighbours, peaks.T)
        # print(N)
        pattern = find_isotopic_pattern(N, tolerance, nb_charges)
        # print("Pattern=", pattern)
        other_peaks = peaks_isotopic_pattern(pattern)
        # print("Other peaks=", other_peaks)
        #isotope_ind = isotope_indices(pattern, other_peaks)
        #print("Isotope ind=", isotope_ind)
        isotopes = isotopes_from_pattern(pattern, other_peaks)
        # print("Isotopes=", isotopes)
        ignore_indices.extend(isotopes)
        # ignore_indices.extend([i+j for j in isotope_ind if j != 0])
        deisotoped_indices.append(i)
    print("End, ignore ind=", ignore_indices)
    deisotoped_indices = np.array(deisotoped_indices)
    print("Deisotoped ind=", deisotoped_indices)
    for spectrum in spectra:
        mzs, intensities = spectrum
        new_mzs = mzs[deisotoped_indices]
        new_intensities = intensities[deisotoped_indices]
        deisotoped_spectra.append((new_mzs, new_intensities))
    print("Start len=", x, "Final len=", len(deisotoped_spectra[0][0]))
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
