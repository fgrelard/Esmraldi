import numpy as np
import scipy.signal as signal
import esmraldi.spectraprocessing as sp

class PeakDetectionMeanSpectrum:
    def __init__(self, mzs, mean_spectrum, factor_prominence, step_ppm):
        self.mzs = mzs
        self.mean_spectrum = mean_spectrum
        self.factor_prominence = factor_prominence
        self.step_ppm = step_ppm

    def widths_peak_mass_resolution(self, step):
        """
        Finds the widths of peaks (in number of samples)
        for each mz, using the mass resolution (step).
        The width of a peak in mz is expressed as step*mz.
        It is converted to a number of samples by computing
        the minimum number of points required to obtain
        such a width.
        """
        widths = np.zeros_like(self.mzs, dtype=int)
        diffs = np.diff(self.mzs)
        min_range = np.amin(diffs)
        median_range = np.median(diffs)
        for i, mz in enumerate(self.mzs):
            tol = step * mz
            wlen = int(tol / min_range)
            end = min(len(self.mzs), i+wlen)
            current_diffs = np.cumsum(diffs[i:end])
            ind = np.where(current_diffs < tol)[0]
            if ind.size == 0:
                widths[i] = int(tol / median_range)
            else:
                widths[i] = ind[-1]
        return widths

    def find_peak_indices(self, widths):
        peak_indices, _ = signal.find_peaks(tuple(self.mean_spectrum),
                                            prominence=self.factor_prominence,
                                            width=widths,
                                            rel_height=1)
        return peak_indices

    def extract_peaks(self):
        step = self.step_ppm / 1e6
        widths = self.widths_peak_mass_resolution(step)
        size = self.mean_spectrum.shape[0]
        median_signal = np.median(self.mean_spectrum)
        threshold_prominence = median_signal * self.factor_prominence
        self.factor_prominence = threshold_prominence
        peak_indices = self.find_peak_indices(widths=(widths,None))
        groups = sp.index_groups_start_end(self.mzs[peak_indices], self.step_ppm//2, True)
        filtered_indices = []
        cumlen = 0
        for g in groups:
            current_index = cumlen + len(g)//2
            filtered_indices.append(peak_indices[current_index])
            cumlen += len(g)
        return np.array(filtered_indices)

    def not_indices(self, indices, length):
        """
        Compute the complementary of
        the indices in a range of size "length"
        """
        mask = np.ones(length, dtype=bool)
        mask[indices] = False
        full_indices = np.arange(length, dtype=int)
        return full_indices[mask]


    def fill_zeros_with_last(self, arr):
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        return arr[prev]


    def align(self, reference_peaks, keep_mzs=False):
        peaks, peak_intensities = [], []
        indices_peaks_found = np.array([], dtype=int)
        diffs = np.zeros_like(mzs)
        for peak in reference_peaks:
            tolerance = step_ppm / 1e6 * peak
            begin = peak-tolerance
            end = peak+tolerance
            indices = np.where((self.mzs > begin) & (self.mzs < end))[0]
            diffs[indices] = peak - self.mzs[indices]
            intensity = np.sum(self.mean_spectrum[indices])
            peaks += [peak]
            peak_intensities += [intensity]
            indices_peaks_found = np.concatenate((indices_peaks_found, indices))
        if keep_mzs:
            diffs = self.fill_zeros_with_last(diffs)
            keep_indices = self.not_indices(indices_peaks_found, len(mzs))
            shift_mzs = self.mzs[keep_indices] + diffs[keep_indices]
            peaks = np.concatenate((peaks, shift_mzs))
            peak_intensities = np.concatenate((peak_intensities, intensities[keep_indices]))
        else:
            peaks = np.array(peaks)
            peak_intensities = np.array(peak_intensities)
        return peaks, peak_intensities
