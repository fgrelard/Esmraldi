from __future__ import absolute_import

import unittest
import numpy as np
import pandas as pd
import numpy.testing as nptest
import src.spectraprocessing as sp
import matplotlib.pyplot as plt
import src.imzmlio as io
from ms_deisotope import plot
from ms_deisotope import Averagine



def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class TestSpectraProcessing(unittest.TestCase):
    def setUp(self):
        self.spectra = []
        x_values = np.linspace(0, 10, 40)
        for i in range(10):
            y_values = gaussian(x_values, 4.5, 0.375)
            self.spectra.append((x_values, y_values))
        for i in range(3):
            y_values = gaussian(x_values, 5-i/2, 0.5-i/8)
            self.spectra.append((x_values, y_values))
        self.spectra = np.array(self.spectra)
        data = self.spectra.copy()
        data = data.reshape((-1, self.spectra.shape[-1]))
        h1 = np.array([i//2 for i in range(data.shape[0])])
        h2 = np.array(["mz", "intensity"]*(data.shape[0]//2))
        df = pd.DataFrame(data=data.T, columns=pd.MultiIndex.from_tuples(zip(h1,h2)))

    def test_spectra_sum(self):
        sum = sp.spectra_sum(self.spectra)
        print(sum)

    def test_peak_indices(self):
        x_values = np.linspace(0, 10, 40)
        y_values = gaussian(x_values, 5, 0.5)
        indices = sp.peak_indices(y_values, 0.5)
        nptest.assert_array_equal(indices, [20])

    def test_spectra_peak_indices(self):
        indices = sp.spectra_peak_indices(self.spectra, 0.5)
        unique = np.unique(indices)
        nptest.assert_array_equal(unique, [16, 18, 20])

    def test_peak_reference_indices(self):
        indices = sp.spectra_peak_indices(self.spectra, 0.5)
        indices += [18 for i in range(10)]
        reference_indices = sp.peak_reference_indices(indices)
        nptest.assert_array_equal(reference_indices, [18])

    def test_index_groups(self):
        indices = sp.spectra_peak_indices(self.spectra, 0.5)
        groups = sp.index_groups(indices, 1)
        print(groups)

    def test_peak_reference_indices_group(self):
        indices = sp.spectra_peak_indices(self.spectra, 0.5)
        groups = sp.index_groups(indices, 1)
        print(groups)
        reference_indices = sp.peak_reference_indices_group(groups[0])
        print(reference_indices)

    def test_width_peak_indices(self):
        indices = sp.spectra_peak_indices(self.spectra, 0.5)
        indices += [18 for i in range(10)]

        reference_indices = sp.peak_reference_indices(indices)
        indices_to_width = sp.width_peak_indices(reference_indices, indices)
        nptest.assert_equal(indices_to_width[18], 3.0)

    def test_realign(self):
        s = sp.realign(self.spectra, 0.5)
        x = np.unique([i[0][0] for i in s])
        sum_before = sp.spectra_sum(self.spectra)
        sum_after = sp.spectra_sum(s)
        nptest.assert_equal(x[0], self.spectra[0][0][18])

    def test_neighbours(self):
        L = [[10.2, 10.], [11, 23.], [12.23, 11.25],
             [13.23, 8.75], [14.3, 6.2], [15.2, 18.96],
             [16.2, 38.2], [17.21, 20.1], [18.19, 7.5]]
        N = sp.neighbours(0, 5, np.array(L))
        nptest.assert_equal(N, L[:5])

    def test_find_isotopic_pattern(self):
        L = [[10.2, 10.], [11, 23.], [12.23, 11.25],
             [13.23, 8.75], [14.3, 6.2], [15.2, 18.96],
             [16.2, 38.2], [17.21, 20.1], [18.19, 7.5]]
        pattern = sp.find_isotopic_pattern(np.array(L), 0.1, 5)
        print(pattern)
        nptest.assert_equal(pattern, [L[0], L[2], L[3], L[5]])

    def test_peaks_isotopic_pattern(self):
        L = [[10.2, 10.], [11, 23.], [12.23, 11.25],
             [13.23, 8.75], [14.3, 6.2], [15.2, 18.96],
             [16.2, 38.2], [17.21, 20.1], [18.19, 7.5]]
        pattern = sp.find_isotopic_pattern(L, 0.1, 5)
        d = sp.forward_derivatives(pattern)
        peaks = sp.peaks_isotopic_pattern(pattern)
        nptest.assert_equal(peaks[0], L[5])

    def test_forward_derivatives(self):
        L = [[10, 10.], [11, 23.], [12.23, 11.25],
             [13.23, 8.75], [14.3, 6.2], [15.2, 18.96],
             [16.2, 38.2], [17.21, 20.1], [18.19, 7.5]]
        d = sp.forward_derivatives(L)
        nptest.assert_equal(d, [13.0, -9.552845528455281, -2.5, -2.383177570093457, 14.177777777777802, 19.240000000000002, -17.920792079207896, -12.857142857142852])

    def test_isotope_indices(self):
        pattern = [[10.2, 10.0], [12.23, 11.25], [13.23, 8.75], [15.2, 18.96]]
        peaks = sp.peaks_isotopic_pattern(pattern)
        isotopes = sp.isotope_indices(pattern, peaks)
        nptest.assert_equal(isotopes, [0, 1, 2])

    def test_mz_second_isotope_most_abundant(self):
        distrib = {'C': 7.0, 'H': 11.8333, 'N': 0.5, 'O': 5.16666}
        distrib = {}
        mz = sp.mz_second_isotope_most_abundant(distrib)
        nptest.assert_almost_equal(mz, 1778.6831809)

    def test_deisotoping(self):
        spectra = np.load("data/peaksel_650DJ_35.npy")
        print(spectra.shape)
        print(spectra[0, 0, ...])
        deisotoped = sp.deisotoping(spectra)
        print(deisotoped[0, 0, ...])


    def test_deisotoping_deconvolution(spectra):
        spectra = np.load("data/peaksel_650DJ_35.npy")
        deisotoped = sp.deisotoping_deconvolution(spectra)




if __name__ == "__main__":
    unittest.main()
