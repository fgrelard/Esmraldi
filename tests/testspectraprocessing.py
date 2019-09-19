from __future__ import absolute_import

import unittest
import numpy as np
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

    def test_deisotoping(self):
        spectra = np.load("data/peaksel_650DJ_35.npy")
        print(spectra.shape)
        print(spectra[0, 0, ...])
        deisotoped = sp.deisotoping(spectra)
        print(deisotoped[0, 0, ...])


    def test_deisotoping_deconvolution(spectra):
        peptide_averagine = Averagine({"C": 4.9384, "H": 7.7583, "N": 1.3577, "O": 1.4773, "S": 0.0417})

        spectra = np.load("data/peaksel_650DJ_35.npy")
        deisotoped = sp.deisotoping_deconvolution(spectra)




if __name__ == "__main__":
    unittest.main()
