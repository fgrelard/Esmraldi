from __future__ import absolute_import

import unittest
import numpy as np
import numpy.testing as nptest
import src.imzmlio as io
import matplotlib.pyplot as plt


class TestImzMLIO(unittest.TestCase):
    def setUp(self):
        self.data = np.load("data/peaksel.npy")

    def test_get_spectra_from_images(self):
        self.imzml = io.open_imzml("/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel.imzML")
        coords = self.imzml.coordinates
        i, newcoords = io.get_spectra_from_images(self.data)
        self.assertEqual(coords, newcoords)

    def test_get_all_array_images(self):
        self.imzml = io.open_imzml("/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel_small.imzML")
        image = io.get_all_array_images(self.imzml)
        image2 = io.to_image_array(self.imzml)
        print(image2.shape)
        print(image.shape)


if __name__ == "__main__":
    unittest.main()
