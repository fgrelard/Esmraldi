from __future__ import absolute_import

import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import src.speciesrule as sr
from src.theoreticalspectrum import TheoreticalSpectrum
import src.spectrainterpretation as si
import random

class TestSpectraInterpretation(unittest.TestCase):
    def setUp(self):
        ax = sr.SpeciesRule("AX", "MS", mz=132.06, begin=569.26, end=2500, naming_fn=lambda i: "AX"+str(i+3))
        matrix = sr.SpeciesRule("Matrix", "M", mz=551, count=1)
        ac = sr.SpeciesRule("Ac", "A", mz=42.02, count=3, naming_fn=lambda i: str(i) + "Ac")
        self.ts = TheoreticalSpectrum([ax, matrix], [ac])
        self.ts.add_adducts_to_molecules_regexp("AX.*", ".*")
        random.seed(a=1)
        self.observed = [random.randrange(569, 2000) for i in range(200)]
        print(self.observed)

    def test_closest_peak(self):
        closest_peak_1 = si.closest_peak(569, self.ts.spectrum, 0.1)
        closest_peak_2 = si.closest_peak(569.2, self.ts.spectrum, 0.1)
        print(closest_peak_1, " ", closest_peak_2)

    def test_annotation(self):
        annotation = si.annotation(self.observed, self.ts.spectrum, 2)
        print(annotation)


if __name__ == "__main__":
    unittest.main()
