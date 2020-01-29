from __future__ import absolute_import

import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import src.speciesrule as sr
from src.theoreticalspectrum import TheoreticalSpectrum
import random

class TestTheoreticalSpectra(unittest.TestCase):
    def setUp(self):
        ax = sr.SpeciesRule("AX", "MS", mz=132.06, begin=569.26, end=2500, naming_fn=lambda i: "AX"+str(i+3))
        matrix = sr.SpeciesRule("Matrix", "M", mz=551, count=1)
        ac = sr.SpeciesRule("Ac", "A", mz=42.02, count=3, naming_fn=lambda i: str(i) + "Ac")
        self.ts = TheoreticalSpectrum([ax, matrix], [ac])

    def test_add_adducts_to_molecules(self):
        ax = self.ts.molecules
        ac = self.ts.adducts
        theoretical = self.ts.add_adducts_to_molecules(ax, ac)
        print(theoretical)

    def test_add_adducts_to_molecules_regexp(self):
        theoretical = self.ts.add_adducts_to_molecules_regexp(r"AX[4-9]", r"[1-2]Ac")
        print(theoretical)



if __name__ == "__main__":
    unittest.main()
