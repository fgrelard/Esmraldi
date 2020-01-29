from __future__ import absolute_import

import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import src.speciesrule as sr
import random

class TestSpeciesRule(unittest.TestCase):
    def setUp(self):
        self.ax = sr.SpeciesRule("AX", "MS", mz=132.06, begin=569.26, end=2500, naming_fn=lambda i: "AX"+str(i+3))
        self.matrix = sr.SpeciesRule("Matrix", "M", mz=551, count=1)
        self.ac = sr.SpeciesRule("Ac", "A", mz=42.02, count=3, naming_fn=lambda i: str(i) + "Ac")


    def test_generate(self):
        ac_all = self.ac.species()
        matrix_all = self.matrix.species()
        ax_all = self.ax.species()
        print(ac_all)
        print(matrix_all)
        print(ax_all)

    def test_json_to_species(self):
        species = sr.json_to_species("data/species_rule.json")
        print([s.species() for s in species])


if __name__ == "__main__":
    unittest.main()
