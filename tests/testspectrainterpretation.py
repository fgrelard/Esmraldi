from __future__ import absolute_import

import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import src.spectrainterpretation as si
import random

class TestSpectraInterpretation(unittest.TestCase):
    def setUp(self):
        name = "AX"
        self.naming_fn = lambda i: name + str(i)
        self.adducts = {"1Ac": 42.06, "Fe": 172.6}
        self.references = {"Matrix": 521.02, "AX5": 569}
        random.seed(a=1)
        self.observed = [random.randrange(569, 2000) for i in range(200)]
        print(self.observed)

    def test_molecules_from_rule(self):
        mol = si.molecules_from_range(120.3, 836.1, 102.6, self.naming_fn)
        print(mol)

    def test_adducts_from_rule(self):
        naming_fn = lambda i: "Ac" + str(i+1)
        mol = self.adducts["1Ac"]
        new_adducts = si.molecules_from_rule(3, mol,naming_fn)
        print(new_adducts)

    def test_add_adducts(self):
        name = "AX5"
        mz = self.references[name]
        adducts = si.molecule_adducts(name, mz, self.adducts)
        print(adducts)

    def test_adducts_regexp(self):
        naming_fn = lambda i: str(i+1) + "Ac"
        mol = self.adducts["1Ac"]
        new_adducts = si.molecules_from_rule(3, mol,  naming_fn)
        self.adducts = si.update_spectrum(self.adducts, new_adducts)
        mol = si.molecules_from_range(120.3, 836.1, 102.6, self.naming_fn)
        self.references = si.update_spectrum(self.references, mol)
        theoretical = si.molecule_adducts_regexp(self.references, r"AX.*", self.adducts)
        print(theoretical)

    def test_closest_peak(self):
        closest_peak_1 = si.closest_peak(569, self.references, 0.1)
        closest_peak_2 = si.closest_peak(569.2, self.references, 0.1)
        print(closest_peak_1, " ", closest_peak_2)

    def test_annotation(self):
        naming_fn = lambda i: str(i+1) + "Ac"
        mol = self.adducts["1Ac"]
        new_adducts = si.molecules_from_rule(3, mol,  naming_fn)
        self.adducts = si.update_spectrum(self.adducts, new_adducts)
        mol = si.molecules_from_range(120.3, 836.1, 102.6, self.naming_fn)
        self.references = si.update_spectrum(self.references, mol)
        theoretical = si.molecule_adducts_regexp(self.references, r"AX.*", self.adducts)
        self.references = si.update_spectrum(self.references, theoretical)
        annotation = si.annotation(self.observed, self.references, 2)



if __name__ == "__main__":
    unittest.main()
