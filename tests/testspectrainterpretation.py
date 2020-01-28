from __future__ import absolute_import

import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import src.spectrainterpretation as si


class TestSpectraInterpretation(unittest.TestCase):
    def setUp(self):
        name = "AX"
        self.naming_fn = lambda i: name + str(i)
        self.adducts = {"1Ac": 42.06, "Fe": 172.6}
        self.references = {"Matrix": 521.02, "AX5": 569}

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

if __name__ == "__main__":
    unittest.main()
