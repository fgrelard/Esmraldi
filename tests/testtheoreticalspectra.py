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
        ax = sr.SpeciesRule("AX", "MS", mz=132.0423, begin=4, end_mz=2500, naming_fn=lambda i: "AX"+str(i))
        matrix = sr.SpeciesRule("Matrix", "M", mz=551, count=1, naming_fn=lambda i:"Matrix")
        h2o = sr.SpeciesRule("H2O", "A", mz=18.0106, count=1, naming_fn=lambda i: "", family_number=0, adduct_fn=r"AX.*")
        na  = sr.SpeciesRule("Na", "A", mz=22.9898, count=1, naming_fn=lambda i: "Na+", family_number=2, adduct_fn=r"AX.*")
        k = sr.SpeciesRule("K", "A", mz=38.9637, count=1, naming_fn=lambda i: "K+", family_number=2, adduct_fn=r"AX.*")
        ac = sr.SpeciesRule("Ac", "A", mz=42.0105, count=3, naming_fn=lambda i: str(i) + "Ac", family_number=1, adduct_fn=r"AX.*")
        fe = sr.SpeciesRule("Fe", "A", mz=176.05, count=3, naming_fn=lambda i: str(i) + "Fe", family_number=1, adduct_fn=r"AX[0-9]{2}.*")
        self.ts = TheoreticalSpectrum([ax, matrix], [h2o, na, k], [ fe, ac])

    def test_merge_dicts_mz(self):
        d1 = self.ts.adducts[0].species()
        d2 = self.ts.adducts[1].species()
        d3 = self.ts.adducts[-1].species()
        new_d = self.ts.merge_dicts_mz(d1, d2)
        new_d = self.ts.merge_dicts_mz(new_d, d3)
        print(new_d)

    def test_expand_mix(self):
        mix = self.ts.mix_species(self.ts.adducts)
        mix_dict = self.ts.expand_mix(mix[0])
        print(mix_dict)

    def test_mix_species(self):
        mix = self.ts.mix_species(self.ts.adducts)
        for m in mix:
            for adduct in m:
                print(adduct.species())
        print([[s.name for s in m] for m in mix])

    def test_mix_molecules_regexp(self):
        mix = self.ts.mix_species(self.ts.adducts)
        mols = self.ts.mix_molecules_regexp(mix[0])
        print(self.ts.full_molecules)
        print(mols)

    def test_add_adduct_to_molecules(self):
        ax = self.ts.full_molecules
        ac = self.ts.adducts[-1].species()
        theoretical = self.ts.add_adduct_to_molecules(ax, ac)
        print(theoretical)

    def test_add_all_adducts_to_molecules(self):
        spectrum = self.ts.add_all_adducts_to_molecules(self.ts.full_molecules, self.ts.adducts)
        print(spectrum)
        print(len(spectrum))

    def test_add_adducts_to_molecules_regexp(self):
        theoretical = self.ts.add_adducts_to_molecules_regexp(self.ts.adducts[0])
        print(theoretical)



if __name__ == "__main__":
    unittest.main()
