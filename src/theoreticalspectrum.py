import re
import numpy as np
import itertools

class TheoreticalSpectrum:
    def __init__(self,  molecules, adducts, modifications=[]):
        self.molecules = molecules
        self.adducts = adducts
        self.modifications = modifications
        self.full_molecules = {}
        for mol in self.molecules:
            self.full_molecules.update(mol.species())

        self.spectrum = self.add_all_adducts_to_molecules(self.full_molecules, self.adducts)


    def mix_species(self, species, optional=False):
        family_numbers = np.unique([s.family_number for s in species])
        if len(family_numbers) == 0:
            return []
        families = [ [s for s in species if s.family_number == i] for i in family_numbers]
        if optional:
            families = [ l+[None] for l in families]
        mix = np.stack(np.meshgrid(*families)).T.reshape(-1, len(families))
        #Remove None elements if they exist
        b = []
        for l in mix:
            cleaned = [elem for elem in l if elem is not None]
            if len(cleaned):
                b.append(np.array(cleaned))
        mix = np.array(b)
        return mix


    def merge_dicts_mz(self, dict1, dict2):
        D = {}
        for k1, v1 in dict1.items():
            for k2, v2 in dict2.items():
                D[k1+k2] = v1+v2
        return D

    def expand_mix(self, mix):
        new_dict = mix[0].species()
        for i in range(1, len(mix)):
            new_dict = self.merge_dicts_mz(new_dict, mix[i].species())
        return new_dict

    def molecules_regexp(self, adduct, molecules):
        pattern = re.compile(adduct.adduct_fn)
        list_names = '\n'.join(list(molecules.keys()))
        matches = pattern.findall(list_names)
        molecules_matching = {k:molecules[k] for k in matches if k in molecules}
        return molecules_matching

    def mix_molecules_regexp(self, mix, molecules):
        molecules_accepted = []
        for mol in mix:
            molecules = self.molecules_regexp(mol, molecules)
            molecules_accepted.append(molecules)
        molecules_matching_all = molecules_accepted[0]
        for i in range(1, len(molecules_accepted)):
            d1 = molecules_accepted[i]
            molecules_matching_all = {k:v for k, v in molecules_matching_all.items() if k in d1}
        return molecules_matching_all


    def add_all_adducts_to_molecules(self, molecules, adducts):
        spectrum = {}
        mix = self.mix_species(adducts)
        mix_modifications = self.mix_species(self.modifications, optional=True)
        for m in mix:
            expanded = self.expand_mix(m)
            theoretical = self.add_adduct_to_molecules(molecules, expanded)
            spectrum.update(theoretical)
        molecules_with_adducts = spectrum.copy()
        for m in mix_modifications:
            molecules_re = self.mix_molecules_regexp(m, molecules_with_adducts)
            modifications = self.expand_mix(m)
            theoretical = self.add_adduct_to_molecules(molecules_re, modifications)
            spectrum.update(theoretical)
        return spectrum

    def add_adduct_to_molecules(self, molecules, adduct):
        mol_with_adducts = {}
        for name, mz in adduct.items():
            for mol_name, mol_mz in molecules.items():
                current_mz = mol_mz + mz
                current_name = mol_name + "_" + name
                mol_with_adducts[current_name] = current_mz
        return mol_with_adducts

    def add_adducts_to_molecules_regexp(self, adduct):
        molecules = self.molecules_regexp(adduct, self.full_molecules)

        mol_with_adducts = self.add_adduct_to_molecules(molecules, adduct.species())
        return mol_with_adducts
