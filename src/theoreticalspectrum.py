import re
import numpy as np

class TheoreticalSpectrum:
    def __init__(self,  molecules, adducts):
        self.molecules = molecules
        self.adducts = adducts
        self.full_molecules = {}
        for mol in self.molecules:
            self.full_molecules.update(mol.species())

        self.spectrum = {}
        self.add_all_adducts_to_molecules()


    def mix_adducts(self, adducts):
        family_numbers = np.unique([add.family_number for add in adducts])
        families = [ [add for add in adducts if add.family_number == i] for i in family_numbers]
        mix = np.stack(np.meshgrid(*families)).T.reshape(-1, len(families))
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

    def mix_molecules_regexp(self, mix):
        molecules_accepted = []
        for mol in mix:
            molecules = self.molecules_regexp(mol, self.full_molecules)
            molecules_accepted.append(molecules)
        molecules_matching_all = molecules_accepted[0]
        for i in range(1, len(molecules_accepted)):
            d1 = molecules_accepted[i]
            molecules_matching_all = {k:v for k, v in molecules_matching_all.items() if k in d1}
        return molecules_matching_all


    def add_all_adducts_to_molecules(self):
        mix = self.mix_adducts(self.adducts)
        for m in mix:
            molecules = self.mix_molecules_regexp(m)
            adducts = self.expand_mix(m)
            theoretical = self.add_adduct_to_molecules(molecules, adducts)
            self.spectrum.update(theoretical)

    def add_adduct_to_molecules(self, molecules, adduct):
        mol_with_adducts = {}
        for name, mz in adduct.items():
            for mol_name, mol_mz in molecules.items():
                current_mz = mol_mz + mz
                current_name = mol_name + "_" + name
                mol_with_adducts[current_name] = current_mz
        self.spectrum.update(mol_with_adducts)
        return mol_with_adducts

    def add_adducts_to_molecules_regexp(self, adduct):
        molecules = self.molecules_regexp(adduct, self.full_molecules)

        mol_with_adducts = self.add_adduct_to_molecules(molecules, adduct.species())
        return mol_with_adducts
