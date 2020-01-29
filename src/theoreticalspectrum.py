import re

class TheoreticalSpectrum:
    def __init__(self,  molecules, adducts):
        self.molecules = {}
        self.adducts = {}
        for mol in molecules:
            self.molecules.update(mol.species())

        for add in adducts:
            self.adducts.update(add.species())
        print(self.molecules)
        self.spectrum = {}


    def add_adducts_to_molecules(self, molecules, adducts):
        mol_with_adducts = dict(self.molecules)
        for name, mz in adducts.items():
            for mol_name, mol_mz in molecules.items():
                current_mz = mol_mz + mz
                current_name = mol_name + "_" + name
                mol_with_adducts[current_name] = current_mz
        self.spectrum.update(mol_with_adducts)
        return mol_with_adducts

    def add_adducts_to_molecules_regexp(self, mol_regexp, adduct_regexp):
        theoretical = {}

        pattern = re.compile(mol_regexp)
        list_names = '\n'.join(list(self.molecules.keys()))
        matches = pattern.findall(list_names)
        molecules = {k:self.molecules[k] for k in matches if k in self.molecules}

        pattern = re.compile(adduct_regexp)
        list_names = '\n'.join(list(self.adducts.keys()))
        matches = pattern.findall(list_names)
        adducts = {k:self.adducts[k] for k in matches if k in self.adducts}

        mol_with_adducts = self.add_adducts_to_molecules(molecules, adducts)
        return mol_with_adducts
