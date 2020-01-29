import re

class TheoreticalSpectrum:
    def __init__(self,  molecules, adducts):
        self.molecules = molecules
        self.adducts = adducts
        self.full_molecules = {}
        for mol in self.molecules:
            self.full_molecules.update(mol.species())

        self.spectrum = dict(self.full_molecules)

        for add in adducts:
            theoretical = self.add_adducts_to_molecules_regexp(add)
            self.spectrum.update(theoretical)


    def add_adducts_to_molecules(self, molecules, adducts):
        mol_with_adducts = {}
        for name, mz in adducts.items():
            for mol_name, mol_mz in molecules.items():
                current_mz = mol_mz + mz
                current_name = mol_name + "_" + name
                mol_with_adducts[current_name] = current_mz
        self.spectrum.update(mol_with_adducts)
        return mol_with_adducts

    def add_adducts_to_molecules_regexp(self, adduct):
        theoretical = {}
        pattern = re.compile(adduct.adduct_fn)
        list_names = '\n'.join(list(self.full_molecules.keys()))
        matches = pattern.findall(list_names)

        molecules = {k:self.full_molecules[k] for k in matches if k in self.full_molecules}

        mol_with_adducts = self.add_adducts_to_molecules(molecules, adduct.species())
        return mol_with_adducts
