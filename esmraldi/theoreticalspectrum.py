import re
import numpy as np
import itertools

class TheoreticalSpectrum:
    """
    This class aims at generating a theoretical spectrum from a list of species
    """

    def __init__(self,  molecules, adducts, modifications=[]):
        """
        Constructs a theoretical spectrum
        from a list of species: molecules,
        adducts, modifications on molecules.

        The theoretical spectrum is stored in the
        "spectrum" attribute

        Parameters
        ----------
        molecules: list
            list of molecules
        adducts: list
            list of adducts
        modifications: list
            list of modifications

        """
        self.molecules = molecules
        self.adducts = adducts
        self.modifications = modifications
        self.full_molecules = {}
        for mol in self.molecules:
            self.full_molecules.update(mol.species())

        self.spectrum = self.add_all_adducts_to_molecules(self.full_molecules, self.adducts)


    def mix_species(self, species, optional=False):
        """
        Mix species of different families.

        All possible combinations are extracted.

        Parameters
        ----------
        species: list
            SpeciesRule list
        optional: bool
            whether this species if optional or not

        Returns
        ----------
        np.ndarray
            mix of SpeciesRule

        """
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
        """
        Merges two dictionaries, where keys=values
        and values=mz
        by summing their keys and values.

        Parameters
        ----------
        dict1: dict
            first dictionary
        dict2: dict
            second dictionary

        Returns
        ----------
        dict
            merged dictionary

        """
        D = {}
        for k1, v1 in dict1.items():
            for k2, v2 in dict2.items():
                D[k1+k2] = v1+v2
        return D

    def expand_mix(self, mix):
        """
        Generates the full list of species from
        all species rules contained in the mix.

        Maps the masses to all possible names.

        Parameters
        ----------
        mix: np.ndarray
            mix of SpeciesRule

        Returns
        ----------
        dict
             mapping of all species (name to mz)

        """
        new_dict = mix[0].species()
        for i in range(1, len(mix)):
            new_dict = self.merge_dicts_mz(new_dict, mix[i].species())
        return new_dict

    def molecules_regexp(self, adduct, molecules):
        """
        Extracts all molecules with fit the
        regexp in adduct.

        Parameters
        ----------
        adduct: SpeciesRule
            adduct or modification
        molecules: dict
            all molecules

        Returns
        ----------
        dict
            molecules which match the adduct or modification
        """
        pattern = re.compile(adduct.adduct_fn)
        list_names = '\n'.join(list(molecules.keys()))
        matches = pattern.findall(list_names)
        molecules_matching = {k:molecules[k] for k in matches if k in molecules}
        return molecules_matching

    def mix_molecules_regexp(self, mix, molecules):
        """
        All molecules matching regexp from species in a mix.

        Parameters
        ----------
        mix: list
            list of SpeciesRule
        molecules: dict
            all molecules

        Returns
        ----------
        dict
            molecules matching every modification add functions

        """
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
        """
        Add adducts and modifications, and every permutation
        of them to molecules.

        Parameters
        ----------
        molecules: dict
            molecule mapping names to mz
        adducts: list
            adduct species rule list

        Returns
        ----------
        dict
            theoretical spectrum

        """
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
            rules_max = {m[i].name:m[i].count_per_mol for i in range(len(m))}
            modifications = self.expand_mix(m)
            theoretical = self.add_adduct_to_molecules(molecules_re, modifications, rules_max)
            spectrum.update(theoretical)
        return spectrum

    def add_adduct_to_molecules(self, molecules, adduct, rules_max={}):
        """
        Add one adduct to all molecules.

        Parameters
        ----------
        molecules: list
            molecule molecule names to mz
        adduct: SpeciesRule
            adduct or modification
        rules_max: dict
            rules constraining number of adduct per molecule

        Returns
        ----------
        dict
            molecules with every combination of adduct

        """
        mol_with_adducts = {}
        for name, mz in adduct.items():
            names = re.findall('\d*\D+', name)
            keys = [''.join(i for i in n if not i.isdigit()) for n in names]
            current_numbers = [int(i) for i in re.findall("\d+", name)]
            number_max = [rules_max[key] for key in keys if key in rules_max]
            for mol_name, mol_mz in molecules.items():
                current_number = int(re.findall('\d+', mol_name)[0])
                accepted_numbers = [current_number*number_max[i] for i in range(len(number_max))]
                condition = [accepted_numbers[i]>=current_numbers[i] for i in range(len(accepted_numbers))]
                if not all(condition):
                    continue
                current_mz = mol_mz + mz
                current_name = mol_name + "_" + name
                mol_with_adducts[current_name] = current_mz
        return mol_with_adducts

    def add_adducts_to_molecules_regexp(self, adduct):
        """
        Add adducts based on regexp.

        Parameters
        ----------
        adduct: SpeciesRule
            adduct or modification

        Returns
        ----------
        dict
            molecules matching regexp of adduct

        """
        molecules = self.molecules_regexp(adduct, self.full_molecules)

        mol_with_adducts = self.add_adduct_to_molecules(molecules, adduct.species())
        return mol_with_adducts
