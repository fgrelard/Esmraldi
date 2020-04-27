"""
Example which annotates an observed spectrum
with a reference theoretical spectrum
"""
import argparse
import csv
import re
import src.speciesrule as sr
import src.spectrainterpretation as si
from src.theoreticalspectrum import TheoreticalSpectrum
import pprint


def column_names(values, ions, adducts, modifications, delimiter="_"):
    """
    Extract different column names
    where observed spectrum is split in
    different columns

    Parameters
    ----------
    values: list
        all existing species names
    ions: list
        SpeciesRule list
    adducts: list
        SpeciesRule list
    modifications: list
        SpeciesRule list
    delimiter: str
        delimiter between species

    Returns
    ----------
    list
        column names
    """
    s = set()
    flat_list = [item for sublist in values for item in sublist]
    value_names = "\n".join(list(flat_list))
    column_names = []
    for ion in ions:
        for add in adducts:
            pattern = re.compile(ion.name + ".*" + add.name.replace("+", "\+"))
            match = pattern.search(value_names)
            if match is not None:
                column_names.append(ion.name + delimiter + add.name)
    for mod in modifications:
        for ion in ions:
            for add in adducts:
                pattern = re.compile(ion.name + ".*" + add.name.replace("+", "\+") + ".*" + mod.name  + "\n")
                match = pattern.search(value_names)
                if match is not None:
                    column_names.append(ion.name + delimiter + add.name + delimiter + mod.name)
    for i in range(len(modifications)):
        first_mod = modifications[i].name
        for j in range(i+1, len(modifications)):
            second_mod = modifications[j].name
            for ion in ions:
                for add in adducts:
                    pattern = re.compile(ion.name + ".*" + add.name.replace("+", "\+") + ".*" + first_mod + ".*" + second_mod + "\n")
                    match = pattern.search(value_names)
                    if match is not None:
                        column_names.append(ion.name + delimiter + add.name + delimiter + first_mod + delimiter + second_mod)
    return column_names


def assign_names_to_columns(columns, values, delimiter="_"):
    """
    Assign species name to corresponding column

    Parameters
    ----------
    columns: list
        column names
    values: list
        all species names
    delimiter: str
        separation between species

    Returns
    ----------
    list
        species names in column
    """
    new_values = []
    for possibilities in values:
        L = [""] * len(columns)
        for name in possibilities:
            split_name = name.split(delimiter)
            max_size = 0
            index = -1
            for i in range(len(columns)):
                col = columns[i]
                split_columns = col.split(delimiter)
                if all([any([elem_col in elem_name for elem_name in split_name]) for elem_col in split_columns]):
                    if len(split_columns) > max_size:
                        max_size = len(split_columns)
                        index = i
            L[index] = name
        new_values.append(L)
    return new_values


def export_theoretical_to_csv(theoretical, filename):
    keys_sorted = {k:v for k,v in sorted(theoretical_spectrum.spectrum.items(), key=lambda item: item[1])}
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=";")
        for k, v in keys_sorted.items():
            writer.writerow([k, v])



parser = argparse.ArgumentParser()
parser.add_argument("-t", "--theoretical", help="Theoretical spectrum")
parser.add_argument("-o", "--observed", help="Observed spectrum (.csv)")
parser.add_argument("-c", "--output_csv", help="Output file with annotated observed spectrum (.csv)")
parser.add_argument("-s", "--separate_species", help="Switch whether species names are split in separate columns", action="store_true")

args = parser.parse_args()

theoretical_name = args.theoretical
observed_name = args.observed
output_name = args.output_csv
is_separate = args.separate_species

species = sr.json_to_species(theoretical_name)
ions = [mol for mol in species if mol.category=="Ion"]
adducts = [mol for mol in species if mol.category=="Adduct"]
modifications = [mol for mol in species if mol.category=="Modification"]
print(len(ions), "ions,", len(adducts), "adducts,", len(modifications), "modifications.")

theoretical_spectrum = TheoreticalSpectrum(ions, adducts, modifications)

print(theoretical_spectrum.spectrum)

# export_theoretical_to_csv(theoretical_spectrum, "100_spectra_wheat_theoretical.csv")

with open(observed_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=";")
    observed_spectrum = [float(row[0]) for row in csv_reader]

annotation = si.annotation(observed_spectrum, theoretical_spectrum.spectrum, 0.1)

print([v for k, v in annotation.items()][:10])
d = {k:v for k, v in annotation.items() if len(v) > 0}
keys_sorted = {k:v for k,v in sorted(annotation.items(), key=lambda item: item[0])}

if is_separate:
    values = keys_sorted.values()
    columns = column_names(values, ions, adducts, modifications)
    new_values = assign_names_to_columns(columns, values)
    keys_sorted = {list(keys_sorted.keys())[i]:new_values[i] for i in range(len(new_values))}

if output_name:
    with open(output_name, "w") as f:
        writer = csv.writer(f, delimiter=";")
        if is_separate:
            writer.writerow([""] + columns)
        for k, v in keys_sorted.items():
            row = [k] + [e for e in v]
            writer.writerow(row)
else:
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint(keys_sorted)
