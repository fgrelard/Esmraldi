import csv
import src.speciesrule as sr
import src.spectrainterpretation as si
from src.theoreticalspectrum import TheoreticalSpectrum

theoretical_name = "data/species_rule.json"
observed_name = "data/peaksel_deisotoped.csv"
species = sr.json_to_species(theoretical_name)
ions = [mol for mol in species if mol.category=="Ion"]
adducts = [mol for mol in species if mol.category=="Adduct"]
print(len(ions), "ions,", len(adducts), "adducts")

theoretical_spectrum = TheoreticalSpectrum(ions, adducts)

with open(observed_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=";")
    observed_spectrum = [float(row[0]) for row in csv_reader]

annotation = si.annotation(observed_spectrum, theoretical_spectrum.spectrum, 0.5)

print({k:v for k, v in annotation.items() if v is not None})
