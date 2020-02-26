import argparse
import csv
import src.speciesrule as sr
import src.spectrainterpretation as si
from src.theoreticalspectrum import TheoreticalSpectrum
import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--theoretical", help="Theoretical spectrum")
parser.add_argument("-o", "--observed", help="Observed spectrum (.csv)")
parser.add_argument("-c", "--output_csv", help="Output csv file")
args = parser.parse_args()

theoretical_name = args.theoretical
observed_name = args.observed
output_name = args.output_csv

species = sr.json_to_species(theoretical_name)
ions = [mol for mol in species if mol.category=="Ion"]
adducts = [mol for mol in species if mol.category=="Adduct"]
modifications = [mol for mol in species if mol.category=="Modification"]
print(len(ions), "ions,", len(adducts), "adducts,", len(modifications), "modifications.")

theoretical_spectrum = TheoreticalSpectrum(ions, adducts, modifications)

with open(observed_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=";")
    observed_spectrum = [float(row[0]) for row in csv_reader]

annotation = si.annotation(observed_spectrum, theoretical_spectrum.spectrum, 0.1)

print([v for k, v in annotation.items()][:10])
d = {k:v for k, v in annotation.items() if len(v) > 0}
keys_sorted = {k:v for k,v in sorted(annotation.items(), key=lambda item: item[0])}

if output_name:
    with open(output_name, "w") as f:
        writer = csv.writer(f, delimiter=";")
        for k, v in keys_sorted.items():
            row = [k] + [e for e in v]
            writer.writerow(row)
else:
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint(keys_sorted)
