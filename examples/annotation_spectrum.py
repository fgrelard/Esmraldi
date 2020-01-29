import csv

with open("/mnt/d/theoretical_spectrum.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    for row in reader:
        species_type = row[0]
        species_name = row[1]
        species_mz = float(row[2].replace(",", "."))
        begin = row[3]
        end = row[4]
        if species_type.endswith("S"):
            species_begin = float(begin)
            species_end = float(end)
        else:
            species_begin = int(begin)
            species_end = int(end)
