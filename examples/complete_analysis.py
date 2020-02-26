import argparse
import xlsxwriter
import csv
import pprint
import numpy as np
import src.imzmlio as imzmlio
import src.speciesrule as sr
import src.spectraprocessing as sp
import src.spectrainterpretation as si
from xlsxwriter.utility import xl_rowcol_to_cell
from src.theoreticalspectrum import TheoreticalSpectrum

def get_col_widths(data):
    mylen = np.vectorize(len)
    lengths = mylen(data.astype('str'))
    maximum = np.amax(lengths, axis=1)
    return maximum

def write_mass_list(worksheet, masses, mean_spectrum):
    max_len = max([len(v) for k,v in masses.items()])
    headers = ["m/z", "Average intensities"] + ["Ion (#" +str(i+1) +")" for i in range(max_len)]
    for i in range(len(headers)):
        worksheet.write(0, i, headers[i], header_format)

    row = 1
    col = 0
    mz = list(masses.keys())
    values = list(masses.values())
    values_array = [ [names[i] if i < len(names) else None for names in values] for i in range(max_len)]
    data = np.vstack((mz, mean_spectrum, *values_array))

    for d in data:
        worksheet.write_column(1, col, d)
        col+=1

    widths = get_col_widths(data)
    for i in range(widths.shape[0]):
        worksheet.set_column(i, i, int(widths[i]))


def add_table(worksheet, masses, image):

    mz_curated = list(masses.keys())
    names_curated = list(masses.values())

    average_replicates = np.mean(np.mean(image, axis=1), axis=0)
    average_peaks = np.mean(np.mean(image, axis=0), axis=0)
    std_replicates = np.std(np.mean(image, axis=1), axis=0)
    std_peaks = np.std(np.mean(image, axis=0), axis=0)
    variability_replicates = np.divide(std_replicates, average_replicates)
    variability_peaks = np.divide(std_peaks, average_peaks)
    stats = np.vstack((mz_curated, variability_peaks, average_peaks, std_peaks, variability_replicates, average_replicates, std_replicates)).T
    worksheet.add_table(0, 0, stats.shape[0]-1, stats.shape[1]-1, {"data":stats, 'columns':[
        {'header': 'm/z'},
        {'header': 'Variability peaks'},
        {'header': 'Average peaks'},
        {'header': 'Stddev peaks'},
        {'header': 'Variability replicates'},
        {'header': 'Average replicates'},
        {'header': 'Stddev replicates'}]
    })
    for i in range(stats.shape[0]):
        worksheet.write_url(i+1, 0, "internal:'Mass list'!"+xl_rowcol_to_cell(i+1, 0), string=str(mz_curated[i]))

    widths = get_col_widths(stats)
    for i in range(widths.shape[0]):
        worksheet.set_column(i, i, int(widths[i]))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="MALDI image")

parser.add_argument("-a", "--annotation", help="Annotation (csv file)")
parser.add_argument("-m", "--mzs", help="MZS corresponding to MALDI image (optional)")

args = parser.parse_args()

annotation_name = args.annotation
input_name = args.input
mzs_name = args.mzs

if input_name.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(input_name)
    spectra = imzmlio.get_spectra(imzml)
    image = imzmlio.to_image_array(imzml)
    observed_spectrum, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(input_name)).T
    if observed_spectrum_name:
        with open(observed_spectrum_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            observed_spectrum = [float(row[0]) for row in csv_reader]
    else:
        observed_spectrum = [i for i in range(image.shape[2])]
    observed_spectrum = np.asarray(observed_spectrum)

masses = {}
with open(annotation_name, "r") as f:
    reader = csv.reader(f, delimiter=";")
    for row in reader:
        k = float(row[0])
        v = [row[1+i] for i in range(len(row)-1)]
        masses[k] = v

mean_spectrum = sp.spectra_mean(spectra)



masses_curated = {}
mean_spectrum_curated = []
index = 0
for k, v in masses.items():
    if len(v) > 0:
        masses_curated[k] = v
        mean_spectrum_curated.append(mean_spectrum[index])
    index += 1

mean_spectrum_curated = np.array(mean_spectrum_curated)

workbook = xlsxwriter.Workbook("results.xlsx")

header_format = workbook.add_format({'bold': True, 'font_color': 'red'})
worksheet = workbook.add_worksheet("Mass list")
worksheet2 = workbook.add_worksheet("Mass list (curated)")
worksheet3 = workbook.add_worksheet("Statistics")

write_mass_list(worksheet, masses, mean_spectrum)
write_mass_list(worksheet2, masses_curated, mean_spectrum_curated)
add_table(worksheet3, masses, image)

workbook.close()

# pp = pprint.PrettyPrinter(indent=1)
# pp.pprint(keys_sorted)
