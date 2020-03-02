import argparse
import xlsxwriter
import csv
import pprint
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

import src.imzmlio as imzmlio
import src.speciesrule as sr
import src.spectraprocessing as sp
import src.spectrainterpretation as si
from src.theoreticalspectrum import TheoreticalSpectrum

def get_col_widths(data):
    mylen = np.vectorize(len)
    lengths = mylen(data.astype('str'))
    maximum = np.amax(lengths, axis=1)
    return maximum

def split_name(name):
    list_names = name.split("_")
    new_name = list_names[0] + ("." + list_names[2] if len(list_names) > 2 else "") + " (" + list_names[1] + ")"
    return new_name

def dict_to_array(masses):
    max_len = max([len(v) for k,v in masses.items()])
    mz = list(masses.keys())
    values = list(masses.values())
    values_array = [ [split_name(names[i]) if i < len(names) else None for names in values] for i in range(max_len)]
    data = np.vstack((mz, *values_array))
    return data

def write_mass_list(worksheet, masses, mean_spectrum):
    max_len = max([len(v) for k,v in masses.items()])

    headers = ["m/z", "Average intensities"] + ["Ion (#" +str(i+1) +")" for i in range(max_len)]
    for i in range(len(headers)):
        worksheet.write(0, i, headers[i], header_format)

    row = 1
    col = 0

    data = dict_to_array(masses)
    data = np.vstack((data[0], mean_spectrum, data[1:]))
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
    worksheet.add_table(0, 0, stats.shape[0], stats.shape[1]-1, {"data":stats, 'columns':[
        {'header': 'm/z'},
        {'header': 'Variability samples'},
        {'header': 'Average samples'},
        {'header': 'Stddev samples'},
        {'header': 'Variability replicates'},
        {'header': 'Average replicates'},
        {'header': 'Stddev replicates'}]
    })

    for i in range(stats.shape[1]):
        worksheet.conditional_format(1, i, stats.shape[0], i,
                                     {'type': '3_color_scale',
                                      'min_color': "#92d050",
                                      'mid_color': "#ffcc00",
                                      'max_color': "#ed6161"})
    for i in range(stats.shape[0]):
        col_letter = xl_col_to_name(i+1)
        worksheet.write_url(i+1, 0, "internal:'Images'!"+col_letter+":"+col_letter, string=str(mz_curated[i]))

    widths = get_col_widths(stats)
    for i in range(widths.shape[0]):
        worksheet.set_column(i, i, int(widths[i]))


def insertable_image(image, size):
    image_i = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')
    new_im = np.array(Image.fromarray(image_i).resize(size))
    im, a_numpy = cv2.imencode(".png", new_im)
    a = a_numpy.tostring()
    image_data = BytesIO(a)
    return image_data

def add_images(worksheet, masses, image):
    data = dict_to_array(masses)


    for i in range(image.shape[-1]):
        image_i = image[..., i].T
        image_data = insertable_image(image_i, (number_samples*20, number_replicates*20))
        worksheet.insert_image(0, i+1, "", {'image_data': image_data, 'object_position': 4})


    max_len = max([len(v) for k,v in masses.items()])

    headers = ["Sample #"+str(i+1) for i in range(number_replicates)] + ["m/z"] + ["Ion (#" +str(i+1) +")" for i in range(max_len)]

    for i in range(len(headers)):
        worksheet.write(i, 0, headers[i], header_format)
    worksheet.freeze_panes(0, 1)

    for i in range(data.shape[0]):
        worksheet.write_row(number_replicates+i, 1, data[i], cell_format=left_format)

    widths = get_col_widths(data)
    width_headers = get_col_widths(np.array([headers]))
    worksheet.set_column(0, 0, width_headers[0])
    max_width = np.amax(widths.astype("int"))
    for i in range(data.shape[1]):
        worksheet.set_column(i+1, i+1, max_width)

def add_reduction(worksheet, reduction_dir):
    files_reduction = os.listdir(reduction_dir)
    csv_name = [reduction_dir + os.path.sep + filename for filename in files_reduction if filename.endswith(".csv")][0]
    image_names = [reduction_dir + os.path.sep + filename for filename in files_reduction if filename.endswith(".png")]
    n = len(image_names) // 2
    gradients = gradient(n, 120, 255)
    formats = []
    for g in gradients:
        f = workbook.add_format({"fg_color":g})
        formats.append(f)

    with open(csv_name) as f:
        csv_reader = csv.reader(f, delimiter=";")
        i = 0
        for row in csv_reader:
            j = 0
            for cell in row:
                worksheet.write(i+12, j, cell, formats[j//3 if j//3 < len(formats) else -1])
                j += 1
            i += 1
    for i in range(0, len(image_names), 2):
        representative = image_names[i]
        score = image_names[i+1]
        np_representative = plt.imread(representative)[..., 0]
        h_representative, w_representative = np_representative.shape[0], np_representative.shape[1]
        w, h = int(number_replicates*20*w_representative/h_representative), number_replicates*20
        worksheet_representative = insertable_image(np_representative, (w, h))
        worksheet.insert_image(0, int(i*1.5), "", {'image_data': worksheet_representative, 'object_position': 4})


def gradient(n, start, end):
    g = []
    for i in range(n):
        gray_level = int(i * (end-start) / (n-1) + start)
        gray_hex = hex(gray_level).replace("0x", "")
        gray_code = "#" + gray_hex * 3
        g.append(gray_code)
    return g

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="MALDI image")

parser.add_argument("-a", "--annotation", help="Annotation (csv file)")
parser.add_argument("-r", "--reduction_dir", help="Dimension reduction analysis directory, as generated by analysis_reduction (optional)")
parser.add_argument("-m", "--mzs", help="MZS corresponding to MALDI image (optional)")
parser.add_argument("-o", "--output", help="Output .xlsx file")

args = parser.parse_args()

annotation_name = args.annotation
input_name = args.input
mzs_name = args.mzs
output_name = args.output
reduction_dir = args.reduction_dir

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

number_samples = image.shape[0]
number_replicates = image.shape[1]

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

workbook = xlsxwriter.Workbook(output_name)

header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})


worksheet = workbook.add_worksheet("Mass list")
worksheet2 = workbook.add_worksheet("Mass list (curated)")
worksheet3 = workbook.add_worksheet("Statistics")
worksheet4 = workbook.add_worksheet("Images")


write_mass_list(worksheet, masses, mean_spectrum)
write_mass_list(worksheet2, masses_curated, mean_spectrum_curated)
add_table(worksheet3, masses, image)
add_images(worksheet4, masses, image)

if reduction_dir:
    worksheet5 = workbook.add_worksheet("Reduction")
    add_reduction(worksheet5, reduction_dir)


workbook.close()

# pp = pprint.PrettyPrinter(indent=1)
# pp.pprint(keys_sorted)
