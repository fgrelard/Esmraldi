"""
Gives a complete analysis of the
selected peaks and their annotation
Helps finding correspondences between different
samples

Generates a summary file (.xls)
"""
import argparse
import xlsxwriter
import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name
from PIL import Image
from io import BytesIO

import esmraldi.imzmlio as imzmlio
import esmraldi.speciesrule as sr
import esmraldi.spectraprocessing as sp
import esmraldi.spectrainterpretation as si
from esmraldi.theoreticalspectrum import TheoreticalSpectrum

def get_col_widths(data):
    """
    Compute the maximum column width
    (in number of characters)

    Parameters
    ----------
    data: np.ndarray
        input data

    Returns
    ----------
    int
        max width
    """
    mylen = np.vectorize(len)
    lengths = mylen(data.astype('str'))
    maximum = np.amax(lengths, axis=1)
    return maximum

def split_name(name):
    """
    Split the species name
    to a readable format

    Example:
    Mol_Adduct_Modif is converted to:
    Mol Modif (Adduct)

    Parameters
    ----------
    name: str
        input name

    Returns
    ----------
    str
        split name
    """
    for c in "[']":
        name = name.replace(c, "")
    names = name.split(",")
    new_name = ""
    for i, n in enumerate(names):
        list_names = n.split("_")
        if len(list_names) > 1:
            new_name += list_names[0] + ("." + ".".join(list_names[2:]) if len(list_names) > 2 else "") + " (" + list_names[1] + ")"
        else:
            try:
                float(name)
                new_name = name
            except:
                new_name += "?"
    return new_name

def dict_to_array(annotation):
    """
    Converts a mz/name dict
    to a numpy array

    Parameters
    ----------
    annotation: list
        annotated species

    Returns
    ----------
    np.ndarray
        annotated species array
    """
    annotation_array = np.array(annotation, dtype=object)
    max_len = max([len(a[0]) for a in annotation_array[..., ::2]])
    values_array = []
    for i in range(annotation_array.shape[-1]-1):
        for j in range(annotation_array.shape[0]):
            dist = annotation_array[j, i+1]
            name = annotation_array[j, i]
            current_dist = split_name(str(dist[0]))
            current_names = [current_dist] + [split_name(name[k]) if k < len(name) and name[k] != "" else None for k in range(max_len)]
            values_array.append(current_names)
    data = np.vstack(values_array)
    return data

def write_mass_list(worksheet, column_names, annotation, mzs, spectra_mzs, mean_spectrum):
    """
    Write annotated mass list to a spreadsheet

    Parameters
    ----------
    worksheet: xlsxwriter.Worksheet
        current worksheet
    column_names: list
        column header names
    annotation: np.ndarray
        data array (annotation)
    mzs: np.ndarray
        mzs list
    spectra_mzs: np.ndarray
        spectra mzs (full list)
    mean_spectrum: np.ndarray
        average intensity values for the species

    """
    annotation_array = np.array(annotation, dtype=object)
    max_len = max([len(v[0]) for v in annotation_array[..., ::2]])
    print(max_len)
    headers = ["Order", "m/z", "Distance", "Annotation"]
    for i in range(len(headers)-1):
        worksheet.write(0, i, headers[i], header_format)
    worksheet.merge_range(0, len(headers)-1, 0, len(headers)-1+max_len-1, headers[-1], header_format)

    worksheet.freeze_panes(1, 1)
    row = 1
    col = 0

    data = dict_to_array(annotation)
    rankings = [int(i+1) for i in range(len(annotation))]
    mzs_float = [split_name(mz) for mz in mzs]

    data = np.vstack((rankings, mzs_float, data.T))

    for d in data:
        worksheet.write_column(1, col, d)
        col+=1

    indices = np.abs(np.array(mzs_float, dtype=float) - spectra_mzs[:, None]).argmin(axis=0)

    for i in range(data.shape[1]):
        col_letter = xl_col_to_name(indices[i]+1)
        worksheet.write_url(i+1, 1, "internal:'Images'!"+col_letter+":"+col_letter, string=str(mzs_float[i]))

    widths = get_col_widths(data)
    for i in range(widths.shape[0]):
        worksheet.set_column(i, i, int(widths[i])+2)




def insertable_image(image, size):
    """
    Converts an image to binary
    for insertion inside a spreadsheet

    Parameters
    ----------
    image: np.ndarray
        numpy image
    size: tuple
        new size

    Returns
    ----------
    BytesIO
        insertable image
    """
    image_i = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')
    new_im = np.array(Image.fromarray(image_i).resize(size))
    im, a_numpy = cv2.imencode(".png", new_im)
    a = a_numpy.tostring()
    image_data = BytesIO(a)
    return image_data

def add_images(worksheet, column_names, mzs, image):
    """
    Add images in spreadsheet

    Parameters
    ----------
    worksheet: xlsxwriter.Worksheet
        spreadsheet
    column_names: list
        column header names
    masses: np.ndarray
        data: mz
    image: np.ndarray
        MALDI image datacube
    """
    for i in range(image.shape[-1]):
        image_i = image[..., i].T
        if len(image.shape) > 2:
            for j in range(image_i.shape[0]):
                image_current = image_i[j, ...]
                start_row = image_i.shape[1]//20
                image_data = insertable_image(image_current, (image_current.shape[0],image_current.shape[1]))
                worksheet.insert_image(j*start_row+1, i+1, "", {'image_data': image_data, 'object_position': 4})


    headers = ["m/z"] + column_names
    for i in range(len(headers)):
        worksheet.write(0, 0, headers[i], header_format)
    worksheet.freeze_panes(1, 0)

    worksheet.write_row(0, 1, mzs, cell_format=left_format)

    widths = get_col_widths(np.array([mzs]))
    width_headers = get_col_widths(np.array([headers]))
    worksheet.set_column(0, 0, width_headers[0])
    max_width = np.amax(widths.astype("int"))
    for i in range(mzs.shape[0]):
        worksheet.set_column(i+1, i+1, max_width)


def gradient(n, start, end):
    """
    Gray-level gradient (hexadecimal)

    Parameters
    ----------
    n:  int
        number of colors
    start: int
        starting gray level
    end: int
        end gray level

    Returns
    ----------
    list
        graient list

    """
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
parser.add_argument("-m", "--mzs", help="MZS file corresponding to MALDI image (csv file)")
parser.add_argument("-o", "--output", help="Output .xlsx file")
parser.add_argument("--skip", help="Skip every nth column of the annotation file.", default=0)
parser.add_argument("--memmap", help="Memory mapped file for imzML", action="store_true")


args = parser.parse_args()

annotation_name = args.annotation
input_name = args.input
mzs_name = args.mzs
output_name = args.output
skip = int(args.skip)
is_memmap = args.memmap


memmap_dir = os.path.dirname(input_name) + os.path.sep + "mmap" + os.path.sep
memmap_basename = os.path.splitext(os.path.basename(input_name))[0]
memmap_image_filename = memmap_dir + memmap_basename + ".npy"
memmap_spectra_filename = memmap_dir + memmap_basename + "_spectra.npy"
memmap_files_exist = (os.path.exists(memmap_dir)
                      and os.path.exists(memmap_image_filename)
                      and os.path.exists(memmap_spectra_filename))

if is_memmap and memmap_files_exist:
    print("Reading from memmap")
    spectra = np.load(memmap_spectra_filename, mmap_mode="r")
    image = np.load(memmap_image_filename, mmap_mode="r")
else:
    imzml = imzmlio.open_imzml(input_name)
    spectra = imzmlio.get_spectra(imzml)
    full_spectra = imzmlio.get_full_spectra(imzml)
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    image = imzmlio.get_images_from_spectra(full_spectra, (max_x, max_y, max_z))
    observed_spectrum, intensities = imzml.getspectrum(0)
    if is_memmap:
        os.makedirs(memmap_dir, exist_ok=True)
        np.save(memmap_image_filename, image)
        np.save(memmap_spectra_filename, spectra)

spectra_mzs = spectra[0, 0]
annotation = []
with open(annotation_name, "r") as f:
    reader = list(csv.reader(f, delimiter=";"))

for row in reader:
    v = [row[i].split(",") for i in range(0, len(row), skip+1)]
    annotation.append(v)

mzs = []
with open(mzs_name, "r") as f:
    reader = list(csv.reader(f, delimiter=";"))

for row in reader:
    v = [row[i] for i in range(0, len(row), 2)]
    mzs += v


max_len = max([len(v) for v in annotation])

column_names = ["Ion (#" +str(i+1) +")" for i in range(max_len)]

mean_spectrum = sp.spectra_mean(spectra)



workbook = xlsxwriter.Workbook(output_name)

header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})


worksheet = workbook.add_worksheet("Mass list")
worksheet2 = workbook.add_worksheet("Images")


write_mass_list(worksheet, [], annotation, mzs, spectra_mzs, mean_spectrum)
add_images(worksheet2, [], spectra_mzs, image)


workbook.close()
