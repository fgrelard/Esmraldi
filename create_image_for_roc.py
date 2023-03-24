import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import SimpleITK as sitk
import esmraldi.imzmlio as io

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input xlsx file")
parser.add_argument("-o", "--output", help="Output name")

args = parser.parse_args()

inputname = args.input
outputname = args.output
xls = pd.ExcelFile(inputname)

s = 4
e = 6943

all_intensities = []
lengths = []
for i, sheet_name in enumerate(xls.sheet_names):
    df = pd.read_excel(inputname, sheet_name=i)
    data = np.array(df)
    intensities = data[s:e, 6:].astype(np.float32)
    all_intensities.append(intensities)
    print(intensities.shape[1])
    lengths.append(intensities.shape[1])


mzs = data[s:e, 0]
all_intensities = np.hstack(all_intensities)
all_intensities = all_intensities[..., np.newaxis]
print(all_intensities.shape)

root = os.path.splitext(outputname)[0]
sitk.WriteImage(sitk.GetImageFromArray(all_intensities), outputname)
io.to_csv(mzs, root + ".csv")

previous = 0
for i, length in enumerate(lengths):
    current = previous + length
    print(previous, length, current)
    regions = np.zeros(all_intensities.shape[-2:], dtype=np.uint8)
    print(regions.shape)
    regions[previous:current, 0] = 1
    previous = current
    sitk.WriteImage(sitk.GetImageFromArray(regions),  root + "_" + xls.sheet_names[i] + ".tif")
