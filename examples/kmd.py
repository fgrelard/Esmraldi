# -*- coding: utf-8 -*-

# Copyright 2021 Florent Grélard
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import numpy as np
import matplotlib
#to get blending effects
matplotlib.use("module://mplcairo.qt")
import matplotlib.pyplot as plt
from mplcairo import operator_t
from mpldatacursor import datacursor
import pandas as pd

def display_onclick(**kwargs):
    label = kwargs["label"]
    label = label.strip("_collection")
    return "mz:" + str(kwargs["x"])+ ", label:" + label

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .csv")
parser.add_argument("-r", "--r_exact_mass", help="Exact mass of group (default m_CH2=14.0156)", default=14.0156)
args = parser.parse_args()

inputname = args.input
r = float(args.r_exact_mass)

# data = np.genfromtxt(inputname, skip_header=0, delimiter=";", filling_values=0)
data_df = pd.read_excel(inputname)
data = np.array(data_df)
#CSV file is organized as columns (mzs, intensities, relative_intensities)
mzs = data[1:, 0]
intensities = data[1:, 1::3]
kendrick_mass = mzs * round(r) / r
kendrick_mass_defect = kendrick_mass - np.floor(kendrick_mass)

#Colors for different ROIs
regions = np.array(data_df.columns[~data_df.columns.str.match("Unnamed")])

second_indices = (regions == "20220622 Rate2#19 LymphB") | (regions == "20220622 Rate2#19 LymphT") | (regions == "20220622 Rate2#19 Macrophage")
print(second_indices)
colors = ["r", "g", "b"]
intensities = intensities.astype(np.float64)
intensities[intensities==0] = intensities[intensities>0].min()
intensities = intensities[:, second_indices]
max_color = intensities.max(axis=-1)
colors = intensities/max_color[:, np.newaxis]
colors_new = np.ones((colors.shape[0], colors.shape[1]+1))
colors_new[:, :-1] = colors
colors = colors_new
colors[:, [0, 1, 2]] = colors[:, [1, 2, 0]]
closest_mz_index = np.abs(mzs - 987.497).argmin()
print(colors[closest_mz_index])
print(intensities[:10, :])

fig, ax = plt.subplots()
ax.set_xlabel("m/z")
ax.set_ylabel("KMD")

fig.patch.set(alpha=0)
ax.patch.set(alpha=0)

#Arbitrary threshold to make low intensity points not apparent in the resulting KMD plot
threshold = 1000

log_intensities = np.log10(max_color*10000)

#Min clipping to avoid 0 valued sizes
size_intensities = 20*np.clip(log_intensities, np.finfo(float).eps, None)
print(kendrick_mass_defect.shape, size_intensities.shape)
#Iterating over ROIs
# for i in range(mzs.shape[-1]):
for i in range(1):
    pc = ax.scatter(mzs, kendrick_mass_defect, s=size_intensities, c=colors, picker=True, ec=None)
    #blending colors
    # operator_t.EXCLUSION.patch_artist(pc)
    #picking events
    datacursor(pc, formatter=display_onclick)

plt.show()
