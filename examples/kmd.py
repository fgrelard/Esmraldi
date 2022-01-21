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

data = np.genfromtxt(inputname, skip_header=0, delimiter=";", filling_values=0)

#CSV file is organized as columns (mzs, intensities, relative_intensities)
mzs = data[:, ::3]
intensities = data[:, 1::3]
kendrick_mass = mzs * round(r) / r
kendrick_mass_defect = kendrick_mass - np.floor(kendrick_mass)

#Colors for different ROIs
colors = ["r", "g", "b"]

fig, ax = plt.subplots()
ax.set_xlabel("m/z")
ax.set_ylabel("KMD")

fig.patch.set(alpha=0)
ax.patch.set(alpha=0)

#Arbitrary threshold to make low intensity points not apparent in the resulting KMD plot
threshold = 1000
log_intensities = np.log10(intensities/threshold)

#Min clipping to avoid 0 valued sizes
size_intensities = 20*np.clip(log_intensities, np.finfo(float).eps, None)

#Iterating over ROIs
for i in range(mzs.shape[-1]):
    pc = ax.scatter(mzs[:, i], kendrick_mass_defect[:, i], s=size_intensities[:, i], c=colors[i], picker=True, ec=None)
    #blending colors
    operator_t.EXCLUSION.patch_artist(pc)
    #picking events
    datacursor(pc, formatter=display_onclick)

plt.show()
