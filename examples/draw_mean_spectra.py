import numpy as np
import argparse
import pandas as pd
import matplotlib
matplotlib.use("module://mplcairo.qt")
import matplotlib.pyplot as plt
from mplcairo import operator_t


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .xlsx")
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
output_name = args.output

values_df = pd.read_excel(input_name)
values = np.array(values_df)
mzs = values[1:, 0]
means = values[1:, 1::3]
regions = np.array(values_df.columns[~values_df.columns.str.match("Unnamed")])

first_indices = (regions == "Red Pulp") | (regions == "White Pulp")

second_indices = (regions == "20220622 Rate2#19 LymphB") | (regions == "20220622 Rate2#19 LymphT") | (regions == "20220622 Rate2#19 Macrophage")

regions = np.array([r.replace('20220622 Rate2#19 ', '') for r in regions])

print(first_indices, second_indices)
print(regions.shape)
print(means.shape)
print(mzs[:10])
print(values.shape)


colors = ["c", "m", "y"]

def draw(mzs, means, indices, colors):
    mean_indices = means[:, indices]
    fig, ax = plt.subplots()
    for i in range(mean_indices.shape[-1]):
        m = mean_indices[:, i]
        pc = ax.plot(mzs, m, color=colors[i])
        if i >= 1:
            operator_t.DARKEN.patch_artist(pc[0])
    ax.legend(regions[indices])
    ax.set_xlabel("m/z")
    ax.set_ylabel("Abundance")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    plt.show()

draw(mzs, means, first_indices, colors)
draw(mzs, means, second_indices, colors)
