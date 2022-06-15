import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy.spatial.distance as distance

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .xlsx")
parser.add_argument("-v", "--value", help="Value", default=0.7)
args = parser.parse_args()

input_name = args.input
value = float(args.value)
print(value)

roc_values_df = pd.read_excel(input_name)
mzs = roc_values_df.columns[1:]
roc_values = np.array(roc_values_df)

end = 3
region_names = roc_values[:end, 0]
roc_values = roc_values[:end, 1:]

n_reg = roc_values.shape[0]
print(n_reg)
n = roc_values.shape[-1]

indices = np.argwhere((roc_values > value) | (roc_values < 1-value))
colors = np.zeros(n).astype(int) + n_reg

#Determining colors according to highest AUC
for region, mz_index in indices:
    previous_region = colors[mz_index]
    if previous_region != n_reg:
        previous_roc = np.abs(0.5 - roc_values[previous_region, mz_index])
        current_roc =  np.abs(0.5 - roc_values[region, mz_index])
        if previous_roc < current_roc:
            colors[mz_index] = region
    else:
        colors[mz_index] = region

cond = np.where(colors != n_reg)[0]
roc_values = roc_values[:, cond]
colors = colors[cond]
n = roc_values.shape[-1]
pairwise_distances = 1e6 * np.ones((n,n))

diffs = np.abs(np.diff(np.sort(roc_values)))
smallest_value = diffs[diffs > 0].min()
print(smallest_value)
#Find ROC distance
#Expressed as min AUC distance over regions
#Multiplied by a factor depending on whether the
#color is the same or not
for i in range(n_reg):
    roc = roc_values[i, :, None].astype(float)
    if np.isnan(roc).all():
        continue
    pd = distance.pdist(roc, "minkowski", p=1)
    factor = np.abs(0.5 - roc)
    k = np.maximum(factor, factor.T)
    k = np.where(colors[:, None] == colors[:, None].T, 1, 1/smallest_value)
    pd = distance.squareform(pd) * k
    pairwise_distances = np.minimum(pd, pairwise_distances)

G = nx.from_numpy_matrix(pairwise_distances)
pos = nx.kamada_kawai_layout(G)
pos_array = np.array(list(pos.values())).T
scatter = plt.scatter(*pos_array, marker='o', s=50, c=colors, edgecolor='None', cmap="Set3")
for k, p in pos.items():
    plt.text(*p, "{:.2f}".format(mzs[cond[k]]))

elems = scatter.legend_elements()
legend1 = plt.legend(elems[0], region_names,
                    loc="lower left", title="Regions")
plt.axis('equal')
plt.tight_layout()
plt.show()
