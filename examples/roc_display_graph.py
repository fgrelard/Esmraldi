import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy.spatial.distance as distance
from sklearn.manifold import TSNE
import re
import esmraldi.utils as utils
import mplcursors


def text_annotation(index):
    k = cond[index]
    current_mz = mzs[k]
    if current_mz not in mzs_annotated:
        t = "{:.2f}".format(current_mz)
    else:
        t = "{:.2f}".format(current_mz) + "(" + annotated[k] + ")"
    return t

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .xlsx")
parser.add_argument("-v", "--value", help="Value", default=0.7)
parser.add_argument("--annotations", help="Annotations from Metaspace", default=None)
args = parser.parse_args()

input_name = args.input
value = float(args.value)
annotation_name = args.annotations
print(value)

if annotation_name is not None:
    data = pd.read_csv(annotation_name, header=2)
    mzs_csv = np.array(data.mz)
    mzs_from_annotation, indices = np.unique(data.mz, return_index=True)
    molecule_names = np.array([re.sub(r'([0-9]+)\(([^()]+)\)', r'\1', s.split(", ")[0]) for s in data.moleculeNames])
    molecule_names = molecule_names[indices]

roc_values_df = pd.read_excel(input_name)
roc_values = np.array(roc_values_df).T
mzs = roc_values[0, :]

indices = utils.indices_search_sorted(mzs, mzs_from_annotation)
current_step = 14 * mzs / 1e6
indices_ppm = np.abs(mzs_from_annotation[indices] - mzs) < current_step
mzs_annotated = mzs[indices_ppm]
annotated = np.zeros_like(mzs, dtype=object)
annotated[indices_ppm] = molecule_names[indices[indices_ppm]]

region_names = np.array(roc_values_df.columns[1:])
roc_values = roc_values[1:, :]
n = roc_values.shape[-1]

diffs = np.abs(np.diff(np.sort(roc_values)))
smallest_value = diffs[diffs > 0].min()
print("Smallest", smallest_value)

n_reg = roc_values.shape[0]

indices = np.argwhere((roc_values > value))
colors = np.zeros(n).astype(int) + n_reg

#Determining colors according to highest AUC
for region, mz_index in indices:
    previous_region = colors[mz_index]
    if previous_region != n_reg:
        previous_roc = roc_values[previous_region, mz_index] - 0.5
        current_roc =  roc_values[region, mz_index] - 0.5
        if previous_roc < current_roc and current_roc > 0:
            colors[mz_index] = region
    else:
        colors[mz_index] = region

cond = np.where(colors != n_reg)[0]
roc_values = roc_values[:, cond]
colors = colors[cond]
n = roc_values.shape[-1]
pairwise_distances = 1e6 * np.ones((n,n))


#Find ROC distance
#Expressed as min AUC distance over regions
#Multiplied by a factor depending on whether the
#color is the same or not
for i in range(1, n_reg):
    roc = roc_values[i, :, None].astype(float)
    if np.isnan(roc).all():
        continue
    pd = distance.pdist(roc, "minkowski", p=1)
    factor = np.abs(0.5 - roc)
    k = np.maximum(factor, factor.T)
    k = np.where(colors[:, None] == colors[:, None].T, 1, 1/smallest_value)
    pd = distance.squareform(pd) * k
    pairwise_distances = np.minimum(pd, pairwise_distances)

tsne = TSNE(n_components=2, metric="precomputed", random_state=0, learning_rate="auto", perplexity=30.0)
pos_array = tsne.fit_transform(pairwise_distances).T

# G = nx.from_numpy_matrix(pairwise_distances)
# pos = nx.kamada_kawai_layout(G)
# pos_array = np.array(list(pos.values())).T

colors_previous = colors.copy()
unique_colors, colors = np.unique(colors, return_inverse=True)
region_names = region_names[unique_colors]
fig, ax = plt.subplots()
scatter = ax.scatter(*pos_array, marker='o', s=50, c=colors, edgecolor='None', cmap="Set3")
for color in unique_colors:
    ind = np.where(colors_previous==color)[0][0]
    t = text_annotation(ind)
    p = pos_array.T[ind]
    ax.text(*p, t)

# texts = []
# for k, p in enumerate(pos_array.T):
#     current_mz = mzs[cond[k]]
#     if current_mz not in mzs_annotated:
#         t = ax.text(*p, "{:.2f}".format(current_mz))
#     else:
#         t = ax.text(*p, annotated[cond[k]] + "{:.2f}".format(current_mz))
#     texts.append(t)


mplcursors.cursor(multiple=True).connect("add", lambda sel: sel.annotation.set_text(text_annotation(sel.index)))
elems = scatter.legend_elements()
legend1 = ax.legend(elems[0], region_names,
                    loc="lower left", title="Regions")
plt.axis('equal')
plt.tight_layout()
plt.show()
