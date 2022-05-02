import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .xlsx")
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
output_name = args.output

roc_values_df = pd.read_excel(input_name)
mzs = roc_values_df.columns[1:]
roc_values = np.array(roc_values_df)

region_names = roc_values[:, 0]
roc_values = roc_values[:, 1:]
condition = (roc_values > 0.6) | (roc_values < 0.4)
print(condition.shape)
for i, region in enumerate(region_names):
    print(mzs.shape, roc_values.shape)
    x_sel = mzs[condition[i]]
    y_sel = roc_values[i][condition[i]]
    plt.plot(x_sel, y_sel, "o", label=region)
    for x,y in zip(x_sel, y_sel):
        plt.annotate('{:.3f}'.format(x), xy=(x,y), xytext=(0,5), textcoords='offset points',ha='center')

plt.legend()
plt.show()
