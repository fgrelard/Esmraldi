"""
Compare two mass lists
"""

import csv
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--observed", help="Observed annotation (.csv)")

parser.add_argument("-t", "--theoretical", help="Theoretical annotation (.csv)")
args = parser.parse_args()

theoretical_name = args.theoretical
observed_name = args.observed

observed = pd.read_csv(observed_name, delimiter=";", header=None)
observed = observed[observed.columns[0]].values.tolist()

theoretical = pd.read_csv(theoretical_name, delimiter=";", header=None)
theoretical = theoretical[theoretical.columns[0]].values.tolist()

dists = 0
n = len(theoretical)
n = 20
for i in range(n):
    elem = theoretical[i]
    j = observed.index(elem)
    dists += abs(i - j)

dists /= n
print("Average dist=", dists)
print("Ratio=", dists/len(theoretical))
