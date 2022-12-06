import joblib
import argparse
import numpy as np
import os
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.linear_model import Lasso

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input models", nargs="+", type=str)
parser.add_argument("-o", "--output", help="Output model")
parser.add_argument("--lasso", help="Use LASSO", action="store_true")

args = parser.parse_args()

input_names = args.input
output_name = args.output
is_lasso = args.lasso

coefs = []
intercepts = []
dual_gaps = []
n_iter = []
for name in input_names:
    regression = joblib.load(name)
    coefs.append(regression.coef_)
    dual_gaps.append(regression.dual_gap_)
    n_iter.append(regression.n_iter_)
    intercepts.append(regression.intercept_)

coefs = np.mean(coefs, axis=0)
dual_gaps = np.mean(dual_gaps, axis=0)
intercepts = np.mean(intercepts, axis=0)
n_iter = np.mean(n_iter, axis=0)

if is_lasso:
    model = Lasso()
else:
    model = PLSRegression()

model.coef_ = coefs
model.intercept_ = intercepts
model.dual_gap_ = dual_gaps
model.n_iter_ = n_iter

input_name = input_names[0]
mzs_name = os.path.splitext(input_name)[0] + "_mzs.csv"
names_name = os.path.splitext(input_name)[0] + "_names.csv"
y_name = os.path.splitext(input_name)[0] + "_y.csv"

mzs = np.loadtxt(mzs_name)
names = np.loadtxt(names_name, dtype=str)
y = np.loadtxt(y_name, dtype=float, delimiter=",")

joblib.dump(model, output_name)
np.savetxt(os.path.splitext(output_name)[0] + "_mzs.csv", mzs, delimiter=",")
np.savetxt(os.path.splitext(output_name)[0] + "_names.csv", names, delimiter=",", fmt="%s")
# np.savetxt(prefix_name + os.path.splitext(os.path.basename(outname))[0] + "_train.csv", out, delimiter=",")
np.savetxt(os.path.splitext(output_name)[0] + "_y.csv", y, delimiter=",")
print(model.coef_)
