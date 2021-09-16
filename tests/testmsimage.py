import numpy as np
import matplotlib.pyplot as plt

import esmraldi.sparsematrix as sparse
import esmraldi.imzmlio as io
import esmraldi.msimage as msi

from sparse import COO

x = np.random.random((100, 100, 100))
x[x < 0.9] = 0  # fill most of the array with zeros

s = sparse.SparseMatrix(x)  # convert to sparse array
mzs = np.arange(1000).reshape((5, 2, 100))


msx = msi.MSImage(mzs, x)
print(sparse.SparseMatrix(s))
mss = msi.MSImage(mzs, s)

msx = msx.astype(np.float32)
mss = mss.astype(np.float32)
print(type(mss), type(msx))

print(mss.mzs)
from_npy = msx.get_ion_image_index(10)
from_sparse = mss.get_ion_image_index(10)

print((from_npy == from_sparse).all())
