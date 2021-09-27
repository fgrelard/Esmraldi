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

mss = msi.MSImage(mzs, s)
print(mss)
msx = msx.astype(np.float32)
mss = mss.astype(np.float32)
print(type(msx.image), type(mss.image))
print(mss.nnz, len(mss.data))
mss_t = mss.transpose((1,0,2))

assert all(mss.mzs == mss_t.mzs)

msx2 = msi.MSImage(mzs, np.zeros((100, 100, 100)))
msx_concat = np.concatenate((msx,msx2,msx), axis=-1)
print(type(msx), msx_concat.shape, msx_concat.spectra.shape)

mss_delete = np.delete(mss, [9, 10, 11, 20, 21, 22])
print(mss_delete.shape, mss_delete.spectra.shape)

# mss.is_maybe_densify=True
# n = mss[..., 0].mean(axis=-1)
# # n[(n==0).todense()]=1
# out=np.zeros_like(mss)
# div=np.divide(mss, n)


mss.is_maybe_densify = True

from_npy = msx.get_ion_image_index(10)
from_sparse = mss.get_ion_image_index(10)
print(type(from_npy), type(from_sparse))
assert (from_npy == from_sparse).all()

fig, ax = plt.subplots(1,2)
ax[0].imshow(from_npy)
ax[1].imshow(from_sparse)
plt.show()
