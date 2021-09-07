import numpy as np
import matplotlib.pyplot as plt

import esmraldi.sparsematrix as sparse
import esmraldi.imzmlio as io

from sparse import COO

x = np.random.random((100, 100, 100))
x[x < 0.9] = 0  # fill most of the array with zeros

s = sparse.SparseMatrix(x)  # convert to sparse array


# s2 = sparse.SparseMatrix(x)

# sigma = s + s2

# sigma2 = s + 2

# x=x+s

s[..., 3]= 123
print(s[0,0,3])

array = (np.arange(8)+1).reshape(4,2)
to_reshape = sparse.SparseMatrix(array)
shape = (2,4)
order = 'F'
reshaped = to_reshape.reshape(shape, order=order)
np_reshaped = array.reshape(shape, order=order)
print((reshaped.todense() == np_reshaped).all())
print("Array=\n",array)
print("Reshape NP=\n",np_reshaped)
print("Reshape sparse=\n", reshaped.todense())

spectra = np.load("data/test_spectra_sparse.npy")
# all_mzs = spectra[:, 0, ...]
# mzs = np.unique(all_mzs[np.nonzero(all_mzs)])

spectra_sparse = sparse.SparseMatrix(spectra)
print(spectra_sparse)

intensities = spectra[:, 1, :]
intensities_sparse = spectra_sparse[:, 1, :]

print("Equality get", (intensities == intensities_sparse.todense()).all())

reshape2 = intensities_sparse.todense().reshape((107,25) + (intensities_sparse.shape[-1],), order='F')


image = io.get_images_from_spectra(spectra, (107,25))
image_fromsparse = io.get_images_from_spectra(spectra_sparse, (107, 25))
image_fromsparse = sparse.SparseMatrix(image_fromsparse)

print((spectra_sparse.todense() == spectra).all())
print((image == reshape2).all())

image = image.transpose((1, 0, 2))
image_fromsparse = image_fromsparse.transpose((1, 0, 2))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.mean(image, axis=-1))
ax[1].imshow(np.mean(image_fromsparse, axis=-1).todense())
plt.show()
