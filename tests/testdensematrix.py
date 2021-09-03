import esmraldi.sparsematrix as sparse
import numpy as np
from sparse import COO

x = np.random.random((100, 100, 100))
x[x < 0.9] = 0  # fill most of the array with zeros

s = sparse.SparseMatrix(x)  # convert to sparse array

# print(s)

s2 = sparse.SparseMatrix(x)
# print(s.data)
sigma = s + s2

sigma2 = s + 2
# print(sigma2)
# s+=x
x=x+s
# print(x+s, s+x)
# print(sigma, s.todense())
# print((sigma == s).all())
# print(s.mean())
print(type(s))
s[..., 3]= 123
print(s[0,0,3])
