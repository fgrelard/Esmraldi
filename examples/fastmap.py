import numpy as np
from esmraldi.fastmap import FastMap
from sklearn.metrics import pairwise_distances

X = np.array([[10, 20, 30, 40], [20, 25, 30, 40], [10, 20, 30, 50], [1, 2, 3, 4]])
fm = FastMap(X, 4)
fm.compute_projections()
pd_X =pairwise_distances(X)**2
pd_proj = pairwise_distances(fm.projections)**2

print(fm.projections)
print(pd_X)
print(pd_proj)
