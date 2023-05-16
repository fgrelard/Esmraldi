import numpy as np
from numba import jit, njit
from numba.experimental import jitclass
from numba import int32, float64    # import the types

spec = [
    ("X", float64[:, :]),
    ("k", int32),
    ("projections", float64[:, :])
]

@jitclass(spec)
class FastMap:
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.projections = np.zeros(shape=(self.X.shape[0], k))

    def dist2_minus_proj(self, i1, i2):
        p1 = self.X[i1]
        p2 = self.X[i2]
        res2 = np.sum((p1 - p2)**2)
        for j in range(self.k):
            tmp = self.projections[i1, j] - self.projections[i2, j]
            res2 -= tmp*tmp
        return res2

    def furthest(self, index):
        max_d = 0.0
        furthest = 0

        for j in range(self.X.shape[0]):
            d2 = self.dist2_minus_proj(j, index)
            if d2 > max_d:
                max_d = d2
                furthest = j
        return furthest

    def pick_pivots(self):
        pivot2 = self.furthest(0)
        pivot1 = self.furthest(pivot2)
        return pivot1, pivot2

    def compute_projections(self):
        rows = self.X.shape[0]
        for j in range(self.k):
            print("Pick pivots", j)
            a, b = self.pick_pivots()
            d_ab = self.dist2_minus_proj(a, b)
            proj_tmp = np.zeros(shape=(rows))
            for i in range(rows):
                d_ai = self.dist2_minus_proj(a, i)
                d_bi = self.dist2_minus_proj(b, i)
                x = (d_ai + d_ab - d_bi) / (2 * np.sqrt(d_ab))
                proj_tmp[i] = x
            self.projections[:, j] = proj_tmp
        return self.projections
