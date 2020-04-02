import math
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge

def dot(u, v):
    return u[0] * v[0] + u[1] * v[1]

def norm(v):
    return math.sqrt(math.pow(v[0], 2) + math.pow(v[1], 2))

def ortho_projection(u, v):
    length = dot(u, v) / math.pow(norm(v), 2)
    proj = np.multiply(v, length)
    return proj

def update(i):
    angle = (i * math.pi) / (nb-1)
    x = [min_val * math.cos(angle), max_val * math.cos(angle)]
    y = [-min_val * math.sin(angle), -max_val * math.sin(angle)]
    projections = np.array([ortho_projection(u, [x[1], y[1]]) for u in X])
    proj_min = projections[np.argmin(projections, axis=0)[0]]
    proj_max = projections[np.argmax(projections, axis=0)[0]]
    axis.set_xdata(x)
    axis.set_ydata(y)
    variance.set_xdata([proj_min[0], proj_max[0]])
    variance.set_ydata([proj_min[1], proj_max[1]])
    plt.savefig(output_name + str(i) + ".png")
    # plt.scatter(projections[:, 0], projections[:, 1])

def update_cone(i):
#    ax.clear()
    we = Wedge((0, 0),10,0+i*1.23 ,90-i*2.5,edgecolor='k',facecolor='red',alpha=0.2)
    [p.remove() for p in reversed(ax.patches)]
    ax.add_patch(we)
    plt.savefig(output_name + str(i) + ".png")


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pca", help="PCA or NMF", action="store_true")
parser.add_argument("-o", "--output", help="Output fomes")
args = parser.parse_args()

is_pca = args.pca
output = args.output
output_name, ext = os.path.splitext(output)

plt.rcParams['savefig.bbox'] = 'tight'

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
pca = PCA(n_components=1)
pca.fit(X)
first_component = pca.transform(X)
first_component_xy = pca.inverse_transform(first_component)
min_val = np.amin(X) * 1.5
max_val = np.amax(X) * 1.5
nb = 20

if not is_pca:
    X -= X.min()
    X[:, 0] += 2
fig, ax = plt.subplots(tight_layout=True)
ax.scatter(X[:, 0], X[:, 1])
ax.set_aspect('equal', adjustable='box')

if is_pca:
    ax.axis((-4, 4, -4, 4))
    axis, = ax.plot([0,0.1], [0,0.1], "k")
    variance, = ax.plot([0,0], [0,0], "r", linewidth=4)
    anim = FuncAnimation(fig, update, frames=np.arange(0, nb), interval=200, repeat=False)
else:
    ax.axis((0, 8, 0, 6))
    anim = FuncAnimation(fig, update_cone, frames=np.arange(0, nb), interval=200, repeat=False)

plt.show()

#anim.save("line.gif", dpi=80, writer="imagemagick")
