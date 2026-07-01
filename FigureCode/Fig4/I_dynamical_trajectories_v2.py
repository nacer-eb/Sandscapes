import numpy as np
import matplotlib.pyplot as plt
import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
import scipy.spatial.distance as distance
from matplotlib.collections import LineCollection

def get_spatial_indices(data, eps):
    diffs = np.linalg.norm(np.diff(data, axis=0), axis=1)
    cumulative_dist = np.insert(np.cumsum(diffs), 0, 0.0)
    thresholds = np.arange(0, cumulative_dist[-1], eps)
    indices = np.searchsorted(cumulative_dist, thresholds)
    return np.unique(np.append(indices, len(data) - 1))

data = np.load("./run_save_growth.npz")
Xs_saves_all = data["Xs_saves"]
tmax = len(Xs_saves_all)
kmax = Xs_saves_all.shape[1]

custom_borders = [0, 20000, 40000, 60000, 80000, tmax]

K = int(data["K"])
XIs = data["XIs"]
beta = float(data["beta"])

fig, ax = plt.subplots(figsize=(16, 9), dpi=150)

nwin = len(custom_borders) - 1
base_cmap = plt.get_cmap("gray")
colors_cmap = base_cmap(np.linspace(0, 1, nwin))
cmap = ListedColormap(colors_cmap)
norm = BoundaryNorm(custom_borders, nwin)

size = 200
xs = np.linspace(-1.5, 1.5, size)
ys = np.linspace(-1.5, 1.5, size)
Ms_probes_x, Ms_probes_y = np.meshgrid(xs, ys)
Ms_probes = np.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
XIs_colors = np.asarray([[176.0, 213.0, 173.0, 255.0], [141, 211.0, 242.0, 255.0], [250.0, 205.0, 211.0, 255.0]]) / 255.0

space_colors = np.exp(-beta * distance.cdist(Ms_probes, XIs)**2) @ XIs_colors
space_colors /= space_colors[:, -1][:, None]
space_colors[:, -1] *= 0.9

ax.pcolormesh(
    Ms_probes[:, 0].reshape(size, size),
    Ms_probes[:, 1].reshape(size, size),
    space_colors.reshape(size, size, -1),
    zorder=-2
)
ax.scatter(*XIs.T, marker="*", color=XIs_colors, s=600, edgecolor="k", lw=1.5, zorder=20)


for k in range(0, kmax, 10):
    print(k)
    ax.scatter(Xs_saves_all[:, k, 0], Xs_saves_all[:, k, 1], color="k", lw=1)


ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect(1.0)
ax.set_xlabel(r"$g_1$", fontsize=20)
ax.set_ylabel(r"$g_2$", fontsize=20)


plt.savefig("Fig/C_dynamical_trajectories.png")
plt.close(fig)
