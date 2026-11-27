import numpy as np
import matplotlib.pyplot as plt
import tqdm

from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable

import scipy.spatial.distance as distance

def get_spatial_indices(data, eps):
    diffs = np.linalg.norm(np.diff(data, axis=0), axis=1)
    cumulative_dist = np.insert(np.cumsum(diffs), 0, 0.0)
    thresholds = np.arange(0, cumulative_dist[-1], eps)
    indices = np.searchsorted(cumulative_dist, thresholds)
    return np.unique(np.append(indices, len(data) - 1))



tmin = 0
data_h = np.load("/localhome/nacer/data_4_wmu/run_save_growth.npz")
Ss_saves_h = data_h['Ss_saves'][:8000]
tmax = len(Ss_saves_h)
w_mu = data_h["w_mu"]
Ss_saves_h = Ss_saves_h/w_mu[None, :]
z_h = Ss_saves_h/Ss_saves_h[:, -1][:, None]

K = data_h["K"]
XIs = data_h["XIs"]
beta = data_h["beta"]
gamma = data_h["gamma"]

Xs_saves_all = np.load("/localhome/nacer/data_4_wmu/Xs_saves.mmap", mmap_mode="r")[:8000, :] #70000


fig, ax = plt.subplots(figsize=(16, 9), dpi=150)

XIs_colors = np.asarray([[176.0, 213.0, 173.0, 255.0], [141, 211.0, 242.0, 255.0], [250.0, 205.0, 211.0, 255.0]])/255.0
ax.scatter(*XIs.T, marker="*", color="k", s=600, edgecolor="k", lw=1.5, zorder=20)

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
cmap_xi2 = LinearSegmentedColormap.from_list("xi2", XIs_colors[1:3])
im = ax.scatter(Xs_saves_all[:, :, 0], Xs_saves_all[:, :, 1], c=z_h, s=2, cmap=cmap_xi2)


ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect(1.0)

ax.set_xlabel(r"$g_1$", fontsize=20)
ax.set_ylabel(r"$g_2$", fontsize=20)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.15)

"""
cbar = fig.colorbar(im, cax=cax, ticks=np.arange(nwin))
cbar.ax.set_yticklabels([f"T-{x}" for x in ["I", "II", "III", "IV", "V"][:nwin]])

cbar.ax.tick_params(labelsize=16, width=2, length=6, colors="black")
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor("black")
cbar.set_label("Trajectory window", fontsize=18, color="black")
"""

#plt.tight_layout()
plt.savefig(f"Fig/C_dynamical_trajectories_topology_colors.png")
plt.close(fig)
