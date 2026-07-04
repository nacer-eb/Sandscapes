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


Xs_saves_all = np.load("Xs_saves.mmap", mmap_mode="r")[:80000, :] #70000
#Xs_saves_all = Xs_saves
tmax = len(Xs_saves_all)
gamma_vlines = np.unique(np.r_[np.arange(0, tmax-1, 10000, dtype=int)[[0, 1, 2, 4, 6]], tmax-1])

data = np.load("run_save_growth.npz")
K = data["K"]
XIs = data["XIs"]
beta = data["beta"]
gamma = data["gamma"]


fig, ax = plt.subplots(figsize=(16, 9), dpi=150)


nwin = len(gamma_vlines) - 1
cmap_name = "gray"
base_cmap = plt.get_cmap(cmap_name)
colors = base_cmap(np.linspace(0, 1, nwin))   # first nwin colors
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(nwin + 1) - 0.5, nwin)


size = 200
xs = np.linspace(-1.5, 1.5, size)
ys = np.linspace(-1.5, 1.5, size)
Ms_probes_x, Ms_probes_y = np.meshgrid(xs, ys)
Ms_probes = np.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
XIs_colors = np.asarray([[176.0, 213.0, 173.0, 255.0], [141, 211.0, 242.0, 255.0], [250.0, 205.0, 211.0, 255.0]])/255.0
space_colors = np.exp(-beta*distance.cdist(Ms_probes, XIs)**2)@XIs_colors
space_colors /= space_colors[:, -1][:, None]
space_colors[:, -1] *= 0.9
ax.pcolormesh(Ms_probes[:, 0].reshape(size, size), Ms_probes[:, 1].reshape(size, size), space_colors.reshape(size, size, -1), zorder=-2)
ax.scatter(*XIs.T, marker="*", color=XIs_colors, s=600, edgecolor="k", lw=1.5, zorder=20)

from matplotlib.collections import LineCollection
for w in range(len(gamma_vlines) - 1):
    tmin = gamma_vlines[w]
    tmax = gamma_vlines[w + 1]

    kmin = (tmin * 20) // 30
    kmax = (tmax * 20) // 30
    kmax = min(kmax, Xs_saves_all.shape[1])

    print("window", w, "t:", tmin, tmax, "k:", kmin, kmax)

    Xs_saves = Xs_saves_all[tmin:tmax, ::1]

    segments = []
    for k in tqdm.tqdm(range(kmin, kmax), desc=f"window {w}"):
        indx = get_spatial_indices(Xs_saves[:, k], 0.01)
        Xs_ss = Xs_saves[indx, k]

        segments.append(Xs_ss)
        
    lw = 1
    
    lc = LineCollection(
        segments,
        colors=[cmap(norm(w))],
        linewidths=lw,
        alpha=0.2,
        antialiaseds=False,
        zorder=5,
    )
    ax.add_collection(lc)


ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect(1.0)

ax.set_xlabel(r"$g_1$", fontsize=20)
ax.set_ylabel(r"$g_2$", fontsize=20)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.15)

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax, ticks=np.arange(nwin))
cbar.ax.set_yticklabels([f"T-{x}" for x in ["I", "II", "III", "IV", "V"][:nwin]])

cbar.ax.tick_params(labelsize=16, width=2, length=6, colors="black")
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor("black")
cbar.set_label("Trajectory window", fontsize=18, color="black")


#plt.tight_layout()
plt.savefig(f"Fig/C_dynamical_trajectories.png")
plt.close(fig)
