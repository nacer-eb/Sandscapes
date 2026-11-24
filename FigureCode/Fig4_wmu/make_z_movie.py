import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

z = np.load("run_save_growth.npz")['Ss_saves'][::10]
z = z / z[:, -1][:, None]



import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append('../../common') # include the common folder

import tqdm

from sandscapes_main import *

import tqdm

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(palette="pastel")

import numpy as np
import matplotlib.colors


colors = [    (0.8352941176, 0.3686274510, 0.0000000000, 1.0),  # 4: #D55E00 Vermilion
              (0.9019607843, 0.6235294118, 0.0000000000, 1.0),  # 1: #E69F00 Orange
              (0.3372549020, 0.7058823529, 0.9137254902, 1.0),  # 2: #56B4E9 Sky-blue
              (0.0000000000, 0.6196078431, 0.4509803922, 1.0),  # 3: #009E73 Bluish-green
          ]



common_cmap = matplotlib.colors.ListedColormap(colors)
common_norm = matplotlib.colors.Normalize(0, 4)


data = np.load("data_catastrophe_2/catastrophe_data_growth.npz") #long_iso
N_fps_space = data["N_fps_space"]
z_mus = data["z_mus"]
size_gamma = data["size_gamma"]
beta = data["beta"]
XIs = data["XIs"]
w_mu = data["w_mu"]


# preprocess the FPs so that only odd numbers are kept
N_fps_space_pp = np.ceil((N_fps_space - 1)/2.0)

slinesize = 100


fig, ax = plt.subplots(1, 1, figsize=(15, 6), dpi=400)

im=ax.pcolormesh(z_mus[:, 0].reshape(size_gamma, size_gamma),
              z_mus[:, 1].reshape(size_gamma, size_gamma),
                 N_fps_space_pp, cmap=common_cmap, norm=common_norm)


ax.set_axisbelow(False)
ax.grid(True, which="both", linewidth=0.8, alpha=0.25)

# Increase fontsize for both x and y axes
ax.tick_params(axis='both', labelsize=16)


# CBAR

from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
cb = fig.colorbar(im, cax=cax)

cb.set_ticks(np.arange(4)+0.5)
cb.set_ticklabels([1, 3, 5, 7], fontsize=16)
cb.set_label("Number of FPs", labelpad=20, fontsize=24)

ax.set_xlabel(r"$\tilde{z}_1$", fontsize=24)
ax.set_ylabel(r"$\tilde{z}_2$", fontsize=24)

ax.set_xlim(np.min(z_mus[:, 0]), np.max(z_mus[:, 0]))
ax.set_ylim(np.min(z_mus[:, 1]), np.max(z_mus[:, 1]))
ax.set_aspect("auto")
plt.tight_layout()


ln, = ax.plot([], [], ".", color="crimson")
ax.set_xlim(z[:, 0].min(), z[:, 0].max())
ax.set_ylim(z[:, 1].min(), z[:, 1].max())

def upd(t):
    if t%100==0: print(t)

    ln.set_data(z[:t, 0], z[:t, 1]); 
    if t>100:
        ln.set_data(z[t-100:t, 0], z[t-100:t, 1]); 
    return ln,

FuncAnimation(fig, upd, frames=range(0, len(z), 10), blit=False).save("z_movie.mp4", fps=30)