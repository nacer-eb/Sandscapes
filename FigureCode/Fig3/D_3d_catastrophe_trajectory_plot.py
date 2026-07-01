import sys
sys.path.append('../../common') # include the common folder

import jax
jax.config.update("jax_enable_x64", True) 

from sandscapes_main import *

import tqdm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import seaborn as sns
sns.set_theme(palette="pastel")

import jax
import jax.numpy as jnp
import numpy as np


import matplotlib.colors

colors = [    (0.8352941176, 0.3686274510, 0.0000000000, 1.0),  # 4: #D55E00 Vermilion
              (0.9019607843, 0.6235294118, 0.0000000000, 1.0),  # 1: #E69F00 Orange
              (0.3372549020, 0.7058823529, 0.9137254902, 1.0),  # 2: #56B4E9 Sky-blue
              (0.0000000000, 0.6196078431, 0.4509803922, 1.0),  # 3: #009E73 Bluish-green
          ]



common_cmap = matplotlib.colors.ListedColormap(colors)
common_norm = matplotlib.colors.Normalize(0, 4)




data = np.load("data_catastrophe/catastrophe_data_5.npz") #long_iso
N_fps_space = data["N_fps_space"]
gammas = data["gammas"]
size_gamma = data["size_gamma"]

# preprocess the FPs to odd numbers
N_fps_space_pp = ((N_fps_space - 1)//2)

data = jnp.load("../Fig1/run_save.npz") # Why?
Xs_saves = data['Xs_saves']
K = data['K']
XIs = data['XIs']
beta = data['beta']
gamma = data['gamma']

tmax = len(Xs_saves)

D_2 = jax.vmap(calculate_distances, in_axes=(0, None))(Xs_saves, XIs)
gammas_eff = gamma[None, :] + jax.vmap(calculate_signals, in_axes=(None, 0))(beta, D_2).sum(axis=1) # aggregated
z = gammas_eff/gammas_eff[:, -1][:, None]



## GENERATE CATASTROPHE PLOT
dpi=500
fig_mpl = plt.Figure(figsize=((size_gamma)/dpi, (size_gamma)/dpi), dpi=dpi)
canvas = FigureCanvas(fig_mpl)
ax = fig_mpl.add_axes([0, 0, 1.0, 1.0])

ax.tick_params(axis='both', which='both', 
               bottom=False, left=False, 
               labelbottom=False, labelleft=False)

# Hide the outer bounding box (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

fig_mpl.patch.set_facecolor((1., 1., 1.))
canvas = FigureCanvas(fig_mpl)
ax.pcolormesh(gammas[:, 0].reshape(size_gamma, size_gamma),
              gammas[:, 1].reshape(size_gamma, size_gamma),
              N_fps_space_pp, cmap=common_cmap, norm=common_norm)
ax.plot(z[:, 0], z[:, 1], zorder=10, color="k", lw=5, alpha=0.5)

from matplotlib.ticker import MultipleLocator
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.25))

ax.set_axisbelow(False)
ax.grid(True, which="both", linewidth=1.0, alpha=0.125)
canvas.draw()
buf = np.asarray(canvas.buffer_rgba())[::-1]/255.0 # origin should be bottom left not top left but mpl..
plt.close(fig_mpl)


## GENERATES 3d PLOT
fig = plt.figure(figsize=(16, 9), dpi=200)
ax = fig.add_subplot(projection='3d', computed_zorder=False)

ax.set_proj_type('ortho')
ax.set_zlim(0, 1500)
ax.set_xlim(0, np.max(gammas))
ax.set_ylim(0, np.max(gammas))

surf = ax.plot_surface(gammas[:, 0].reshape(size_gamma, size_gamma),
                       gammas[:, 1].reshape(size_gamma, size_gamma),
                       np.zeros((size_gamma, size_gamma)), facecolors=buf, shade=False, alpha=0.5, lw=0)

ts = np.linspace(0, tmax, tmax)
ax.plot(z[:, 0], z[:, 1], ts, zorder=10, color="k", lw=5)


ts_i = np.asarray([0, 30, 70, 100, 150, 260, 580, 900], dtype=int)
ax.scatter(z[ts_i, 0], z[ts_i, 1], ts[ts_i], zorder=20, color=(0, 0, 0, 1), s=90, depthshade=False)


import os
os.makedirs("Fig", exist_ok=True)

ax.view_init(elev=18.09, azim=-73.15, roll=0.00)
ax.zaxis.set_rotate_label(False)
ax.zaxis._axinfo['juggled'] = (1, 2, 0) # force 3d axis on the left

ax.set_xlabel(r"$\tilde{z}_1$", fontsize=20)
ax.set_ylabel(r"$\tilde{z}_2$", fontsize=20)
#ax.set_zlabel(r"$Time$", fontsize=20, rotation=90)


fig.patch.set_facecolor('white')  # Changes the outer figure background to white
ax.set_facecolor('white')

ax.tick_params(axis='x', labelsize=14); ax.tick_params(axis='y', labelsize=14); ax.tick_params(axis='z', labelsize=14);


# Changing the tick length
ax.xaxis._axinfo['tick']['inward_factor'] = 0.5
ax.yaxis._axinfo['tick']['inward_factor'] = 0.25

ax.xaxis.line.set_color('black'); ax.yaxis.line.set_color('black'); ax.zaxis.line.set_color('white')
ax.xaxis.line.set_linewidth(2.5); ax.yaxis.line.set_linewidth(2.5); ax.zaxis.line.set_linewidth(1.0)


ax.zaxis.set_pane_color((0, 0, 0, 0.2))
ax.zaxis.line.set_lw(0)
#ax.set_zticks([])
ax.set_zlabel("")

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    # Modifed for splitting figure in 2
    axis.pane.set_facecolor((0.925, 0.925, 0.925, 1.0))  # RGBA, alpha=0
    axis.pane.set_edgecolor((1, 1, 1, 0.0))
ax.set_box_aspect([1, 1, 1.0])


ax.set_box_aspect([1, 1, 1.0])

plt.tight_layout()

zticks = np.arange(0, 1400+200, 200)
ax.set_zticks(zticks)
ax.set_zticklabels(zticks)
ax.tick_params(axis='z', color='none')


#plt.show()
print(f"ax.view_init(elev={ax.elev:.2f}, azim={ax.azim:.2f}, roll={getattr(ax, 'roll', 0):.2f})")
print(f"ax.set(xlim={ax.get_xlim()}, ylim={ax.get_ylim()}, zlim={ax.get_zlim()})")
plt.savefig("Fig/D_3d_catastrophe_trajectory.png", transparent=True)
