import jax
#jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True) # WARNING !!!!!!! WARNING !!!!!!! # Get rid of this at some point

import sys
sys.path.append('../../highdim/common') # include the common folder

import tqdm

from sigdiff_k_metric_jax_betas_opt_mute_reset import *

# SET GPU TO USE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tqdm

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(palette="pastel")


import numpy as np
import matplotlib.colors

#r = jax.random.normal(jax.random.key(42), (100000, 100000))
#print(r)

def calculate_growth_grad(z_base, growth_point, XIs, beta):
    N_cells = 100.0
    gamma_base = jnp.asarray([N_cells*z_base[0], N_cells*z_base[1], N_cells])
    tmax = 4000
    lr = 0.01
    noise_level = 5 # i.e. enough noise to break away from very weak attractors
    keys = jax.random.split(jax.random.key(42), (tmax,))
    mute_cells_ts = jnp.ones((tmax, 1))#*1E-5

    D_2 = calculate_distances(jnp.asarray([growth_point]), XIs)
    S_0 = gamma_base# + calculate_signals(beta, D_2, jnp.asarray([1.0]))[0]
    z_0 = S_0[:-1]/(S_0[-1])
    
    Xs = training_loop_unsave(jnp.asarray([growth_point]), jnp.asarray([growth_point]), XIs,
                              jnp.asarray([beta]), gamma_base, tmax, lr, noise_level, keys, mute_cells_ts)
    #Xs = Xs_saves[-1]


    D_2 = calculate_distances(Xs, XIs)
    S_f = calculate_signals(beta, D_2, jnp.asarray([1.0]))[0]+gamma_base
    z_f = S_f[:-1]/(S_f[-1])

    dz = z_f - z_0
    return dz




colors = [    (0.8352941176, 0.3686274510, 0.0000000000, 1.0),  # 4: #D55E00 Vermilion
              (0.9019607843, 0.6235294118, 0.0000000000, 1.0),  # 1: #E69F00 Orange
              (0.3372549020, 0.7058823529, 0.9137254902, 1.0),  # 2: #56B4E9 Sky-blue
              (0.0000000000, 0.6196078431, 0.4509803922, 1.0),  # 3: #009E73 Bluish-green
          ]



common_cmap = matplotlib.colors.ListedColormap(colors)
common_norm = matplotlib.colors.Normalize(0, 4)


data = np.load("data_link/catastrophe_data_growth.npz") #long_iso
N_fps_space = data["N_fps_space"]
gammas = data["gammas"]
size_gamma = data["size_gamma"]
beta = data["beta"]
XIs = data["XIs"]



# Still debugging..
dz = jax.vmap(calculate_growth_grad, in_axes=(0, None, None, None))(gammas, XIs[0],
                                                                    XIs, beta)

# preprocess the FPs so that only odd numbers are kept
N_fps_space_pp = np.ceil((N_fps_space - 1)/2.0)

slinesize = 100

fig, ax = plt.subplots(1, 1, figsize=(15, 6), dpi=400)

#convergence_pnt = np.asarray([[1145.06033642,   105.70164595]])
#ax.scatter(convergence_pnt[0, 0], convergence_pnt[0, 1], color="grey", s=100, zorder=12)

im=ax.pcolormesh(gammas[:, 0].reshape(size_gamma, size_gamma),
              gammas[:, 1].reshape(size_gamma, size_gamma),
                 N_fps_space_pp, cmap=common_cmap, norm=common_norm)

X = gammas[:, 0].reshape(size_gamma, size_gamma)
Y = gammas[:, 1].reshape(size_gamma, size_gamma)
U = dz[:, 0].reshape(size_gamma, size_gamma)
V = dz[:, 1].reshape(size_gamma, size_gamma)

n_y = 16
n_x = 20

starts_x0    = np.c_[np.full(n_y, X.min()),    np.linspace(Y.min(), Y.max()*0.95, n_y)]
starts_x2000 = np.c_[np.full(n_y, X.max()), np.linspace(150, Y.max()*0.95, n_y)]
starts_y0    = np.c_[np.linspace(X.min(), X.max(), n_x), np.full(n_x, Y.min())]

start_points = np.vstack([starts_x2000, starts_y0])

ax.streamplot(
    X, Y, U, V,
    color="white",
    linewidth=3,
    start_points=start_points,
    arrowsize=0.0,
    density=10.0,
    maxlength=10.0,
    integration_direction="forward"
)


ax.streamplot(
    X, Y, U, V,
    color="white",
    linewidth=3,
    start_points=starts_x0,
    arrowsize=0.0,
    density=10.0,
    maxlength=slinesize,
    integration_direction="forward"
)


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

ax.set_xlim(np.min(gammas[:, 0]), np.max(gammas[:, 0]))
ax.set_ylim(np.min(gammas[:, 1]), np.max(gammas[:, 1]))

ax.set_aspect(1.0)
plt.tight_layout()

import os
os.makedirs("Fig", exist_ok=True)
#plt.show()
plt.savefig("Fig/catastrophe_plot_G_growth.png")
