import sys
sys.path.append('../../common') # include the common folder

# SET GPU TO USE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tqdm

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(palette="pastel")

import scipy.spatial.distance as distance

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
#from sigdiff_k_metric_jax import *
import numpy as np
from sandscapes_main import *

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=SMALL_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=SMALL_SIZE)    
plt.rc('ytick', labelsize=SMALL_SIZE)    
plt.rc('legend', fontsize=SMALL_SIZE)    
plt.rc('figure', titlesize=BIGGER_SIZE)

d_x = 2.0
d_y = 2.6
Is = jnp.asarray([ [-d_x/2.0, 0], [d_x/2.0, 0], [0, d_y] ])
K = 100

size = 200
xs = np.linspace(-d_y/2.0-0.1, d_y/2.0+0.1, size)
ys = np.linspace(-0.1, d_y+0.1, size)
Ms_probes_x, Ms_probes_y = np.meshgrid(xs, ys)
Ms_probes = np.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T


#Is_colors = np.zeros((len(Is), 4))
cmap_tab10 = plt.get_cmap("tab10")
norm_tab10 = matplotlib.colors.Normalize(0, 10)
#Is_colors = cmap_tab10(norm_tab10([1, 4, 2])) # next would be 6
Is_colors = np.asarray([[176.0, 213.0, 173.0, 255.0], [141, 211.0, 242.0, 255.0], [250.0, 205.0, 211.0, 255.0]])/255.0

X_0 = jnp.zeros((K, 2)) + jnp.mean(Is, axis=0)
tmax = 800
lr= 0.03
noise_level = 7

seed = 42
key = jax.random.key(seed)
keys = jax.random.split(key, (tmax))
gamma = 0
def plot_ax(ax, X_0, d_x, d_y, beta, gamma, tmax, lr, noise_level, keys, plot_y):
    Ms_saves = training_loop(X_0, X_0, Is, jnp.asarray([beta]), gamma, tmax, lr, noise_level, keys) 
    tmax, K, N = Ms_saves.shape

    ax.set_ylabel("y", labelpad=10)
    ax.set_xlabel("x", labelpad=5)
    ax.set_xlim(-d_y/2.0-0.1, d_y/2.0+0.1)
    ax.set_ylim(-0.1, d_y+0.1)

    ax.scatter(*Is.T, marker="*", color=Is_colors, s=600, edgecolor="k", lw=1.5, zorder=20)

    space_colors = np.exp(-beta*distance.cdist(Ms_probes, Is)**2)@Is_colors
    space_colors /= space_colors[:, -1][:, None]
    space_colors[:, -1] *= 0.9

    ax.pcolormesh(Ms_probes[:, 0].reshape(size, size), Ms_probes[:, 1].reshape(size, size), space_colors.reshape(size, size, -1), zorder=-2)

    # Setting grid to be lighter
    st = sns.axes_style()
    base = st["grid.color"]
    grid_rgba = matplotlib.colors.to_rgba(base, alpha=0.3)
    ax.grid(True, color=grid_rgba, linewidth=0.4)


    for k in range(K):
        ax.scatter(Ms_saves[0, k, 0], Ms_saves[0, k, 1], color="k", alpha=1.0, lw=0.2)
        ax.plot(Ms_saves[:, k, 0], Ms_saves[:, k, 1], color="k", alpha=0.5, lw=0.2)

    if not plot_y:
        ax.set_ylabel("")




fig, axs = plt.subplots(1, 3, figsize=(21, 7), sharey=True)

for i, beta in enumerate([1.5, 3.0, 11.0]):
    plot_ax(axs[i], X_0, d_x, d_y, beta, gamma, tmax, lr, noise_level, keys, i==0)


plt.tight_layout()


import os
os.makedirs("Fig", exist_ok=True)

plt.savefig("Fig/B_triplot.png", bbox_inches="tight")


