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
# DISABLED WARNING --- Enable float64 globally
jax.config.update("jax_enable_x64", True) # WARNING !!!!!!! WARNING !!!!!!!

import jax.numpy as jnp
#from sigdiff_k_metric_jax import *
import numpy as np
from sandscapes_main import *

import matplotlib
matplotlib.rcParams.update({"axes.grid" : True, "grid.color": "(0.1, 0.1, 0.1, 0.05)"})


def cosmetic_3d(ax):
    # keep a few ticks so grid stays

    xt = ax.get_xticks()
    yt = ax.get_yticks()
    zt = ax.get_zticks()

    # keep ticks but hide labels
    ax.set_xticklabels([""] * len(xt))
    ax.set_yticklabels([""] * len(yt))
    ax.set_zticklabels([""] * len(zt))


    # hide tick markers
    ax.tick_params(axis='both', which='both', length=0)

    # keep grid
    ax.grid(True)

    bg_rgba = np.array([245, 248, 252, 255]) / 255.0 

    for axis in (ax.xaxis, ax.yaxis,):
        axis.pane.set_facecolor(bg_rgba)   # set pane color
        axis.pane.set_alpha(0.9)      # adjust transparency (0–1)

    for axis in (ax.zaxis,):
        axis.pane.set_facecolor(bg_rgba)   # set pane color
        axis.pane.set_alpha(0.8)      # adjust transparency (0–1)



    # white background
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")


    for tl in ax.xaxis.get_ticklines():
        tl.set_color("None")

    for tl in ax.yaxis.get_ticklines():
        tl.set_color("None")

    for tl in ax.zaxis.get_ticklines():
        tl.set_color("None")


    # labels
    ax.set_xlabel("x", labelpad=-10, fontsize=25)
    ax.set_ylabel("y", labelpad=-10, fontsize=25)

    ax.set_zlim(np.float64(-0.020833333333333332), np.float64(1.0208333333333333))
    ax.view_init( azim= 26.002123648873255 , elev= 6.301249500708365 , roll= 0.0 )
    ax.set_box_aspect((1, 1, 1.1))  # make x twice as long as y,z



d_x = 2.0
d_y = 2.6
Is = jnp.asarray([ [-d_x/2.0, 0], [d_x/2.0, 0], [0, d_y] ])
K = 100

X_0 = jnp.zeros((K, 2)) + jnp.mean(Is, axis=0)
tmax = 800
lr= 0.03
noise_level = 7

seed = 42
key = jax.random.key(seed)
keys = jax.random.split(key, (tmax))
gamma = 0




cmap_tab10 = plt.get_cmap("tab10")
norm_tab10 = matplotlib.colors.Normalize(0, 10)
Is_colors = np.asarray([[176.0, 213.0, 173.0, 255.0], [141, 211.0, 242.0, 255.0], [250.0, 205.0, 211.0, 255.0]])/255.0
Is_colors = np.array([
    [120, 190, 120, 255],   # greener, slightly darker
    [ 90, 170, 230, 255],   # more saturated blue
    [240, 150, 170, 255],   # stronger pink
]) / 255.0



def plot_ax(ax, X_0, d_x, d_y, beta, gamma, tmax, lr, noise_level, keys):
    Ms_saves = training_loop(X_0, X_0, Is, jnp.asarray([beta]), gamma, tmax, lr, noise_level, keys)
    tmax, K, N = Ms_saves.shape


    ax.set_ylabel("y", labelpad=10)
    ax.set_xlabel("x", labelpad=10)
    ax.set_xlim(-d_y/2.0-0.1, d_y/2.0+0.1)
    ax.set_ylim(-0.1, d_y+0.1)


    Ms_colors = np.exp(-beta * distance.cdist(Ms_saves[-1], Is)**2)@Is_colors
    Ms_colors = Ms_colors/Ms_colors[:, -1][:, None]


    ts = np.linspace(0, 1, tmax)
    for k in range(K):
        ax.plot(Ms_saves[:, k, 0], Ms_saves[:, k, 1], ts, color=Ms_colors[k], alpha=1.0, lw=0.3)
        ax.plot(Ms_saves[:, k, 0], Ms_saves[:, k, 1], 0*ts, color="grey", alpha=0.1, lw=0.2)

    cosmetic_3d(ax)


    
fig = plt.figure(figsize=(21, 7), dpi=200)

for i, beta in enumerate([1.5, 3.0, 11.0]):
    ax = fig.add_subplot(1, 3, 1+i, projection='3d')

    if i ==0:
        ax.set_zlabel("time", rotation=90, labelpad=-10, fontsize=25)

    plot_ax(ax, X_0, d_x, d_y, beta, gamma, tmax, lr, noise_level, keys)


plt.subplots_adjust(top=1.0,
bottom=0.0,
left=0.0,
right=1.0,
hspace=0.0,
wspace=0.0)

import os
os.makedirs("Fig", exist_ok=True)
plt.savefig("Fig/A_3d_triplot.png")
