import sys
sys.path.append('../../common') # include the common folder

# SET GPU TO USE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tqdm

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import seaborn as sns
sns.set_theme(palette="pastel")

import scipy.spatial.distance as distance

import jax
# DISABLED WARNING --- Enable float64 globally
jax.config.update("jax_enable_x64", True) # WARNING !!!!!!! WARNING !!!!!!!

import jax.numpy as jnp

import numpy as np
from sandscapes_main import *




def cosmetic_3d(ax):
    # keep a few ticks so grid stays

    xt = ax.get_xticks()
    yt = ax.get_yticks()
    zt = ax.get_zticks()


    # keep ticks but hide labels
    ax.set_xticklabels([""] * len(xt))
    #ax.set_yticklabels([""] * len(yt))
    ax.set_zticklabels([""] * len(zt))


    # hide tick markers
    #ax.tick_params(axis='both', which='both', length=0)

    # keep grid
    ax.grid(True)

    # white background
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")

    bg_rgba = np.array([245, 248, 252, 255]) / 255.0   # hex ≈ #F5F8FC

    for axis in (ax.xaxis, ax.yaxis,):
        axis.pane.set_facecolor(bg_rgba)   # set pane color
        axis.pane.set_alpha(0.9)      # adjust transparency (0–1)

    for axis in (ax.zaxis,):
        axis.pane.set_facecolor(bg_rgba)   # set pane color
        axis.pane.set_alpha(0.8)      # adjust transparency (0–1)

    for tl in ax.xaxis.get_ticklines():
        tl.set_color("None")

    for tl in ax.yaxis.get_ticklines():
        tl.set_color("None")
        

    for tl in ax.zaxis.get_ticklines():
        tl.set_color("None")


    ax.yaxis.set_tick_params(length=16, width=1.5, labelsize=15, pad=2)

    # labels
    #ax.set_xlabel("y", labelpad=10)
    #ax.set_zlabel("x", rotation=90, labelpad=10)
    ax.set_ylabel(r"$\beta$", labelpad=30, fontsize=16)

    #ax.view_init( azim= 11.301342935978388 , elev= 9.124868410615269 , roll= 0.0 )
    ax.view_init( azim= 11.023987757468914 , elev= 9.106913408650614 , roll= 0.9618456087647804 )
    ax.view_init( azim= 11.030150407282191 , elev= 9.101749175895284 , roll= 1.4852486321061718 )

    ax.set_xlim(-d_x/2.0 - 0.1, d_x/2.0 + 0.1)
    ax.set_ylim(1.0, 13.0)
    ax.set_zlim(-0.1, d_y + 0.1)
    ax.set_box_aspect((d_x + 0.2, 8.0, d_y + 0.2))
    
    #ax.set_xlim(-d_y/2.0-0.1, d_y/2.0+0.1)
    #ax.set_ylim(-0.1, d_y+0.1)
    #ax.set_ylim(1.0, 13.0)
    #ax.set_box_aspect((1, 3.0, 1))  # make x twice as long as y,z



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
# more pop
Is_colors = np.array([
    [120, 190, 120, 255],   # greener, slightly darker
    [ 90, 170, 230, 255],   # more saturated blue
    [240, 150, 170, 255],   # stronger pink
]) / 255.0

# o.g
Is_colors = np.asarray([[176.0, 213.0, 173.0, 255.0], [141, 211.0, 242.0, 255.0], [250.0, 205.0, 211.0, 255.0]])/255.0



def calc_y_tau_eps(beta, d_x, d_y, x_0=10**(-7), eps=0.1):

    beta_C_x = 2/(d_x**2)

    t_A = - 1.0/(beta/beta_C_x - 1) * jnp.log(eps/x_0)
    t_B = + jnp.log( jnp.abs( (1.0-2.0*eps/d_x) / (1.0-beta_C_x/beta) )  )
    t_C = + 1.0/(beta/beta_C_x - 1.0) * jnp.log(eps*beta*d_x)
    y_n_t_eps = (1.0/3.0) * d_y * jnp.exp(
        t_A + t_B + t_C
    )

    return y_n_t_eps #- d_y/2.0



size = 200
xs = np.linspace(-d_y/2.0-0.1, d_y/2.0+0.1, size)
ys = np.linspace(-0.1, d_y+0.1, size)
Ms_probes_x, Ms_probes_y = np.meshgrid(xs, ys)
Ms_probes = np.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T

def plot_dynamics(ax, X_0, d_x, d_y, beta, gamma, tmax, lr, noise_level, keys):
    Ms_saves = training_loop(X_0, X_0, Is, jnp.asarray([beta]), gamma, tmax, lr, noise_level, keys)
    tmax, K, N = Ms_saves.shape

    Ms_colors = np.exp(-beta * distance.cdist(Ms_saves[-1], Is)**2)@Is_colors
    Ms_colors = Ms_colors/Ms_colors[:, -1][:, None]

    for k in range(K):
        ax.plot(-Ms_saves[:, k, 0], beta, Ms_saves[:, k, 1], color="k", lw=0.4, alpha=0.4, zorder=beta+0.001)

    y_tau_eps = calc_y_tau_eps(beta, d_x, d_y)
    ax.scatter(0, beta, y_tau_eps, color="red", s=20.5, zorder=beta+0.002)

    space_colors = np.exp(-beta*distance.cdist(Ms_probes, Is)**2)@Is_colors
    space_colors /= space_colors[:, -1][:, None]
    space_colors[:, -1] *= 0.2

    ax.plot_surface(Ms_probes[:, 0].reshape(size, size), beta,  Ms_probes[:, 1].reshape(size, size), facecolors=space_colors.reshape(size, size, -1)[:, ::-1], zorder=beta+0.0001 , cstride=1, rstride=1)


    bound_x = d_x / 2.0 + 0.1
    z_min = -0.1
    z_max = d_y + 0.1

    verts = [[(-bound_x, beta, z_min),
              (-bound_x, beta, z_max),
              (+bound_x, beta, z_max),
              (+bound_x, beta, z_min)]]

    rect = Poly3DCollection(verts, facecolors=(1.0, 1.0, 1.0, 0.5), edgecolors=(0.1, 0.1, 0.1, 0.3), linewidths=2, zorder=beta-0.001)
    ax.add_collection3d(rect)
    




fig = plt.figure(figsize=(16, 9), dpi=400)
ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)




#14.0
betas = np.arange(1.5, 13.0, 1.0)
for beta in tqdm.tqdm(betas): #0.5
    plot_dynamics(ax, X_0, d_x, d_y, beta, gamma, tmax, lr, noise_level, keys)

beta = 0


ax.set_yticks(betas)


for i in range(len(betas)-1):
    betas_lin = np.linspace(betas[i], betas[i+1], 100)

    y_tau_eps = calc_y_tau_eps(betas_lin, d_x, d_y)
    ax.scatter(0, betas_lin, y_tau_eps, color="red", s=0.5, zorder=betas[i]+0.001)

cosmetic_3d(ax)

plt.subplots_adjust(top=1.0,
bottom=0.0,
left=0.0,
right=1.0,
hspace=0.0,
wspace=0.0)

import os
os.makedirs("Fig", exist_ok=True)
plt.savefig("Fig/D_3d_beta_plot.png")
