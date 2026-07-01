import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append('../../common') # include the common folder

import tqdm

from sandscapes_main import *
import jax
import jax.numpy as jnp

import seaborn as sns
sns.set_theme(palette="pastel")
import matplotlib
import matplotlib.pyplot as plt

import scipy.spatial.distance as distance
import sklearn.cluster as cluster

import numpy as np

xi_x = 1.0
xi_y = 1.3
XIs = jnp.asarray([ [-xi_x, -xi_y], [xi_x, -xi_y], [0, xi_y] ])
beta = 1.5


def get_grad(Ms_probe, gamma, XIs, beta):
    # This is the case where it is a pure probe, i.e. no signal emitted by it -
    # it only serves to measure the gamma-environment i.e. other cells - hence the g_metric
    grad = calculate_gradient(Ms_probe, XIs, beta, gamma, jnp.asarray([1E-5]))
    return grad

def get_grads_field(Ms_probes, gamma, XIs, beta):
    return jax.vmap(get_grad, in_axes=(0, None, None, None))(Ms_probes, gamma, XIs, beta)



@jax.jit
def calculate_nullclines(grads_q):
    DT_q = jnp.sign(grads_q)

    DT_q_diff_x = jnp.sign(jnp.abs(DT_q[2:, 1:-1] - DT_q[:-2, 1:-1])) # if there is a sign flip this will be 1
    DT_q_diff_y = jnp.sign(jnp.abs(DT_q[1:-1, 2:] - DT_q[1:-1, :-2])) # if there is a sign flip this will be 1
    DT_q_diff = 1-(1-DT_q_diff_x)*(1-DT_q_diff_y) # if there is a sign flip along either direction then it will be 1

    return DT_q_diff


@jax.jit
def calculate_nullclines_overlaps(grads):
    DT_x_diff = calculate_nullclines(grads[:, :, 0])
    DT_y_diff = calculate_nullclines(grads[:, :, 1])

    DT_diff = DT_x_diff*DT_y_diff
    return DT_diff

@jax.jit
def calculate_N_fps(Ms_probes, gamma, XIs, beta):
    grads = get_grads_field(Ms_probes, gamma, XIs, beta)[:, 0].reshape(size, size, -1)
    DT_diff = calculate_nullclines_overlaps(grads)
    max_fps = 50
    fps = jnp.argwhere(DT_diff, size=max_fps, fill_value=-1)

    min_dist = 7
    N_fps=0
    # refine fps
    for i in range(max_fps-1):
        nearest_dist = jnp.min(jnp.sqrt(jnp.sum((fps[i][None, :] - fps[i+1:])**2, axis=-1)))
        N_fps += nearest_dist>min_dist
    return grads, fps, N_fps



def get_fps_clustered(gamma_point_2d, Ms_probes):
    gamma = jnp.asarray([*gamma_point_2d, 1.0])
    grads, fps, N_fps = calculate_N_fps(Ms_probes, gamma, XIs, beta)
    fps = fps[fps[:, 0]!=-1]


    # TRANSLATE TO x-y
    fps = Ms_probes.reshape(size, size, -1)[fps[:, 0], fps[:, 1]]

    fps_cluster_labels = cluster.DBSCAN(eps=0.1, min_samples=1).fit_predict(fps)
    fps_clustered = np.zeros((np.max(fps_cluster_labels)+1, 2))
    print(fps_cluster_labels)
    for i in range(len(fps_clustered)):
        fps_clustered[i] = np.mean(fps[fps_cluster_labels==i], axis=0)



    Ms_probes_cpu = np.asarray(Ms_probes, copy=True)
    grads_cpu = np.asarray(grads, copy=True)

    return fps_clustered, grads_cpu, Ms_probes_cpu


def sim_sys(Ms_init, gamma_point_2d, tmax, lr):
    gamma = jnp.asarray([*gamma_point_2d, 1.0])
    
    # Our probe emits no/negligible signal compared to the env. 
    mute_cells_ts = jnp.ones((tmax, 1))*1E-5
    keys = jax.random.split(jax.random.key(42), (tmax,))
    Ms_saves = training_loop(jnp.asarray([Ms_init]), jnp.asarray([XIs[0]]), XIs, jnp.asarray([beta]), gamma, tmax, lr, noise_level=20, keys=keys, mute_cells_ts=mute_cells_ts)

    return Ms_saves



size = 1000
xs = jnp.linspace(-1.5, 1.5, size)
ys = jnp.linspace(-1.5, 1.5, size)
Ms_probes_x, Ms_probes_y = jnp.meshgrid(xs, ys)
Ms_probes = jnp.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
print(Ms_probes.nbytes)



gamma_points = jnp.asarray([[  0.01 ,   0.01 ],
                           [240.  ,  30. ],
                           [950.  ,  98],
                           [320.  , 240.  ],
                           [ 30.0 , 240.  ],
                           [ 98 , 950.  ],
                           [100.  , 100.  ],
                           [500, 15],
                           [15, 500]])



for gamma_point_i in tqdm.tqdm(range(len(gamma_points))):
    fps_clustered, grads_cpu, Ms_probes_cpu = get_fps_clustered(gamma_points[gamma_point_i], Ms_probes[:, None, :])
    Ms_probes_cpu = Ms_probes_cpu[:, 0]


    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.set_xticks([])
    ax.set_yticks([])

    fig.patch.set_alpha(0.0)        # or fig.patch.set_facecolor("none")
    ax.set_facecolor("none")        # keep axes bg transparent too

    for spine in ax.spines.values():
        spine.set_visible(False)

    # For debugging
    #ax.streamplot(Ms_probes_cpu[:, 0].reshape(size, size),
    #              Ms_probes_cpu[:, 1].reshape(size, size),
    #              grads_cpu[:, :, 0], grads_cpu[:, :, 1], density=2.0)

    ax.scatter(*fps_clustered.T, color="k", s=10, zorder=10)

    # For debugging
    for i in range(len(fps_clustered)):
        ax.text(*fps_clustered[i], str(i), color="k", fontsize=12)


    red = np.asarray([240, 163, 169, 255])/255.0
    blue = np.asarray([163, 163, 245, 255])/255.0
    lw = 15*2

    if gamma_point_i == 7:
        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)
        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([0, -0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([0.01, 0.0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)
        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([-0.01, 0.0]), gamma_points[gamma_point_i], 1500, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

    if gamma_point_i == 8:
        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)
        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([0, -0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([-0.01, 0.0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)
        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([0.01, 0.0]), gamma_points[gamma_point_i], 1600, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

    
    if gamma_point_i == 0:
        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([-0.003, 0.0]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[1]+jnp.asarray([+0.008, 0.0]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)


        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([+0.00, -0.005]), gamma_points[gamma_point_i], 5000,
                                np.asarray([-0.003, 0.003]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([+0.00, 0.14]), gamma_points[gamma_point_i], 5000, np.asarray([0.003, 0.003]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)


        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([-0.01, 0.00]), gamma_points[gamma_point_i], 5000, np.asarray([-0.01, -0.01]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0.01, 0.00]), gamma_points[gamma_point_i], 5000, np.asarray([-0.01, -0.01]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)


    if gamma_point_i == 1:
        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([-0.005, 0.0]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0.01, 0.0]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, -0.005]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)



        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, -0.005]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([ -0.005, 0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([ 0.01, 0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)


    if gamma_point_i == 2:
        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, -0.003]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, +0.005]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([-0.005, 0.0]), gamma_points[gamma_point_i], 5000, np.asarray([-0.003, -0.003]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)


        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([-0.001, 0.0]), gamma_points[gamma_point_i], 5000, np.asarray([0.003, 0.003]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)



        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, -0.01]), gamma_points[gamma_point_i], 10000, np.asarray([-0.01, -0.01]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 10000, np.asarray([-0.01, -0.01]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)



    if gamma_point_i == 3:
        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0.0, -0.005]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0.0, 0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([-0.005, 0.0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0.01, 0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)



        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, -0.005]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([-0.005, 0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([ 0.02, 0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)


    if gamma_point_i == 4:
        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([-0.005, 0.0]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0.01, 0]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0 -0.005]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)



        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, -0.005]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([ -0.005, 0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([ 0.01, 0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)



    if gamma_point_i == 5:
        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, -0.003]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0, +0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)

        manifold_line = sim_sys(fps_clustered[3]+jnp.asarray([0.01, 0.0]), gamma_points[gamma_point_i], 5000, np.asarray([-0.003, -0.003]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)


        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0.005, 0.0]), gamma_points[gamma_point_i], 5000, np.asarray([0.003, 0.003]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)



        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, -0.01]), gamma_points[gamma_point_i], 10000, np.asarray([-0.01, -0.01]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 10000, np.asarray([-0.01, -0.01]))
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)


    if gamma_point_i == 6:
        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, 0.01]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)
        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0, -0.01]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([0.01, 0.0]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)
        manifold_line = sim_sys(fps_clustered[2]+jnp.asarray([-0.01, 0.0]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)


        manifold_line = sim_sys(fps_clustered[4]+jnp.asarray([0.003, 0.0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)
        manifold_line = sim_sys(fps_clustered[4]+jnp.asarray([-0.003, 0.0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[4]+jnp.asarray([0.0, 0.015]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)
        manifold_line = sim_sys(fps_clustered[4]+jnp.asarray([0.0, -0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)


        manifold_line = sim_sys(fps_clustered[5]+jnp.asarray([0.01, 0.003]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)
        manifold_line = sim_sys(fps_clustered[5]+jnp.asarray([-0.003, 0.0]), gamma_points[gamma_point_i], 5000, -0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=red, lw=lw)

        manifold_line = sim_sys(fps_clustered[5]+jnp.asarray([0.0, 0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)
        manifold_line = sim_sys(fps_clustered[5]+jnp.asarray([0.0, -0.01]), gamma_points[gamma_point_i], 5000, 0.003)
        ax.plot(manifold_line[:, 0, 0], manifold_line[:, 0, 1], color=blue, lw=lw)


    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.set_aspect(1.0)

    import os
    os.makedirs('dynamical_plots', exist_ok=True)
    plt.savefig("dynamical_plots/dynamical_plot_"+str(gamma_point_i)+".svg", transparent=True) # +".png") #
    plt.close()
