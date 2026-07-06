import jax
jax.config.update("jax_enable_x64", True) # Required for precise grad/E calc


import sys
sys.path.append('../../common') # include the common folder

import tqdm

from sandscapes_main import *
import jax
import jax.numpy as jnp

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils


# create a 1D device mesh over all local GPUs
devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
mesh = Mesh(devices, ('data',))

in_shard = NamedSharding(mesh, P('data'))
out_shard = NamedSharding(mesh, P('data'))


xi_x = 1.0
xi_y = 1.3 
XIs = jnp.asarray([ [-xi_x, -xi_y], [xi_x, -xi_y], [0, xi_y] ])
beta = 1.5 


def get_grad(Ms_probe, gamma, XIs, beta):
    # This is the case where it is a pure probe, i.e. no signal emitted by it - it only serves to measure the gamma-environment i.e. other cells - hence the g_metric
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
    return N_fps


calculate_catastrophe = jax.vmap(calculate_N_fps, in_axes=(None, 0, None, None))

def calculate_catastrophe_shard(Ms_probes, gammas, XIs, beta):
    N_fps_gamma_space = jax.jit(calculate_catastrophe, in_shardings=(None, in_shard, None, None), out_shardings=out_shard)(Ms_probes, gammas, XIs, beta)
    return N_fps_gamma_space


def calculate_catastrophe_byblock(Ms_probes, gammas, XIs, beta, blocksize):
    assert(len(gammas)%blocksize == 0) # make sure the blocksize makes sense

    i=0
    gammas_block = gammas[i*blocksize:(i+1)*blocksize]
    N_fps_all = calculate_catastrophe_shard(Ms_probes, gammas_block, XIs, beta)

    N_blocks = len(gammas)//blocksize
    print("N_BLOCKS", N_blocks)

    for i in tqdm.tqdm(range(1, N_blocks)):
        gammas_block = gammas[i*blocksize:(i+1)*blocksize]

        N_fps_block = calculate_catastrophe_shard(Ms_probes, gammas_block, XIs, beta)
        N_fps_all = jnp.concatenate((N_fps_all, N_fps_block), axis=0)

    return N_fps_all



size = 500 # //2
xs = jnp.linspace(-1.5, 1.5, size)
ys = jnp.linspace(-1.5, 1.5, size)
Ms_probes_x, Ms_probes_y = jnp.meshgrid(xs, ys)
Ms_probes = jnp.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
print(Ms_probes.nbytes)


## The broad z-space
size_gamma = 1000
gammas_lin = jnp.linspace(0.001, 1000.0, size_gamma)
gammas_x, gammas_y = jnp.meshgrid(gammas_lin, gammas_lin)

gammas = jnp.ones((size_gamma**2, 3))
gammas = gammas.at[:, 0].set(gammas_x.ravel())
gammas = gammas.at[:, 1].set(gammas_y.ravel())

blocksize=len(gammas)//500
N_fps_space = calculate_catastrophe_byblock(Ms_probes[:, None, :], gammas, XIs, beta, blocksize).reshape(size_gamma, size_gamma)

import os
os.makedirs('data', exist_ok=True)
jnp.savez("data/catastrophe_data.npz",
         N_fps_space=N_fps_space,
         gammas=gammas,
         Ms_probes=Ms_probes,
         size=size, size_gamma=size_gamma, beta=beta, XIs=XIs)


#### ------------------------------------------------------------------------- ########
## Zoomed z-space (For the inset)
gammas_lin = jnp.linspace(0.001, 1.0, size_gamma)
gammas_x, gammas_y = jnp.meshgrid(gammas_lin, gammas_lin)

gammas = jnp.ones((size_gamma**2, 3))
gammas = gammas.at[:, 0].set(gammas_x.ravel())
gammas = gammas.at[:, 1].set(gammas_y.ravel())

blocksize=len(gammas)//500
N_fps_space = calculate_catastrophe_byblock(Ms_probes[:, None, :], gammas, XIs, beta, blocksize).reshape(size_gamma, size_gamma)

import os
os.makedirs('data', exist_ok=True)
jnp.savez("data/catastrophe_data_inset.npz",
         N_fps_space=N_fps_space,
         gammas=gammas,
         Ms_probes=Ms_probes,
         size=size, size_gamma=size_gamma, beta=beta, XIs=XIs)





#### ------------------------------------------------------------------------- ########
## Zoomed z-space (For D, E)

gammas_lin = jnp.linspace(0.001, 5.0, size_gamma)
gammas_x, gammas_y = jnp.meshgrid(gammas_lin, gammas_lin)

gammas = jnp.ones((size_gamma**2, 3))
gammas = gammas.at[:, 0].set(gammas_x.ravel())
gammas = gammas.at[:, 1].set(gammas_y.ravel())

blocksize=len(gammas)//500
N_fps_space = calculate_catastrophe_byblock(Ms_probes[:, None, :], gammas, XIs, beta, blocksize).reshape(size_gamma, size_gamma)

import os
os.makedirs('data', exist_ok=True)

jnp.savez("data/catastrophe_data_5.npz",
         N_fps_space=N_fps_space,
         gammas=gammas,
         Ms_probes=Ms_probes,
         size=size, size_gamma=size_gamma, beta=beta, XIs=XIs)

