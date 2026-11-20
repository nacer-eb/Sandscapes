import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append('../../common') # include the common folder

import tqdm

from sandscapes_main import calculate_gradient
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
xi_y = 1.3 #jnp.sqrt(3)/2.0 #1.0
XIs = jnp.asarray([ [-xi_x, -xi_y], [xi_x, -xi_y], [0, xi_y] ])
beta = 1.5 # was 1.5


def get_grad(Ms_probe, gamma, XIs, beta, w_mu):
    grad = calculate_gradient(Ms_probe, XIs, beta, gamma, jnp.asarray([1E-5]), w_mu)
    return grad

def get_grads_field(Ms_probes, gamma, XIs, beta, w_mu):
    return jax.vmap(get_grad, in_axes=(0, None, None, None, None))(Ms_probes, gamma, XIs, beta, w_mu)



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
def calculate_N_fps(Ms_probes, z_mu, gamma, XIs, beta, w_mu):
    gamma_eff = gamma + z_mu*w_mu

    grads = get_grads_field(Ms_probes, gamma_eff, XIs, beta, w_mu)[:, 0].reshape(size, size, -1)
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


calculate_catastrophe = jax.vmap(calculate_N_fps, in_axes=(None, 0, None, None, None, None))

def calculate_catastrophe_shard(Ms_probes, z_mus, gamma, XIs, beta, w_mu):
    N_fps_gamma_space = jax.jit(calculate_catastrophe, in_shardings=(None, in_shard, None, None, None, None), out_shardings=out_shard)(Ms_probes, z_mus, gamma, XIs, beta, w_mu)
    return N_fps_gamma_space

def calculate_catastrophe_byblock(Ms_probes, z_mus, gamma, XIs, beta, w_mu, blocksize):
    
    N_fps_all = jnp.zeros((len(z_mus)), dtype=int)
    for i in tqdm.tqdm(range(0, len(z_mus), blocksize)):
        i_max = min(len(z_mus), i+blocksize)
        z_mus_block = z_mus[i:i_max] # we multiply by w_mu bc here we are simulating various z_mu conditions through gamma

        N_fps_block = calculate_catastrophe_shard(Ms_probes, z_mus_block, gamma, XIs, beta, w_mu)
        N_fps_all =  N_fps_all.at[i:i_max].set(N_fps_block)

    return N_fps_all


w_mu = jnp.asarray([1.0, 0.10, 0.10])

size = 1000#//2
xs = jnp.linspace(-1.5, 1.5, size)
ys = jnp.linspace(-1.5, 1.5, size)
Ms_probes_x, Ms_probes_y = jnp.meshgrid(xs, ys)
Ms_probes = jnp.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
print(Ms_probes.nbytes)


## CALCULATE BROAD Z-SPACE
size_gamma = 500
z_mus_lin_x = jnp.linspace(0.001, 2000.0, size_gamma)
z_mus_lin_y = jnp.linspace(0.001, 600.0, size_gamma)
z_mus_x, z_mus_y = jnp.meshgrid(z_mus_lin_x, z_mus_lin_y)

z_mus = jnp.ones((size_gamma**2, 3))
z_mus = z_mus.at[:, 0].set(z_mus_x.ravel())
z_mus = z_mus.at[:, 1].set(z_mus_y.ravel())

blocksize=len(z_mus)//500

gamma = 0
N_fps_space = calculate_catastrophe_byblock(Ms_probes[:, None, :], z_mus, gamma, XIs, beta, w_mu, blocksize).reshape(size_gamma, size_gamma)

import os
os.makedirs('data_catastrophe_2', exist_ok=True)

jnp.savez("data_catastrophe_2/catastrophe_data_growth.npz",
         N_fps_space=N_fps_space,
         z_mus=z_mus,
         Ms_probes=Ms_probes,
         size=size, size_gamma=size_gamma, beta=beta, 
         XIs=XIs, w_mu=w_mu)


####






