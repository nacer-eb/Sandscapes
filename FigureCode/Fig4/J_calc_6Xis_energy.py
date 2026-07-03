import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append('../../common') # include the common folder

import tqdm

import jax
jax.config.update("jax_platform_name", "cpu")

from sandscapes_main import *

import jax.numpy as jnp
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default="run_save_growth_5XIs")
args = parser.parse_args()

filename = args.filename 
   
data = jnp.load(f"{filename}.npz")
Xs_saves = data['Xs_saves'][:, ::1]
mute_cells_ts = data["mute_cells_ts"][:, ::1]
K = data['K']
XIs = data['XIs']
beta = data['beta']
gamma_0 = data['gamma']
gamma = jnp.asarray([gamma_0]*len(XIs))

tmax = len(Xs_saves)
a = data['a']



def calculate_signals_all(Xs_saves, XIs, beta, gamma, mute_cells):
    mute_cells_none = jnp.ones((Xs_saves.shape[0], Xs_saves.shape[1])) # or K+1
    if mute_cells is None:
        mute_cells = mute_cells_none

    # Calculate the signal at each time point
    calc_D_2 = jax.vmap(calculate_distances, in_axes=(0, None))
    D_2 = calc_D_2(Xs_saves, XIs)
    gammas_eff = gamma[None, :] + (np.exp(-beta*D_2)*mute_cells[..., None]).sum(axis=1) # aggregated

    return gammas_eff


def calculate_energy_over_cells_and_time(cells, gammas_eff, XIs, beta, gamma,
                                         mute_cells=None, probe_factor=1E-5):
    
    calc_grads_and_Es_all_cells = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_cells_all_t = jax.vmap(calc_grads_and_Es_all_cells, in_axes=(0, None, None, 0, None))
    grads, Es = calc_grads_and_Es_all_cells_all_t(cells[:, :, None, :], XIs, beta,
                                                        gammas_eff, jnp.asarray([probe_factor]))
    
    return (grads, -Es)



def calculate_energy_over_probes_and_time(probes, gammas_eff, XIs, beta, gamma, time_batch_size=10, probe_factor=1E-5):
    calc_grads_and_Es_all_probes = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_probes_many_t = jax.vmap(calc_grads_and_Es_all_probes, in_axes=(None, None, None, 0, None))

    
    grads, Es = calc_grads_and_Es_all_probes_many_t(probes, XIs, beta,
                                                    gammas_eff, jnp.asarray([probe_factor]))

    return (grads, -Es)



size = 400
x = jnp.linspace(-2.6, 2.6, size)
Ms_probes_x, Ms_probes_y = jnp.meshgrid(x, x.copy())
Ms_probes = jnp.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
Ms_probes = Ms_probes[:, None, :]


print("Calculating signals")
gammas_eff = calculate_signals_all(Xs_saves, XIs, beta, gamma, mute_cells=mute_cells_ts)


print("Calculating probe energy")
grads_probes, Es_probes = calculate_energy_over_probes_and_time(Ms_probes, gammas_eff, XIs, beta, gamma)

print("Calculating cell energy")
_, Es_cells = calculate_energy_over_cells_and_time(Xs_saves, gammas_eff, XIs, beta, gamma,
                                                   mute_cells=mute_cells_ts)

Es_min = np.min(Es_probes, axis=-1)[:, None]
Es_max = np.max(Es_probes, axis=-1)[:, None] 


Es_f_norm_cells = (Es_cells - Es_min)/(Es_max-Es_min)
Es_f_norm_manifold = (Es_probes - Es_min)/(Es_max-Es_min)

print(np.max(Es_f_norm_cells), np.min(Es_f_norm_cells))
print(np.max(Es_f_norm_manifold), np.min(Es_f_norm_manifold))

jnp.savez(f"{filename}_energy.npz", manifold_grads=grads_probes, Es_f_norm_manifold=Es_f_norm_manifold, Es_f_norm_cells=Es_f_norm_cells, x=x, y=x.copy())
