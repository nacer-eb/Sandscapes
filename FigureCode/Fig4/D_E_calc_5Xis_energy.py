import jax
jax.config.update("jax_enable_x64", True)

# if gpu runs out of memory
#jax.config.update("jax_platform_name", "cpu") 

import sys
sys.path.append('../../common') # include the common folder

import tqdm

from sandscapes_main import *

import jax.numpy as jnp
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default="run_save_growth_5XIs")
args = parser.parse_args()

filename = args.filename 
   
data = jnp.load(f"{filename}.npz")
Xs_saves = data['Xs_saves'][::2, :]
mute_cells_ts = data["mute_cells_ts"][::2, :]
K = data['K']
XIs = data['XIs']
beta = data['beta']
gamma_0 = data['gamma']
gamma = jnp.asarray([gamma_0]*len(XIs))

tmax = len(Xs_saves)
a = data['a']



def calculate_signals_all(Xs_saves, XIs, beta, gamma, mute_cells, batch_size=100):
    
    # Calculate the signal at each time point
    calc_D_2 = jax.vmap(calculate_distances, in_axes=(0, None))

    # Fast method, but takes too much ram
    #D_2 = np.zeros((Xs_saves.shape[0], Xs_saves.shape[1], XIs.shape[0]))
    #for i in tqdm.tqdm(range(Xs_saves.shape[0]//batch_size)):
    #    D_2_ = calc_D_2(Xs_saves[i*batch_size: (i+1)*batch_size], XIs)
    #    D_2[i*batch_size:(i+1)*batch_size] = np.asarray(D_2_, copy=True)
    #D_2[-batch_size:] = calc_D_2(Xs_saves[-batch_size:], XIs)   
    #gammas_eff = gamma[None, :] + (np.exp(-beta*D_2)*mute_cells[..., None]).sum(axis=1) # aggregated

    gammas_eff = np.zeros((Xs_saves.shape[0], XIs.shape[0]))
    for i in tqdm.tqdm(range(Xs_saves.shape[0]//batch_size)):
        D_2 = np.asarray(calc_D_2(Xs_saves[i*batch_size: (i+1)*batch_size], XIs), copy=True)
        gammas_eff[i*batch_size:(i+1)*batch_size] = gamma[None, :] +\
            (np.exp(-beta*D_2)*mute_cells[i*batch_size:(i+1)*batch_size, :, None]).sum(axis=1) # aggregated
        
    D_2 = np.asarray(calc_D_2(Xs_saves[-batch_size:], XIs), copy=True)   
    gammas_eff[-batch_size:] = gamma[None, :] + (np.exp(-beta*D_2)*mute_cells[-batch_size:, :, None]).sum(axis=1) # aggregated

    return gammas_eff


def calculate_energy_over_cells_and_time(cells, gammas_eff, XIs, beta, gamma,
                                         mute_cells=None, probe_factor=1E-5, batch_size=100):
    
    calc_grads_and_Es_all_cells = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_cells_all_t = jax.vmap(calc_grads_and_Es_all_cells, in_axes=(0, None, None, 0, None))

    
    Es = np.zeros((cells.shape[0], cells.shape[1]))
    for i in tqdm.tqdm(range(cells.shape[0]//batch_size)):
        _, Es_ = calc_grads_and_Es_all_cells_all_t(cells[i*batch_size:(i+1)*batch_size, :, None, :], XIs, beta,
                                                        gammas_eff[i*batch_size:(i+1)*batch_size], jnp.asarray([probe_factor]))

        Es[i*batch_size:(i+1)*batch_size] = np.asarray(Es_, copy=True)

    _, Es_ = calc_grads_and_Es_all_cells_all_t(cells[-batch_size:, :, None, :], XIs, beta,
                                                        gammas_eff[-batch_size:], jnp.asarray([probe_factor]))
    Es[-batch_size:] = np.asarray(Es_, copy=True)

    
    return -Es



def calculate_energy_over_probes_and_time(probes, gammas_eff, XIs, beta, gamma, probe_factor=1E-5, batch_size=10):
    calc_grads_and_Es_all_probes = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_probes_many_t = jax.vmap(calc_grads_and_Es_all_probes, in_axes=(None, None, None, 0, None))

    grads = np.zeros((gammas_eff.shape[0], probes.shape[0], probes.shape[2])) 
    Es = np.zeros((gammas_eff.shape[0], probes.shape[0]))
    for i in tqdm.tqdm(range(gammas_eff.shape[0]//batch_size)):
        grads_, Es_ = calc_grads_and_Es_all_probes_many_t(probes, XIs, beta,
                                                    gammas_eff[i*batch_size:(i+1)*batch_size], jnp.asarray([probe_factor]))

        grads[i*batch_size:(i+1)*batch_size] = np.asarray(grads_, copy=True)
        Es[i*batch_size:(i+1)*batch_size] = np.asarray(Es_, copy=True)

    grads_, Es_ = calc_grads_and_Es_all_probes_many_t(probes, XIs, beta,
                                                    gammas_eff[-batch_size:], jnp.asarray([probe_factor]))
    grads[-batch_size:] = np.asarray(grads_, copy=True)
    Es[-batch_size:] = np.asarray(Es_, copy=True)

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
Es_cells = calculate_energy_over_cells_and_time(Xs_saves, gammas_eff, XIs, beta, gamma,
                                                   mute_cells=mute_cells_ts)

Es_min = np.min(Es_probes, axis=-1)[:, None]
Es_max = np.max(Es_probes, axis=-1)[:, None] 


Es_f_norm_cells = (Es_cells - Es_min)/(Es_max-Es_min)
Es_f_norm_manifold = (Es_probes - Es_min)/(Es_max-Es_min)

print(np.max(Es_f_norm_cells), np.min(Es_f_norm_cells))
print(np.max(Es_f_norm_manifold), np.min(Es_f_norm_manifold))

jnp.savez(f"{filename}_energy.npz", manifold_grads=grads_probes, Es_f_norm_manifold=Es_f_norm_manifold, Es_f_norm_cells=Es_f_norm_cells, x=x, y=x.copy())
