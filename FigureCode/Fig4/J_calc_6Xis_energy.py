import jax
#jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True) # WARNING !!!!!!! WARNING !!!!!!! # Get rid of this at some point

import sys
sys.path.append('../../highdim/common') # include the common folder

import tqdm

from sigdiff_k_metric_jax_betas_opt_mute_reset import *
import jax
import jax.numpy as jnp
import numpy as np

from more_itertools import chunked


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default="run_save_growth_j_K100K")
args = parser.parse_args()

filename = args.filename # "run_save_growth_j_K200K"
   
#filename="run_save_growth_j_K200K"
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



def calculate_signals_all(Xs_saves, XIs, beta, gamma, mute_cells, batch_size=1):
    mute_cells_none = jnp.ones((Xs_saves.shape[0], Xs_saves.shape[1])) # or K+1
    if mute_cells is None:
        mute_cells = mute_cells_none

    # Calculate the signal at each time point
    calc_D_2 = jax.vmap(calculate_distances, in_axes=(0, None))

    #(Xs_saves[:100], XIs)


    # This doesn't fit in gpu mem
    tmax = Xs_saves.shape[0]
    D_2_cpu = np.zeros((tmax, Xs_saves.shape[1], XIs.shape[0]))
    for c_ in tqdm.tqdm(chunked(np.arange(tmax), batch_size), total=tmax//batch_size):
        c = jnp.asarray(c_, dtype=int)
        D_2 = calc_D_2(Xs_saves[c], XIs)
        D_2_cpu[c] = np.asarray(D_2, copy=True)

    gammas_eff = gamma[None, :] + (np.exp(-beta*D_2_cpu)*mute_cells[..., None]).sum(axis=1) # aggregated
    return gammas_eff


def calculate_energy_over_cells_and_time(cells, gammas_eff, XIs, beta, gamma, mute_cells=None, time_batch_size=10, probe_factor=1E-5):
    calc_grads_and_Es_all_cells = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_cells_all_t = jax.vmap(calc_grads_and_Es_all_cells, in_axes=(0, None, None, 0, None))

    # This doesn't fit in gpu mem
    tmax = gammas_eff.shape[0]
    N_probes = cells.shape[1]
    N_dim = cells.shape[-1]
    grads_cpu = np.zeros((tmax, N_probes, N_dim))
    Es_cpu = np.zeros((tmax, N_probes))

    for c_ in tqdm.tqdm(chunked(np.arange(tmax), time_batch_size), total=tmax//time_batch_size):
        c = jnp.asarray(c_, dtype=int)
        grads, Es = calc_grads_and_Es_all_cells_all_t(cells[c, :, None, :], XIs, beta,
                                                        gammas_eff[c], jnp.asarray([probe_factor]))
        grads_cpu[c] = np.asarray(grads, copy=True)
        Es_cpu[c] = np.asarray(Es, copy=True)
    return (grads_cpu, -Es_cpu)



def calculate_energy_over_probes_and_time(probes, gammas_eff, XIs, beta, gamma, time_batch_size=10, probe_factor=1E-5):
    calc_grads_and_Es_all_probes = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_probes_many_t = jax.vmap(calc_grads_and_Es_all_probes, in_axes=(None, None, None, 0, None))

    # This doesn't fit in gpu mem
    tmax = gammas_eff.shape[0]
    N_probes = probes.shape[0]
    N_dim = probes.shape[-1]
    grads_cpu = np.zeros((tmax, N_probes, N_dim))
    Es_cpu = np.zeros((tmax, N_probes))


    for c_ in tqdm.tqdm(chunked(np.arange(tmax), time_batch_size), total=tmax//time_batch_size):
        c = jnp.asarray(c_, dtype=int)
        grads, Es = calc_grads_and_Es_all_probes_many_t(probes, XIs, beta,
                                                        gammas_eff[c], jnp.asarray([probe_factor]))
        grads_cpu[c] = np.asarray(grads, copy=True)
        Es_cpu[c] = np.asarray(Es, copy=True)

    return (grads_cpu, -Es_cpu)



size = 400
x = jnp.linspace(-2.6, 2.6, size)
Ms_probes_x, Ms_probes_y = jnp.meshgrid(x, x.copy())
Ms_probes = jnp.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
Ms_probes = Ms_probes[:, None, :]



gammas_eff = calculate_signals_all(Xs_saves, XIs, beta, gamma, mute_cells=mute_cells_ts, batch_size=100)

grads_probes_, Es_probes_ = calculate_energy_over_probes_and_time(Ms_probes, gammas_eff, XIs, beta, gamma,
                                                                time_batch_size=1)

grads_probes, Es_probes = np.asarray(grads_probes_, copy=True), np.asarray(Es_probes_, copy=True)

_, Es_cells_ = calculate_energy_over_cells_and_time(Xs_saves, gammas_eff, XIs, beta, gamma,
                                                   mute_cells=mute_cells_ts, time_batch_size=1)
Es_cells = np.asarray(Es_cells_, copy=True)

Es_min = np.min(Es_probes, axis=-1)[:, None]
Es_max = np.max(Es_probes, axis=-1)[:, None] 


Es_f_norm_cells = (Es_cells - Es_min)/(Es_max-Es_min)
Es_f_norm_manifold = (Es_probes - Es_min)/(Es_max-Es_min)

print(np.max(Es_f_norm_cells), np.min(Es_f_norm_cells))
print(np.max(Es_f_norm_manifold), np.min(Es_f_norm_manifold))

jnp.savez(f"{filename}_energy.npz", manifold_grads=grads_probes, Es_f_norm_manifold=Es_f_norm_manifold, Es_f_norm_cells=Es_f_norm_cells, x=x, y=x.copy())
