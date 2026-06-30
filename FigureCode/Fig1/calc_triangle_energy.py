import jax
#jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True) # WARNING !!!!!!! WARNING !!!!!!! # Get rid of this at some point

import sys
sys.path.append('../../common') # include the common folder

from sandscapes_main import *
import jax
import jax.numpy as jnp

data = jnp.load("run_save.npz")
Xs_saves = data['Xs_saves']
K = data['K']
XIs = data['XIs']
beta = data['beta']
gamma = data['gamma']
tmax = len(Xs_saves)


def calculate_energy_over_cells_and_time(cells, Xs_saves, XIs, beta, gamma, time_batch_size=10, probe_factor=1E-5):   
    # Calculate the signal at each time point
    D_2 = jax.vmap(calculate_distances, in_axes=(0, None))(Xs_saves, XIs)
    gammas_eff = gamma[None, :] + jax.vmap(calculate_signals, in_axes=(None, 0))(beta, D_2).sum(axis=1) # aggregated

    calc_grads_and_Es_all_cells = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))

    def calc_grads_and_Es_all_cells_prefilled(input_tuple):
        cells_slice, gammas_slice = input_tuple
        return calc_grads_and_Es_all_cells(cells_slice[:, None, :], XIs, beta, gammas_slice, jnp.asarray([probe_factor]))
    
    grads, Es = jax.lax.map(calc_grads_and_Es_all_cells_prefilled, (Xs_saves, gammas_eff))
    return (grads, -Es)


def calculate_energy_over_probes_and_time(probes, Xs_saves, XIs, beta, time_batch_size=10, probe_factor=1E-5):
    # Calculate the signal at each time point
    D_2 = jax.vmap(calculate_distances, in_axes=(0, None))(Xs_saves, XIs)
    gammas_eff = gamma[None, :] + jax.vmap(calculate_signals, in_axes=(None, 0))(beta, D_2).sum(axis=1) # aggregated

    calc_grads_and_Es_all_probes = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_probes_prefilled = lambda gamma_slice: calc_grads_and_Es_all_probes(probes, XIs,
                                                                                              beta, gamma_slice,
                                                                                              jnp.asarray([probe_factor]))
    grads, Es = jax.lax.map(calc_grads_and_Es_all_probes_prefilled, gammas_eff, batch_size=time_batch_size)

    return (grads, -Es)


size = 400
x = jnp.linspace(-1.8, 1.8, size)
Ms_probes_x, Ms_probes_y = jnp.meshgrid(x, x.copy())
Ms_probes = jnp.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
Ms_probes = Ms_probes[:, None, :]

grads_probes, Es_probes = calculate_energy_over_probes_and_time(Ms_probes, Xs_saves, XIs, beta, time_batch_size=5)
_, Es_cells = calculate_energy_over_cells_and_time(Xs_saves, Xs_saves, XIs, beta, gamma, time_batch_size=5)


Es_min = jnp.min(Es_probes, axis=-1)[:, None]
Es_max = jnp.max(Es_probes, axis=-1)[:, None]


Es_f_norm_cells = (Es_cells - Es_min)/(Es_max-Es_min)
Es_f_norm_manifold = (Es_probes - Es_min)/(Es_max-Es_min)

print(jnp.max(Es_f_norm_cells), jnp.min(Es_f_norm_cells))
print(jnp.max(Es_f_norm_manifold), jnp.min(Es_f_norm_manifold))

jnp.savez("energy_save.npz", Es_f_norm_manifold=Es_f_norm_manifold, Es_f_norm_cells=Es_f_norm_cells,
          manifold_grads=grads_probes, x=x, y=x.copy())
