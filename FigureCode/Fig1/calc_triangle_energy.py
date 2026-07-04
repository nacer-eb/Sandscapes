import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append('../../common') # include the common folder

from sandscapes_main import *
from sandscapes_energy_utils import *
import jax
import jax.numpy as jnp

data = jnp.load("run_save.npz")
Xs_saves = data['Xs_saves']
K = data['K']
XIs = data['XIs']
beta = data['beta']
gamma = data['gamma']
tmax = len(Xs_saves)


size = 400
x = jnp.linspace(-1.8, 1.8, size)
Ms_probes_x, Ms_probes_y = jnp.meshgrid(x, x.copy())
Ms_probes = jnp.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T
Ms_probes = Ms_probes[:, None, :]

mute_cells_ts = jnp.ones((Xs_saves.shape[0], Xs_saves.shape[1]))

print("Calculating signals")
gammas_eff = calculate_signals_all(Xs_saves, XIs, beta, gamma, mute_cells=mute_cells_ts)

print("Calculating probe energy")
grads_probes, Es_probes = calculate_energy_over_probes_and_time(Ms_probes, gammas_eff, XIs, beta, gamma)

print("Calculating cell energy")
Es_cells = calculate_energy_over_cells_and_time(Xs_saves, gammas_eff, XIs, beta, gamma,
                                                   mute_cells=mute_cells_ts)


Es_min = jnp.min(Es_probes, axis=-1)[:, None]
Es_max = jnp.max(Es_probes, axis=-1)[:, None]


Es_f_norm_cells = (Es_cells - Es_min)/(Es_max-Es_min)
Es_f_norm_manifold = (Es_probes - Es_min)/(Es_max-Es_min)

print(jnp.max(Es_f_norm_cells), jnp.min(Es_f_norm_cells))
print(jnp.max(Es_f_norm_manifold), jnp.min(Es_f_norm_manifold))

jnp.savez("energy_save.npz", Es_f_norm_manifold=Es_f_norm_manifold, Es_f_norm_cells=Es_f_norm_cells,
          manifold_grads=grads_probes, x=x, y=x.copy())
