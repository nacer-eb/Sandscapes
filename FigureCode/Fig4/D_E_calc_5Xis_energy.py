import jax
jax.config.update("jax_enable_x64", True)

# if gpu runs out of memory
#jax.config.update("jax_platform_name", "cpu") 

import sys
sys.path.append('../../common') # include the common folder

import tqdm

from sandscapes_main import *
from sandscapes_energy_utils import *

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
