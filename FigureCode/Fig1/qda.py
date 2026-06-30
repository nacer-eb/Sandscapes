import jax
#jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True) # WARNING !!!!!!! WARNING !!!!!!! # Get rid of this at some point

import sys
sys.path.append('../../highdim/common') # include the common folder

from sigdiff_k_metric_jax_betas_opt_mute_reset import *

import seaborn as sns
sns.set_theme(palette="pastel")

import matplotlib.pyplot as plt
import scipy.spatial.distance as distance

import numpy as np

data = jnp.load("run_save.npz")
Xs_saves = data['Xs_saves']
K = data['K']
XIs = data['XIs']
beta = data['beta']
gamma = data['gamma']

tmax = len(Xs_saves)

mute_cells = jnp.ones((Xs_saves.shape[0], Xs_saves.shape[1]))

D_2 = jax.vmap(calculate_distances, in_axes=(0, None))(Xs_saves, XIs)
gammas_eff = gamma[None, :] + jax.vmap(calculate_signals, in_axes=(None, 0, 0))(beta, D_2, mute_cells).sum(axis=1) # aggregated

z = gammas_eff/gammas_eff[:, -1][:, None]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection='3d')

ts = np.linspace(0, 1, tmax)
ax.scatter(z[:, 0], z[:, 1], ts)
plt.show()
