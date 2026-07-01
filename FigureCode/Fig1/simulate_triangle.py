import sys
sys.path.append('../../common') # include the common folder

import jax
jax.config.update("jax_enable_x64", True) 


from sandscapes_main import *
import jax
import jax.numpy as jnp

# Parameters
xi_x = 1.0
xi_y = 1.3

XIs = jnp.asarray([ [-xi_x, -xi_y], [xi_x, -xi_y], [0, xi_y] ])
gamma = jnp.asarray([0, 0, 0])
beta = 1.5 # 0.625 # 6
K = 500

tmax=1000
key = jax.random.key(41)
keys = jax.random.split(key, (tmax,))

X_STEM = jnp.asarray([ XIs[-1] + 0.1*jnp.asarray([1.0, -1.0]) ])
X_0 = X_STEM.repeat(K, axis=0) + jax.random.normal(keys[1], shape=(K, 2)) * (1.0/10.0)**6


# Simulation 
Xs_saves = training_loop(X_0, X_STEM, XIs, betas=jnp.asarray([beta]), gamma=gamma, tmax=tmax, lr=0.014, noise_level=50,
                         keys=keys, mute_cells_ts=None)
jnp.savez("run_save.npz", Xs_saves=jnp.asarray(Xs_saves, copy=True), K=K, XIs=XIs, beta=beta, gamma=gamma)

