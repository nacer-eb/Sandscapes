import sys
sys.path.append('../../highdim/common')

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import argparse
import jax
import jax.numpy as jnp
from sigdiff_k_metric_jax_betas_opt_mute_reset import *
import numpy as np
import scipy.spatial.distance as distance
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--q', type=float, default=4.0)
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--filename', type=str, default="run_save_growth.npz")
args = parser.parse_args()

def calculate_signals_t(X, XI, beta, gamma, mute_cells):
    D_2 = calculate_distances(X, XI)
    S = calculate_signals(beta, D_2, mute_cells)
    S_sum = calculate_aggregate_signals(S) + gamma
    return S_sum

calculate_signals_all_t = jax.vmap(calculate_signals_t, in_axes=(0, None, None, None, 0))

xi_x = 1.0
xi_y = 1.3

XIs = jnp.asarray([ [-xi_x, -xi_y], [xi_x, -xi_y], [0, xi_y] ])
q = args.q

gamma = jnp.asarray([800.0, 150.0, 1.0])*q + 1E-5
beta = 1.5 
K = 2000*10*4
tmax= K*30 

a = (tmax)/K

key = jax.random.key(args.seed)
keys = jax.random.split(key, (tmax,))

X_STEM = jnp.asarray([ XIs[0] ])
X_0 = X_STEM.repeat(K, axis=0)

N_TIME_CHUNKS = 800
time_chunk_size = tmax // N_TIME_CHUNKS

k = jnp.arange(K)[None, :]
dt = 20
Xs_saves = np.zeros((tmax//dt, K, 2), dtype=np.float32)
Ss_saves = np.zeros((tmax//dt, 3))

for i in tqdm.tqdm(range(0, N_TIME_CHUNKS)):
    t0 = i * time_chunk_size
    t1 = (i + 1) * time_chunk_size
    t_chunk = jnp.arange(t0, t1)[:, None]
    mute_cells_ts_chunk = k <= t_chunk / a

    keys_chunk = keys[i * time_chunk_size: (i+1) * time_chunk_size]
    Xs_saves_ = training_loop(X_0, X_STEM, XIs, betas=jnp.asarray([beta]), gamma=gamma, tmax=time_chunk_size, lr=0.02, noise_level=4.0,
                              keys=keys_chunk, mute_cells_ts=mute_cells_ts_chunk)
    X_0 = jnp.copy(Xs_saves_[-1])

    Xs_saves[i * time_chunk_size//dt: (i+1) * time_chunk_size//dt] = np.asarray(Xs_saves_[::dt], copy=True)
    Ss_saves[i * time_chunk_size//dt: (i+1) * time_chunk_size//dt] = np.asarray(calculate_signals_all_t(Xs_saves_[::dt], XIs,
                                                                                                        beta, gamma,
                                                                                                        mute_cells_ts_chunk[::dt]),
                                                                                copy=True)

jnp.savez(args.filename, Ss_saves=Ss_saves, Xs_saves=Xs_saves, K=K, XIs=XIs, beta=beta, gamma=gamma)

