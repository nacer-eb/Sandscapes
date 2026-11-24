import sys
sys.path.append('../../common')

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import jax
import jax.numpy as jnp

from sandscapes_main import *

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
    S = calculate_signals(beta, D_2)
    S_sum = calculate_aggregate_signals(S, mute_cells) + gamma
    return S_sum

calculate_signals_all_t = jax.vmap(calculate_signals_t, in_axes=(0, None, None, None, 0))

xi_x = 1.0
xi_y = 1.3

XIs = jnp.asarray([ [-xi_x, -xi_y], [xi_x, -xi_y], [0, xi_y] ])
q = args.q

w_mu = jnp.asarray([600.0, 60.0, 1.0])

gamma = jnp.asarray([800.0, 150.0, 1.0])*q + 1E-5

gamma = gamma / w_mu

beta = 1.5 
K = 1000*10*4  # 4

tmax= K*30  # here it's inverse rate
a = (tmax)/K

key = jax.random.key(args.seed)
keys = jax.random.split(key, (tmax,))

X_STEM = jnp.asarray([ XIs[0] ])
X_0 = X_STEM.repeat(K, axis=0)

N_TIME_CHUNKS = 100
time_chunk_size = tmax // N_TIME_CHUNKS

k = jnp.arange(K)[None, :]
dt = 20

from numpy.lib.format import open_memmap
Xs_saves = open_memmap(
    "Xs_saves.mmap", 
    mode="w+", 
    dtype=np.float32, 
    shape=(tmax // dt, K, 2)
)
#Xs_saves = np.zeros((tmax//dt, K, 2), dtype=np.float32)
Ss_saves = np.zeros((tmax//dt, 3))

k_indices = np.arange(K)
idx_start = np.ceil(k_indices * a).astype(int)
for i in tqdm.tqdm(range(0, N_TIME_CHUNKS)):
    t0 = i * time_chunk_size
    t1 = (i + 1) * time_chunk_size
    t_chunk = jnp.arange(t0, t1)[:, None]
    mute_cells_ts_chunk = k <= t_chunk / a

    keys_chunk = keys[i * time_chunk_size: (i+1) * time_chunk_size]
    Xs_saves_ = training_loop(X_0, X_STEM, XIs, betas=jnp.asarray([beta]), gamma=gamma, tmax=time_chunk_size, lr=0.02, noise_level=4.0,
                              keys=keys_chunk, mute_cells_ts=mute_cells_ts_chunk, w_mu=w_mu)
    X_0 = jnp.copy(Xs_saves_[-1])

    #Xs_saves[i * time_chunk_size//dt: (i+1) * time_chunk_size//dt] = np.asarray(Xs_saves_[::dt], copy=True)
    Xs_saves[i * time_chunk_size//dt: (i+1) * time_chunk_size//dt] = np.asarray(Xs_saves_[::dt], copy=True)
    Ss_saves[i * time_chunk_size//dt: (i+1) * time_chunk_size//dt] = np.asarray(calculate_signals_all_t(Xs_saves_[::dt], XIs,
                                                                                                        beta, gamma,
                                                                                                        mute_cells_ts_chunk[::dt]),
                                                                                copy=True)
Xs_saves.flush()

# Let's not save Xs_saves...
jnp.savez(args.filename, Ss_saves=Ss_saves, K=K, XIs=XIs, beta=beta, gamma=gamma)