import sys
sys.path.append('../../common')

import os

import jax
import jax.numpy as jnp

from sandscapes_main import *
from utils import *

import numpy as np
import scipy.spatial.distance as distance
import tqdm


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=41)
parser.add_argument('--K', type=int, default=200000)
parser.add_argument('--noise', type=float, default=2.75)
parser.add_argument('--filename', type=str, default="run_save_growth_5XIs")
args = parser.parse_args()

XIs = jnp.asarray([ [-1.75, -1.0],
                    [1.0, -1.5],
                    [0, 1.0],
                    [2.0, 1.0],
                    [0.9, 3.0]])


XIs -= jnp.mean(XIs, axis=0)
gamma = 1E-2 #5
beta = 1.0 
K = args.K

a = 3
tmax= K//a
#a = K/tmax

key = jax.random.key(args.seed)
keys = jax.random.split(key, (tmax,))

X_STEM = jnp.asarray([ XIs[0] ])
X_0 = X_STEM.repeat(K, axis=0)

N_TIME_CHUNKS = 200
time_chunk_size = tmax // N_TIME_CHUNKS
k = jnp.arange(K)[None, :]
Xs_saves = np.zeros((tmax, K, 2), dtype=np.float32)


dt = 10
mute_cells_ts = (np.arange(K, dtype=int)[None, :] <= a * np.arange(tmax, dtype=int)[:, None][::dt]) # for the energy calc
print(sizeof_fmt(mute_cells_ts.nbytes))

for i in tqdm.tqdm(range(0, N_TIME_CHUNKS)):
    t0 = i * time_chunk_size
    t1 = (i + 1) * time_chunk_size

    keys_chunk = keys[i * time_chunk_size: (i+1) * time_chunk_size]
    Xs_saves_ = training_loop_linear_growth(X_0, X_STEM, XIs, jnp.asarray([beta]),
                                            gamma, time_chunk_size, 0.02, noise_level=args.noise,
                                            keys=keys_chunk, growth_rate=a, t_offset=t0)    
    X_0 = jnp.copy(Xs_saves_[-1])

    Xs_saves[i * time_chunk_size: (i+1) * time_chunk_size] = np.asarray(Xs_saves_, copy=True)
jnp.savez(args.filename+".npz", Xs_saves=Xs_saves[::dt], K=K, XIs=XIs, beta=beta, gamma=gamma, a=a, mute_cells_ts=mute_cells_ts)
