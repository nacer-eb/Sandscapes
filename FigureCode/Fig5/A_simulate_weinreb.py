import os

import sys
sys.path.append("../../common")

import tqdm

import jax 
import jax.numpy as jnp
from sandscapes_main import *
from utils import *

import numpy as np
import scipy.spatial.distance as distance
import vedo

import numpy as np

import matplotlib.pyplot as plt

import argparse
import vedo

import jax
import jax.numpy as jnp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cell simulation parameter sweep.")

    parser.add_argument("--run_id", type=int, required=False, default="0", help="Run id (0, 1, ..) ")
    parser.add_argument("--beta", type=float, required=False, default="0.025", help="Beta parameter for the run (0.005) ")
    parser.add_argument("--K", type=int, required=False, default="500000", help="Number of cells (1000)")
    parser.add_argument("--a", type=int, required=False, default="50", help="Growth timescale - Simulation time = K//a")
    parser.add_argument("--noise_level", type=float, required=False, default="2.0", help="Noise level for the run (2.0)")
    parser.add_argument("--datafile", type=str, required=False,
                        default="../../datasets/weinreb/data_preprocessed_all_t_markers.npz", help="Path to datafile")
    parser.add_argument("--output_dir", type=str, required=False, default="data_weinreb/", help="Where to save the files (test_dir)")   
    args = parser.parse_args()

    run_id = args.run_id
    betas = jnp.asarray([args.beta])
    noise_level = args.noise_level
    output_dir = args.output_dir; os.makedirs(output_dir, exist_ok=True)
    gamma = 1E-5

    
    K = args.K #1000 #5000;  #for testing reduced by 10
    a = args.a
    tmax = K//a #10000;
    print(f"Starting with betas {betas}, noise_level {noise_level}, and output_dir {output_dir}")
    
    
    ## ---- LOADING XI --------- #
    datafile = args.datafile
    data = np.load(datafile, allow_pickle=True)

    ss=1
    XI = data['XI_pca'][::ss]
    XI_NCA = data['XI_NCA'][::ss]
    Ls = data['Ls'][::ss]
    
    ct_names = data['ct_names']
    cmap = data['cmap'].item()
    NCA_mtx = data['NCA_mtx'].T
    
    stem_mask = ~np.isin(Ls, [0])

    seed = 42
    key_0 = jax.random.PRNGKey(seed)
    keys = jax.random.split(key_0, (tmax,))

    mask = (Ls==0)
    X_STEM = np.repeat(np.mean(XI[mask], axis=0)[None, :], K, axis=0) #jnp.zeros((K, XI.shape[-1])) 
    
    # simulate and compute one chunk of time at a time (we are memory limited)
    n_blocks=400 #
    print(tmax/n_blocks, "debug blocks")
    tmax_blocks = tmax//n_blocks
   
    X_0 = jnp.zeros((K, XI.shape[-1])) + X_STEM

    print("Generating save")
    Ms_saves_cpu_nca = np.zeros((tmax, K, 3)); 
    print("Ms_saves_cpu_nca size", sizeof_fmt(Ms_saves_cpu_nca.nbytes), Ms_saves_cpu_nca.shape)

    t_saves = np.concatenate((np.arange(-2000, 0, 250), np.asarray([-1])), axis=0)
    t_saves_abs = t_saves % tmax # Translate negative indices into absolute step indices for the full simulation
    print(f"The absolute saves are {t_saves_abs}")
    
    Ms_saves_cpu_detail = np.zeros((len(t_saves), K, XI.shape[-1]));
    print("Ms_saves_cpu_detail size", sizeof_fmt(Ms_saves_cpu_detail.nbytes), Ms_saves_cpu_detail.shape)
    for i in tqdm.tqdm(range(n_blocks)):
        keys_block = keys[i*tmax_blocks: (i+1)*tmax_blocks]

        t_offset = i*tmax_blocks
        Ms_saves_ = training_loop_linear_growth(X_0, X_STEM, XI[stem_mask], betas, gamma, tmax_blocks,
                                                lr=0.03*2, noise_level=noise_level, keys=keys_block, growth_rate=a,
                                                t_offset=t_offset)

        # I have to do this because I'm subsampling
        start_idx = i * tmax_blocks
        end_idx = (i + 1) * tmax_blocks
        for save_i, abs_t in enumerate(t_saves_abs):
            if start_idx <= abs_t < end_idx: # this checks if this is the right window to look
                local_idx = abs_t - start_idx
                Ms_saves_cpu_detail[save_i] = np.asarray(Ms_saves_[local_idx], copy=True)

        X_0 = jnp.asarray(Ms_saves_[-1], copy=True)
        
        Ms_saves_ = Ms_saves_
        Ms_saves_cpu_nca[i*(tmax_blocks): (i+1)*(tmax_blocks)] = np.asarray(Ms_saves_@NCA_mtx)
    
    runname = f"{output_dir}/worker_{run_id}" 

    print("SAVING...")
    dt = 4
    np.savez(runname+".npz", Ms_saves_cpu_nca=Ms_saves_cpu_nca[::dt],
             gamma=gamma, betas=betas, seed=seed,
             noise_level=noise_level, X_STEM=X_STEM, t_saves=t_saves, args_dict=vars(args), dt=dt)
    
    np.savez(runname+"_detail.npz", Ms_saves_cpu_detail=Ms_saves_cpu_detail,
             gamma=gamma, betas=betas, seed=seed,
             noise_level=noise_level, X_STEM=X_STEM, t_saves=t_saves, args_dict=vars(args))
    print("SAVED...")


