import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import jax.numpy as jnp
import vedo

sys.path.append("../../common")
from sandscapes_main import calculate_gradient

datafile = "../../datasets/weinreb/data_preprocessed_all_t_markers.npz"
runname = "/localhome/nacer/data_weinreb_26092026//worker_0_detail.npz"
runname_nca = "/localhome/nacer/data_weinreb_26092026//worker_0.npz"

data = np.load(datafile, allow_pickle=True)
Ls = data['Ls']
ct_names = data['ct_names']

data_run = np.load(runname, allow_pickle=True)

K, a = int(data_run["args_dict"].item()["K"]), int(data_run["args_dict"].item()["a"])
tmax = K//a
t_saves = data_run["t_saves"]
beta = jnp.asarray(data_run["betas"])
gamma = jnp.asarray(data_run["gamma"])


Ms_saves_cpu_nca = np.load(runname_nca, mmap_mode='r')['Ms_saves_cpu_nca']

noise_level = float(data_run["noise_level"])
lr = 0.03 * 2                                        # from A_simulate_weinreb.py
noise_mag = (10.0 ** -noise_level) * np.sqrt(data['XI_pca'].shape[1])  # per-step noise norm
print(f"noise {noise_level=}, sigma/sqrt(dim), per-step |noise| = {noise_mag:.4f}, cmap max 2x = {2*noise_mag:.4f}")


stem_mask = ~np.isin(Ls, [0])
XI_stem = jnp.asarray(data['XI_pca'][stem_mask], dtype=jnp.float32)
w_mu = jnp.ones(len(XI_stem))

NCA_mtx = data['NCA_mtx'].T

os.makedirs("Fig", exist_ok=True)
for snap_i in range(len(t_saves)): #len(t_saves)
    t_slice = int(t_saves[snap_i] % tmax)
    print(t_slice, tmax)
    mute_cells = (np.arange(K) <= a * t_slice)

    Ms_saves_m1k = data_run["Ms_saves_cpu_detail"][snap_i]
    mute_cells = mute_cells

    grads = np.asarray(calculate_gradient(jnp.asarray(Ms_saves_m1k, dtype=jnp.float32),
                                              XI_stem, beta, gamma, mute_cells, w_mu))
    grad_mag = np.sqrt(np.linalg.norm(grads, axis=-1))
    grad_mag[~mute_cells] = 0.0
    
    X_nca = Ms_saves_m1k@NCA_mtx

    fig = vedo.Plotter(axes=0, size=(3840, 2160), interactive=False, offscreen=True)

    vmin, vmax = np.percentile(grad_mag[mute_cells], [4, 97])#[5, 95]   # pick your band
    print(np.percentile(grad_mag[mute_cells], [5, 20, 50, 75, 95]), "Roughly noise magnitudes")
    
    cells_active = vedo.Points(X_nca[mute_cells][50*(t_slice-5000):], r=7, alpha=1.0)
    cells_active.cmap('viridis', grad_mag[mute_cells][50*(t_slice-5000):], vmin=vmin, vmax=vmax)
    
    cam = dict(
        pos=(89.9062, 21.2696, -50.2680),
        focal_point=(8.34769, -3.25451, -16.7184),
        viewup=(-0.367635, -0.0479114, -0.928735),
        roll=92.8505,
        distance=91.5358,
        clipping_range=(0.304213, 304.213),
    )

    fig.show(cells_active, axes=0, camera=cam)
    os.makedirs("Fig_DEBUG", exist_ok=True)
    fig.screenshot(f"Fig_DEBUG/M_full_tree_grad_t{t_slice}.png")
    fig.close()

    print(f"saved Fig_DEBUG/M_full_tree_grad_t{t_slice}.png")
