import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.spatial.distance as distance
import numpy as np

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import tqdm
# Plotting
from F_plotting_utils import *

if __name__ == "__main__":
    ts = np.asarray([0, 150, 260, 580, 700, 720, 730, 740, 750, 850, 900, 1400], dtype=int)# + 380+50
   
    # Loading cell position
    data = np.load("run_save_perturbation.npz")
    Xs_saves = np.asarray(data['Xs_saves'])[ts]
    mute_cells_ts = np.asarray(data['mute_cells_ts'], dtype=bool)[ts]
    XIs = data['XIs']
    K = Xs_saves.shape[1]; beta = np.asarray(data['beta']); gamma = np.asarray(data['gamma'])
    tmax = len(Xs_saves)
    
    # Load precomputed energies
    data = np.load("energy_save_perturbation.npz")
    x = data['x']; y = data['y']; size=len(x)
    Xs_probes_x, Xs_probes_y = np.meshgrid(x, y)
    Xs_probes = np.asarray([Xs_probes_x.ravel(), Xs_probes_y.ravel()]).T
    Es_manifold = data['Es_f_norm_manifold'][ts]; Es_cells = data['Es_f_norm_cells'][ts]
    

    # Precompute signaling bakcground color
    ## Colors that pop more
    XIs_colors = np.array([
        [120, 190, 120],   # greener, slightly darker
        [ 90, 170, 230],   # more saturated blue
        [240, 150, 170],   # stronger pink
    ]) / 255.0

    # E - manifold color
    coefs = np.exp(-beta*distance.cdist(Xs_probes, XIs)**2)
    coefs = coefs/coefs.sum(axis=-1)[:, None]
    manifold_colors = np.ones((len(Xs_probes), 4))
    manifold_colors = coefs@XIs_colors


    ## Starting multiprocessing figure

    n_processes = os.cpu_count()//2 # REDUCED
    print(f"Rendering 2d frames using", n_processes, "CPU cores...")

     
    
    dpi = 50
    tasks = [
        #(Xs_probes.reshape(size, size, -1), manifold_grads[t].reshape(size, size, -1), dpi)
        (Xs_probes.reshape(size, size, -1), Es_manifold[t].reshape(size, size), dpi)
        for t in range(tmax)
    ]
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        rgba_mpls = list(tqdm.tqdm(executor.map(generate_rgba_mpl, tasks), total=len(tasks)))
    rgba_mpls = np.asarray(rgba_mpls)


    #mute_cells_t = np.ones((len(Es_cells[0])), dtype=bool)
    dirname = "F_energy_snapshots"
    tasks = [
        (t, Xs_saves[t], Es_cells[t], mute_cells_ts[t], Es_manifold[t], manifold_colors, rgba_mpls[t], x, y, size, dirname)
        for t in range(tmax)
    ]
    
    #if os.path.exists(dirname):
    #    shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)

    contxt = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_processes, mp_context=contxt) as executor:
        list(tqdm.tqdm(executor.map(generate_3d_frames, tasks), total=len(tasks)))



