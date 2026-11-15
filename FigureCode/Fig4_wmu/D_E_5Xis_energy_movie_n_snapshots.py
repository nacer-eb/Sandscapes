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
from D_E_plotting_utils import *

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="run_save_growth_5XIs")
    args = parser.parse_args()

    dt = 1 # 20
    # Loading cell position
    filename = args.filename # "run_save_growth_j_K200K"
    data = np.load(f"{filename}.npz")
    Xs_saves = np.asarray(data['Xs_saves'])[::2, :][::dt]
    mute_cells_ts = np.asarray(data['mute_cells_ts'], dtype=bool)[::2, :][::dt]
    XIs = data['XIs']
    K = Xs_saves.shape[1]; beta = np.asarray(data['beta']); gamma = np.asarray(data['gamma'])
    tmax = len(Xs_saves)-1
    
    # Load precomputed energies
    data = np.load(f"{filename}_energy.npz")
    x = data['x']; y = data['y']; size=len(x)
    Xs_probes_x, Xs_probes_y = np.meshgrid(x, y)
    Xs_probes = np.asarray([Xs_probes_x.ravel(), Xs_probes_y.ravel()]).T

    Es_manifold = data['Es_f_norm_manifold'][::dt]
    manifold_grads = data['manifold_grads'][::dt]
    Es_cells = data['Es_f_norm_cells'][::dt]


    # Precompute signaling bakcground color
    ## Colors that pop more
    XIs_colors = np.array([
        [120, 190, 120],   # greener, slightly darker
        [ 90, 170, 230],   # more saturated blue
        [240, 150, 170],   # stronger pink
        [240, 175, 100],
        [175, 140, 215]
    ]) / 255.0

    # E - manifold color
    coefs = np.exp(-beta*distance.cdist(Xs_probes, XIs)**2)
    coefs = coefs/coefs.sum(axis=-1)[:, None]
    manifold_colors = np.ones((len(Xs_probes), 4))
    manifold_colors = coefs@XIs_colors
    
    ## Starting multiprocessing figure
    n_processes =  os.cpu_count()//2 # REDUCED
    print(f"Rendering 2d frames using", n_processes, "CPU cores...")

    dpi = 50
    tasks = [
        (Xs_probes.reshape(size, size, -1), manifold_grads[t].reshape(size, size, -1), manifold_colors.reshape(size, size, -1), dpi)
        #(Xs_probes.reshape(size, size, -1), Es_manifold[t].reshape(size, size), dpi)
        for t in range(tmax)
    ]
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        rgba_mpls = list(tqdm.tqdm(executor.map(generate_rgba_mpl, tasks), total=len(tasks)))
    rgba_mpls = np.asarray(rgba_mpls)

    print(f"Rendering 2d frames using", n_processes, "CPU cores...")
    tasks = [        
        (Xs_probes.reshape(size, size, -1), Es_manifold[t].reshape(size, size), dpi)
        for t in range(tmax)
    ]
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        rgba_mpls_contour = list(tqdm.tqdm(executor.map(generate_rgba_mpl_contour, tasks), total=len(tasks)))
    rgba_mpls = np.asarray(rgba_mpls)


    print("Generating snapshots")
    #"""
    dirname = "D_E_snapshots"
    tasks = [
        (t, Xs_saves[t], Es_cells[t], mute_cells_ts[t], Es_manifold[t], manifold_colors, rgba_mpls[t], rgba_mpls_contour[t], x, y, size, dirname, True)
        for t in [298, 315, 348, 367, 1975, 2009, 2040, 2853] #2593
    ]
    
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)

    #n_processes = 1
    contxt = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_processes, mp_context=contxt) as executor:
        list(tqdm.tqdm(executor.map(generate_3d_frames, tasks), total=len(tasks)))
    #"""


    print("Generating movie")
    dirname = "temp_frames"
    tasks = [
        (t, Xs_saves[t], Es_cells[t], mute_cells_ts[t], Es_manifold[t], manifold_colors, rgba_mpls[t], rgba_mpls_contour[t], x, y, size, dirname, False)
        for t in range(tmax)
    ]
    
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)

    #n_processes = 1
    contxt = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_processes, mp_context=contxt) as executor:
        list(tqdm.tqdm(executor.map(generate_3d_frames, tasks), total=len(tasks)))


    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "30",
        "-i", f"{dirname}/frame_%07d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        f"{filename}_3d.mp4"
    ]
    subprocess.run(cmd, check=True)
    print("Saved to", filename)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

