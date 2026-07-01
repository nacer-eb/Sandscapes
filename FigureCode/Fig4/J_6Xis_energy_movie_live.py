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
from plotting_utils_v2 import *

if __name__ == "__main__":   
    dt = 10 # 20
    # Loading cell position
    filename="run_save_growth_j_K100K"
    data = np.load(f"{filename}.npz")
    Xs_saves = np.asarray(data['Xs_saves'])[::dt, :]
    mute_cells_ts = np.asarray(data['mute_cells_ts'], dtype=bool)[::dt, :]
    XIs = data['XIs']
    K = Xs_saves.shape[1]; beta = np.asarray(data['beta']); gamma = np.asarray(data['gamma'])
    tmax = len(Xs_saves)-1
    
    # Load precomputed energies
    data = np.load(f"{filename}_energy.npz")
    x = data['x']; y = data['y']; size=len(x)
    Xs_probes_x, Xs_probes_y = np.meshgrid(x, y)
    Xs_probes = np.asarray([Xs_probes_x.ravel(), Xs_probes_y.ravel()]).T

    Es_manifold = data['Es_f_norm_manifold'][::dt]; Es_cells = data['Es_f_norm_cells'][::dt]


    # Precompute signaling bakcground color
    ## Colors that pop more
    XIs_colors = np.array([
        [120, 190, 120],   # greener, slightly darker
        [ 90, 170, 230],   # more saturated blue
        [240, 150, 170],   # stronger pink
        [ 90, 170, 230],   # more saturated blue
        [ 40, 50, 170],   # stronger pink
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
        #(Xs_probes.reshape(size, size, -1), manifold_grads[t].reshape(size, size, -1), dpi)
        (Xs_probes.reshape(size, size, -1), Es_manifold[t].reshape(size, size), dpi)
        for t in range(tmax)
    ]
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        rgba_mpls = list(tqdm.tqdm(executor.map(generate_rgba_mpl, tasks), total=len(tasks)))
    rgba_mpls = np.asarray(rgba_mpls)


    dirname = "temp_frames"
    tasks = [
        (t, Xs_saves[t], Es_cells[t], mute_cells_ts[t], Es_manifold[t], manifold_colors, rgba_mpls[t], x, y, size, dirname)
        for t in range(tmax)
    ]
    
    t, Xs_saves_t, Es_cells_t, mute_cells_t, Es_manifold_t, manifold_colors, rgba_mpls_t, x, y, size, dirname = args
    z_factor = 2.0
    z_shift = 0.0
    #
    dz = 0.135

    offscreen= True # False
    fig = vedo.Plotter(axes=1, size=(1920,1080), interactive=False, offscreen=True, bg=None)

    # Plotting cells
    Xs_probes_3d = np.column_stack((Xs_saves_t, (Es_cells_t + z_shift) * z_factor))
    Xs_probes_3d = Xs_probes_3d[mute_cells_t]
    cells = vedo.Points(Xs_probes_3d, r=25, c=np.asarray([0.97, 0.97, 0.95])).lighting('shiny')


    # Grid
    grid = vedo.Grid(s=[x, y]).linewidth(0).lighting('plastic')
    grid.points[:, 2] = (Es_manifold_t + z_shift) * z_factor

    
    grid.alpha(0.8)  # instead of per-point alpha in the array
    grid = vedo_grid_set_rgb(grid, bake_shadows_xy(manifold_colors, grid, Xs_probes_3d, radius=0.09, strength=0.35))

    ##
    sn = grid.point_normals
    ids = np.asarray([grid.closest_point(p, return_point_id=True) for p in cells.points])
    pts = cells.points
    pts += dz * sn[ids]                     # push along normal
    cells.points = pts

    grid_streamplot = vedo.Grid(s=[x, y], c="red").linewidth(0)#.lighting('plastic')
    grid_streamplot.alpha(1.0)
    grid_streamplot = vedo_grid_set_rgb(grid_streamplot, rgba_mpls_t.reshape(size*size, -1)[:, :-1])
    origin_dot = vedo.Point([np.min(x), np.min(y), 0], c=(0.335, 0.335, 0.335), r=15.12)  # r is the radius in screen pixels

    cam, axes = get_cam_n_axes(grid)
    for part in axes.unpack():
        if "z" in part.name.lower() and t>0:
            part.off()
    
    fig.show(grid, grid_streamplot, cells, origin_dot, camera=cam, axes=axes)
    fig.close()



