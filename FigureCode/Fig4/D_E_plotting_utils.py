import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import scipy.spatial.distance as distance
import numpy as np

import vedo

def get_cam_n_axes(grid, cam=None):
   
    cam = dict(
        pos=(3.52364, -6.31033, 3.00090),
        focal_point=(-0.185445, -0.0439029, 0.943929),
        viewup=(-0.143162, 0.231149, 0.962328),
        roll=-63.6449,
        parallel_scale=1.70268,
        clipping_range=(1.23751, 15.2291),
    )

    # 6
    cam = dict(
        pos=(2.39710, -9.76592, 8.23801),
        focal_point=(-0.0799949, -0.217072, 0.732162),
        viewup=(-0.108269, 0.596839, 0.795022),
        roll=-20.6079,
        distance=12.3957,
        clipping_range=(2.79741, 23.8211),
    )

    cam = dict(
        pos=(7.73413, 0.926445, 6.83426),
        focal_point=(-0.0370819, 0.0488537, 0.665685),
        viewup=(-0.614358, -0.0783841, 0.785124),
        roll=-94.5133,
        distance=9.96059,
        clipping_range=(1.72326, 1723.26),
        parallel_scale=3.5,  # Add this! Controls the orthographic zoom
    )
    
    axes = vedo.Axes(
        grid,
        show_ticks=True,
        xtitle="g_1",
        ytitle="g_2",
        xtitle_position=0.6,
        ytitle_position=0.6,
        ztitle_position=0.85,
        xtitle_offset=0.05,
        ytitle_offset=0.05,
        xtitle_size=0.074,
        ytitle_size=0.074,
        ztitle_size=0.064,
        ztitle="Energy",
        xlabel_size=0.0,
        ylabel_size=0.0,
        zlabel_size=0.0,
        xlabel_color="black",
        ylabel_color="black",
        zlabel_color="black",
        #zrange=[0, 1.2],
        xyplane_color=(0.535, 0.535, 0.535),
        xyalpha=1.0,
        axes_linewidth=16,
        tip_size=0.07,
        xline_color=(0.335, 0.335, 0.335),
        yline_color=(0.335, 0.335, 0.335),
        zline_color=(0.335, 0.335, 0.335),
    )

    return cam, axes

def generate_rgba_mpl(args):   
    X_probes_cpu, grads_cpu, manifold_colors, dpi = args
    #X_probes_cpu, Es_manifold, dpi = args
    size = len(X_probes_cpu)
    fig_mpl = plt.Figure(figsize=((size)/dpi, (size)/dpi), dpi=dpi)
    canvas = FigureCanvas(fig_mpl)
    ax = fig_mpl.add_axes([0, 0, 1.0, 1.0])
    ax.set_axis_off()
    #fig_mpl.patch.set_alpha(0)
    fig_mpl.patch.set_facecolor((0.65, 0.65, 0.65))
    #ax.set_facecolor((1,0,0,1))

    ax.streamplot(X_probes_cpu[:, :, 0], X_probes_cpu[:, :, 1], grads_cpu[:, :, 0], grads_cpu[:, :, 1],
                  color=(0.98, 0.98, 0.98), density=1.2, linewidth=4, arrowsize=2)
    ax.pcolormesh(X_probes_cpu[:, :, 0], X_probes_cpu[:, :, 1], manifold_colors)
    #ax.contourf(X_probes_cpu[:, :, 0], X_probes_cpu[:, :, 1], Es_manifold, vmin=0, vmax=1.1, cmap="gray")
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[::-1] # origin should be bottom left not top left but mpl..
    plt.close(fig_mpl)

    return buf/255.0


def generate_rgba_mpl_contour(args):   
    X_probes_cpu, Es_manifold, dpi = args
    size = len(X_probes_cpu)
    fig_mpl = plt.Figure(figsize=((size)/dpi, (size)/dpi), dpi=dpi)
    canvas = FigureCanvas(fig_mpl)
    ax = fig_mpl.add_axes([0, 0, 1.0, 1.0])
    ax.set_axis_off()
    #fig_mpl.patch.set_alpha(0)
    fig_mpl.patch.set_facecolor((0.65, 0.65, 0.65))
    #ax.set_facecolor((1,0,0,1))

    ax.contourf(X_probes_cpu[:, :, 0], X_probes_cpu[:, :, 1], Es_manifold, vmin=0, vmax=1.1, cmap="gray")
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[::-1] # origin should be bottom left not top left but mpl..
    plt.close(fig_mpl)

    return buf/255.0


import numpy as np
from scipy.spatial import cKDTree
def bake_shadows_xy(rgb_init, mesh, centers_xyz, radius, strength=0.6):
    pts = np.asarray(mesh.points)
    xy = pts[:, :2].astype(np.float32)
    cxy = np.asarray(centers_xyz)[:, :2].astype(np.float32)

    sigma = radius / 2.5
    tree = cKDTree(cxy)
    
    indices = tree.query_ball_point(xy, radius)
    s = np.zeros(xy.shape[0], dtype=np.float32)

    for i, idx in enumerate(indices):
        if idx:
            diff = xy[i] - cxy[idx]
            d2 = (diff**2).sum(axis=1)
            w = np.exp(-d2 / (2 * sigma * sigma))
            s[i] = np.sum(w)

    s = np.clip(strength * s, 0.0, 1.0)
    rgb = np.clip(rgb_init * (1.0 - s[:, None]), 0.0, 1.0)
    
    return rgb


def vedo_grid_set_rgb(grid, rgb_array):
    grid.pointdata['rgb'] = rgb_array

    # Configure the VTK mapper for direct per-point RGB coloring
    grid.mapper.SetScalarModeToUsePointFieldData()
    grid.mapper.SelectColorArray('rgb')
    grid.mapper.SetColorModeToDirectScalars()
    grid.mapper.SetScalarVisibility(True)

    return grid

def generate_3d_frames(args):
    t, Xs_saves_t, Es_cells_t, mute_cells_t, Es_manifold_t, manifold_colors, rgba_mpls_t, rgba_mpls_t_contour, x, y, size, dirname, isTransparant = args
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
    #grid = vedo_grid_set_rgb(grid, bake_shadows_xy(manifold_colors, grid, Xs_probes_3d, radius=0.09, strength=0.35))
    #grid = vedo_grid_set_rgb(grid, manifold_colors)
    #grid = vedo_grid_set_rgb(grid, rgba_mpls_t.reshape(size*size, -1)[:, :-1])
    grid = vedo_grid_set_rgb(grid, bake_shadows_xy(rgba_mpls_t.reshape(size*size, -1)[:, :-1], grid, Xs_probes_3d, radius=0.09, strength=0.35))
    
    
    grid_streamplot = vedo.Grid(s=[x, y], c="red").linewidth(0)#.lighting('plastic')
    grid_streamplot.alpha(1.0) 
    grid_streamplot = vedo_grid_set_rgb(grid_streamplot, rgba_mpls_t_contour.reshape(size*size, -1)[:, :-1])
    
    ##
    # get surface normals at closest points
    sn = grid.point_normals
    ids = np.asarray([grid.closest_point(p, return_point_id=True) for p in cells.points])
    pts = cells.points
    pts += dz * sn[ids]                     # push along normal
    cells.points = pts

    txt = vedo.Text2D(f"t={t}", s=2.0)

    origin_dot = vedo.Point([np.min(x), np.min(y), 0], c=(0.335, 0.335, 0.335), r=15.12)  # r is the radius in screen pixels
    cam, axes = get_cam_n_axes(grid)

    for part in axes.unpack():
        if "z" in part.name.lower() and t>0:
            part.off()
    
    if isTransparant:
        vedo.settings.screenshot_transparent_background = True

    fig.show(txt, grid, grid_streamplot, cells, origin_dot, camera=cam, axes=axes) 
    #fig.show(grid, grid_streamplot, cells, origin_dot, camera=cam, axes=axes)
    #fig.parallel_projection(True) # Put it before show, and it has no effect smh
    fig.camera.SetParallelProjection(True)
    vedo.screenshot(dirname+"/frame_" + str(t).zfill(7) + ".png")
    fig.close()

