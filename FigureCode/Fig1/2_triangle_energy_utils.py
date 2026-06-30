import seaborn as sns
sns.set_theme(palette="pastel")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.spatial.distance as distance
import numpy as np

import vedo

def get_cam_n_axes(grid):
    cam = dict(
        pos=(3.52364, -6.31033, 3.00090),
        focal_point=(-0.185445, -0.0439029, 0.943929),
        viewup=(-0.143162, 0.231149, 0.962328),
        roll=-65.6449,
        parallel_scale=1.70268,
        clipping_range=(1.23751, 15.2291),
    )


    axes = vedo.Axes(
        grid,
        show_ticks=False,
        xtitle="g_1",
        ytitle="g_2",
        xtitle_position=0.55,
        ytitle_position=0.55,
        xtitle_offset=0.05,
        ytitle_offset=0.05,
        xtitle_size=0.04,
        ytitle_size=0.04,
        ztitle="Energy",
        xlabel_size=0.0,
        ylabel_size=0.0,
        zlabel_size=0.0,
        xlabel_color="black",
        ylabel_color="black",
        zlabel_color="black",
        ztitle_position=0.75,
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
    X_probes_cpu, grads_cpu, dpi = args
    fig_mpl = plt_mpl.Figure(figsize=((size)/dpi, (size)/dpi), dpi=dpi)
    canvas = FigureCanvas(fig_mpl)
    ax = fig_mpl.add_axes([0, 0, 1.0, 1.0])
    ax.set_axis_off()
    fig_mpl.patch.set_alpha(0)
    ax.set_facecolor((0,0,0,0))

    ax.streamplot(X_probes_cpu[:, :, 0], X_probes_cpu[:, :, 1], grads_cpu[:, :, 0], grads_cpu[:, :, 1], color="white", density=2.0)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[::-1] # origin should be bottom left not top left but mpl..

    plt_mpl.close(fig_mpl)

    return buf


def bake_shadows_xy(rgb_init, mesh, centers_xyz, radius, strength=0.6):
    pts = np.asarray(mesh.points)          # (M,3)
    xy  = pts[:, :2].astype(np.float32)      # (M,2)
    cxy = np.asarray(centers_xyz)[:, :2].astype(np.float32)  # (N,2)

    sigma = radius / 2.5                 # tweak

    # accumulate shadow factor per mesh point
    d2 = ((xy[:, None, :] - cxy[None, :, :])**2).sum(-1)               # (M,N)
    w  = np.exp(-d2 / (2*sigma*sigma)) * (d2 <= radius*radius)         # (M,N)
    s  = np.clip(strength * w.sum(axis=1), 0.0, 1.0).astype(np.float32) # (M,)

    rgb = np.clip(rgb_init * (1.0 - s[:, None]), 0.0, 1.0)                 # darken
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
    t, Xs_saves_t, Es_cells_t, mute_cells_t, Es_manifold_t, rgba_mpls_t, x, y, size, dirname = args

    ## Colors that pop more
    XIs_colors = np.array([
        [120, 190, 120],   # greener, slightly darker
        [ 90, 170, 230],   # more saturated blue
        [240, 150, 170],   # stronger pink
    ]) / 255.0

    z_factor = 2.0
    dz = 0.075

    fig = vedo.Plotter(axes=1, size=(1920,1080), interactive=True, offscreen=False)


    # Plotting cells
    Xs_probes_3d = np.column_stack((Xs_saves_t, Es_cells_t * z_factor))
    Xs_probes_3d = Xs_probes_3d[mute_cells_t]
    cells = vedo.Points(Xs_probes_3d, r=55, c=np.asarray([0.97, 0.97, 0.95])).lighting('shiny')


    # Grid
    grid = vedo.Grid(s=[x, y]).linewidth(0).lighting('plastic')
    grid.points[:, 2] = Es_manifold_t * z_factor

    # E - manifold color
    coefs = np.exp(-beta*distance.cdist(Xs_probes, XIs)**2)
    coefs = coefs/coefs.sum(axis=-1)[:, None]
    manifold_colors = np.ones((len(Xs_probes), 4))
    manifold_colors = coefs@XIs_colors

    grid.alpha(0.8)  # instead of per-point alpha in the array
    grid = vedo_grid_set_rgb(grid, bake_shadows_xy(manifold_colors, grid, Xs_probes_3d, radius=0.09, strength=0.35))

    ##
    # get surface normals at closest points
    sn = grid.point_normals
    ids = np.asarray([grid.closest_point(p, return_point_id=True) for p in cells.points])
    pts = cells.points
    pts += dz * sn[ids]                     # push along normal
    cells.points = pts



    grid_streamplot = vedo.Grid(s=[x, y], c="red").linewidth(0).lighting('plastic')
    grid_streamplot.alpha(0.95)  # instead of per-point alpha in the array
    grid_streamplot = vedo_grid_set_rgb(grid_streamplot, rgba_mpls_t.reshape(size*size, -1)[:, :-1])

    origin_dot = vedo.Point([np.min(x), np.min(y), 0], c=(0.335, 0.335, 0.335), r=15.12)  # r is the radius in screen pixels

    cam, axes = get_cam_n_axes(grid)
    fig.show(grid, grid_streamplot, cells, origin_dot, camera=cam, axes=axes)
    vedo.screenshot(dirname+"/frame_" + str(t).zfill(7) + ".png")
    fig.close()

