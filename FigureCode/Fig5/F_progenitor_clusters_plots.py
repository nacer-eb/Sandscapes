import numpy as np
import vedo

from loading_utils import *

load_analysis_data(datafile="../../datasets/weinreb/data_preprocessed_all_t_markers.npz",
                                  runname="data_weinreb/worker_0_detail.npz")

ctx = load_analysis_data()
Ms_saves_m1k_nca = ctx["Ms_saves_m1k_nca"]
lin_mask = ctx["lin_mask"]
kit_mask = ctx["kit_mask"]
X_LK_nca = ctx["X_LK_nca"]
prog_gates = ctx["prog_gates"]
pop_masks = ctx["pop_masks"]



fig = vedo.Plotter(axes=0,  size=(3840, 2160), interactive=False, offscreen=True)
cells_ct = [0, 0]
cells_ct[0] = vedo.Points(Ms_saves_m1k_nca[lin_mask, :3][~kit_mask], r=7, alpha=1.0, c=[0.9, 0.9, 0.9])
cells_ct[1] = vedo.Points(Ms_saves_m1k_nca[~lin_mask, :3], r=7, alpha=1.0, c=[0.9, 0.9, 0.9])

gated_population_points = []
for pop_name, pop_mask in pop_masks.items():
    pop_coords = X_LK_nca[pop_mask]
    
    print(f"{len(pop_coords)} cells in {pop_name}")
    if len(pop_coords) > 0:
        pts = vedo.Points(pop_coords, r=8, c=prog_gates[pop_name]["color"])
        gated_population_points.append(pts)


assigned_mask = np.zeros(len(X_LK_nca), dtype=bool)
for mask in pop_masks.values():
    assigned_mask = assigned_mask | mask
cells_unassigned = vedo.Points(X_LK_nca[~assigned_mask], r=7, c="k")

#old view
cam = dict(
    pos=(-26.1649, -1.14319, 7.58693),
    focal_point=(12.2776, 7.61583, -13.6800),
    viewup=(0.203296, -0.978473, -0.0355131),
    roll=176.139,
    distance=44.7976,
    clipping_range=(0.156085, 156.085),
)


fig.show(*cells_ct, cells_unassigned, *gated_population_points, axes=0, camera=cam)
fig.screenshot("Fig/F_LIN_KIT_CLUSTERS.png")
fig.close()


#####
