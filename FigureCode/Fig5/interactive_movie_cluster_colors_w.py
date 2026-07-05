import numpy as np
import scipy.spatial.distance as distance
import vedo
from loading_utils import load_analysis_data

ctx = load_analysis_data(
    datafile="../../highdim/datasets/weinreb_v4/data_preprocessed_all_t_markers.npz",
    runname="WEINREB_VM_1/worker_2_detail.npz"
)

XI_NCA = ctx["XI_NCA"]
Ls = ctx["Ls"]
ct_names = ctx["ct_names"]
Ms_saves_cpu_nca = ctx["Ms_saves_cpu_nca"]

Ms_filtered_end = Ms_saves_cpu_nca[-1]
N_labels = len(ct_names)
XI_onehot = np.eye(N_labels, dtype=np.float32)[Ls.astype(int)]
S_end = np.exp(-0.01 * distance.cdist(Ms_filtered_end, XI_NCA) ** 2) @ XI_onehot
S_end_norm = S_end / np.sum(S_end, axis=-1)[:, None]
Ls_end = np.argmax(S_end_norm, axis=-1)

pastel_colors = np.array([
    [0.850, 0.850, 0.850],
    [0.950, 0.900, 0.550],
    [0.500, 0.900, 0.900],
    [0.600, 0.900, 0.600],
    [1.000, 0.750, 0.500],
    [0.950, 0.600, 0.600],
    [0.800, 0.650, 0.500],
    [0.750, 0.600, 0.900],
    [1.000, 0.600, 0.800],
    [0.850, 0.550, 0.850],
    [0.550, 0.700, 0.950]
])

import vedo
vid = vedo.Video("A_full_tree_movie.mp4", fps=30)
fig = vedo.Plotter(axes=0,  size=(3840, 2160), interactive=False, offscreen=True)

t = 0
selected_ct = np.arange(len(ct_names))
cells_ct = [0 for i in selected_ct]
for i, j in enumerate(selected_ct):
    cells_ct[i] = vedo.Points(Ms_saves_cpu_nca[t, Ls_end==j, :3], r=7, alpha=1.0, c=list(pastel_colors[j]))
    cells_ct[i].name = ct_names[j]
legend = vedo.LegendBox(cells_ct, pos='bottom-left', width=0.15, height=0.4)
txt = vedo.Text2D("t="+str(t))

isSaving = True
def update(event):
    global t, vid, isSaving
    t += 1
    
    if t >= len(Ms_saves_cpu_nca)-1 and isSaving:
        isSaving = False
        vid.close()

    if isSaving:
        vid.add_frame()
        print(f"{t} out of {len(Ms_saves_cpu_nca)}")

    t = t%len(Ms_saves_cpu_nca) # loop
    for i, j in enumerate(selected_ct):
        cells_ct[i].points = Ms_saves_cpu_nca[t, Ls_end==j, :3]
    txt.text( "t="+str(t))
    fig.render()
    #


cam = dict(
    pos=(-87.4642, 7.51542, -47.4414),
    focal_point=(3.70354, -0.350938, 13.7055),
    viewup=(0.0254612, 0.995605, 0.0901198),
    roll=3.47780,
    distance=110.056,
    clipping_range=(0.237621, 237.621),
)

fig.add_callback('Timer', update)
fig.timer_callback('start')
fig.show(*cells_ct, legend, txt, axes=0, camera=cam) # cam ?
fig.close()
#

