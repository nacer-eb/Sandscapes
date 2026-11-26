import tqdm
import numpy as np
import scipy.spatial.distance as distance
import vedo
from loading_utils import load_analysis_data

ctx = load_analysis_data(runname="/localhome/nacer/data_weinreb_26092026//worker_0_detail.npz",
                        runname_nca="/localhome/nacer/data_weinreb_26092026//worker_0.npz")

XI_NCA = ctx["XI_NCA"]
Ls = ctx["Ls"]
ct_names = ctx["ct_names"]
Ms_saves_cpu_nca = ctx["Ms_saves_cpu_nca"]
print(Ms_saves_cpu_nca.shape)

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


label_to_track = 5
cells_to_highlight = np.argwhere(Ls_end==label_to_track).ravel()[::50]
cell_ids = np.arange(0, Ms_saves_cpu_nca.shape[1])
highlight_color = pastel_colors[label_to_track]



vid = vedo.Video("A_full_tree_movie_celltrack.mp4", fps=30)
fig = vedo.Plotter(axes=0,  size=(1920, 1080), interactive=False, offscreen=True)

t = 0

selected_ct = np.arange(len(ct_names))
cells_ct = [0 for i in selected_ct]
for i, j in enumerate(selected_ct):
    cells_ct[i] = vedo.Points(Ms_saves_cpu_nca[t, (Ls_end==j)&(~np.isin(cell_ids, cells_to_highlight)), :3], r=7, alpha=0.01, c=list((0.1, 0.1, 0.1)))
    cells_ct[i].name = ct_names[j]
cells_tracked = vedo.Points(Ms_saves_cpu_nca[t, np.isin(cell_ids, cells_to_highlight), :3], r=7, alpha=1.0, c=list(highlight_color))



legend = vedo.LegendBox(cells_ct, pos='bottom-left', width=0.15, height=0.4)
txt = vedo.Text2D("t="+str(t))

cam = dict(
    pos=(89.9062, 21.2696, -50.2680),
    focal_point=(8.34769, -3.25451, -16.7184),
    viewup=(-0.367635, -0.0479114, -0.928735),
    roll=92.8505,
    distance=91.5358,
    clipping_range=(0.304213, 304.213),
)

fig.show(*cells_ct, legend, txt, axes=0, camera=cam) # cam ?

for t in tqdm.tqdm(range(0, len(Ms_saves_cpu_nca), 10)):
    for i, j in enumerate(selected_ct):
        cells_ct[i].points = Ms_saves_cpu_nca[t, (Ls_end==j)&(~np.isin(cell_ids, cells_to_highlight)), :3]
    cells_tracked.points = Ms_saves_cpu_nca[t, np.isin(cell_ids, cells_to_highlight), :3]
    txt.text("t=" + str(t))
    fig.render()
    vid.add_frame()

vid.close()
fig.close()
#

