import numpy as np
from pathlib import Path
from natsort import natsorted

datafile = "../../datasets/weinreb/data_preprocessed_all_t_markers.npz"
data = np.load(datafile, allow_pickle=True)
XI = data['XI_pca'][::1]
XI_NCA = data['XI_NCA'][::1]
NCA_mtx = data['NCA_mtx'].T
pca = data["pca_components"]
pca_inv = np.linalg.pinv(pca)
gene_names_high = data["gene_names_high"]
GENE = "Dmkn"
gene_id = np.argwhere(np.isin(np.char.upper(gene_names_high), np.char.upper(GENE))).ravel()[0]

Ls = data['Ls'][::1]
ct_names = data['ct_names']
cmap = data['cmap'].item()

runname = "WEINREB_VM_1/worker_2_detail"# worker_2_detail
data_run = np.load(runname+".npz")

Ms_saves_cpu_detail = data_run["Ms_saves_cpu_detail"]
Ms_saves_f = Ms_saves_cpu_detail[-2][::1]
Ms_saves_m1k = Ms_saves_cpu_detail[-5][::1]
Ms_saves_m1k_nca = Ms_saves_m1k@NCA_mtx
Ms_saves_f_nca = Ms_saves_f@NCA_mtx


import scipy.spatial.distance as distance
N_labels = int(len(ct_names))
XI_onehot = np.eye(N_labels, dtype=np.float32)[Ls.astype(int)]
S_end = np.exp(-0.01*distance.cdist(Ms_saves_f_nca, XI_NCA)**2)@XI_onehot
S_end_norm = (S_end)/np.sum(S_end, axis=-1)[:, None]
Ls_end = np.argmax(S_end_norm, axis=-1)

import vedo
fig = vedo.Plotter(axes=0, pos=(1366, 0), size=(1920, 1080), interactive=True, offscreen=False)
cells_ct = [0 for i in selected_ct]
pastel_colors = np.array([
    [0.850, 0.850, 0.850],  # 0  Undifferentiated      light grey
    [0.950, 0.900, 0.550],  # 1  Baso                  butter yellow
    [0.500, 0.900, 0.900],  # 2  Eos                   light cyan
    [0.600, 0.900, 0.600],  # 3  Mast                  mint green
    [1.000, 0.750, 0.500],  # 4  Neutrophil            pastel peach
    [0.950, 0.600, 0.600],  # 5  Erythroid             soft coral
    [0.800, 0.650, 0.500],  # 6  Meg                   warm beige
    [0.750, 0.600, 0.900],  # 7  Monocyte              soft lavender
    [1.000, 0.600, 0.800],  # 8  pDC                   bubblegum pink
    [0.850, 0.550, 0.850],  # 9  Ccr7_DC               muted plum
    [0.550, 0.700, 0.950]   # 10 Lymphoid              baby blue
])

for i, j in enumerate(selected_ct):
    cells_ct[i] = vedo.Points(Ms_saves_cpu_nca[Ls_end==j, :3], r=7, alpha=1.0, c=list(pastel_colors[j]))
    cells_ct[i].name = ct_names[j]
legend = vedo.LegendBox(cells_ct, pos='bottom-left', width=0.15, height=0.4)


txt = vedo.Text2D("GENE="+str(0))


cam = dict(
    pos=(-87.4642, 7.51542, -47.4414),
    focal_point=(3.70354, -0.350938, 13.7055),
    viewup=(0.0254612, 0.995605, 0.0901198),
    roll=3.47780,
    distance=110.056,
    clipping_range=(0.237621, 237.621),
)

fig.show(*cells_ct, txt, axes=0, camera=cam) # cam ?
fig.close()
#

