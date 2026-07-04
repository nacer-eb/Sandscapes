import tqdm
import scipy.spatial.distance as distance
import numpy as np
import vedo


# Load XI data
datafile = "../../datasets/weinreb/data_preprocessed_all_t_markers.npz"
data = np.load(datafile, allow_pickle=True)
XI = data['XI_pca'][::1]
XI_NCA = data['XI_NCA'][::1]
NCA_mtx = data['NCA_mtx'].T
Ls = data['Ls'][::1]
ct_names = data['ct_names']
cmap = data['cmap'].item()

# Load simulation snapshots
runname = "data_weinreb/worker_0_detail"
data_run = np.load(runname+".npz", allow_pickle=True)
Ms_saves_cpu_detail = data_run["Ms_saves_cpu_detail"]
t_saves = data_run["t_saves"]

t_id = -4
kmin, kmax = 50*2000, -1#50*(t_saves[t_id+1])
Ms_saves_f = Ms_saves_cpu_detail[-2][kmin:kmax]
Ms_saves_m1k = Ms_saves_cpu_detail[t_id][kmin:kmax]
Ms_saves_m1k_nca = Ms_saves_m1k@NCA_mtx
Ms_saves_f_nca = Ms_saves_f@NCA_mtx


# Coloring based on final state
N_labels = int(len(ct_names))
XI_onehot = np.eye(N_labels, dtype=np.float32)[Ls.astype(int)]
S_end = np.exp(-0.025*distance.cdist(Ms_saves_f_nca, XI_NCA)**2)@XI_onehot # _nca
S_end_norm = (S_end)/np.sum(S_end, axis=-1)[:, None]
Ls_end = np.argmax(S_end_norm, axis=-1)


fig = vedo.Plotter(axes=0,  size=(3840, 2160), interactive=False, offscreen=True)
selected_ct = np.arange(len(ct_names))
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

axes_len = np.asarray([2.0, 1.0, 1.0])
for i, j in enumerate(selected_ct):
    cells_ct[i] = vedo.Points(Ms_saves_m1k_nca[Ls_end==j, :3], r=7, alpha=1.0, c=pastel_colors[j])
    cells_ct[i].name = ct_names[j]
#legend = vedo.LegendBox(cells_ct, pos='bottom-left', width=0.15, height=0.4)

cam = dict(
    pos=(89.9062, 21.2696, -50.2680),
    focal_point=(8.34769, -3.25451, -16.7184),
    viewup=(-0.367635, -0.0479114, -0.928735),
    roll=92.8505,
    distance=91.5358,
    clipping_range=(0.304213, 304.213),
)


fig.show(*cells_ct, axes=0, camera=cam) # cam ? # legend,
import os
os.makedirs("Fig", exist_ok=True)
fig.screenshot("Fig/A_FULL_TREE.png")
fig.close()
#




fig = vedo.Plotter(axes=0,  size=(3840, 2160), interactive=False, offscreen=True)
XIs_pnts = vedo.Points(XI_NCA[Ls>0], r=10, alpha=0.5, c="k")

b = XIs_pnts.bounds()
cx = (b[0] + b[1]) / 2.0
cy = (b[2] + b[3]) / 2.0
sx = b[1] - b[0]
sy = b[3] - b[2]
z_min = b[4]

floor = vedo.Plane(pos=(cx, cy, z_min - 0.5), s=(sx*1.0 , sy*1.0 ), c="lightgray", alpha=0.2)
shadow = XIs_pnts.add_shadow(plane='z', point=z_min, alpha=0.5)

cam = dict(
    pos=(-26.8855, -375.541, 170.116),
    focal_point=(-7.96292, -20.9642, -5.82531),
    viewup=(0.0110217, 0.443989, 0.895964),
    roll=6.12901,
    distance=396.280,
    clipping_range=(157.327, 712.565),
)

fig.show(XIs_pnts, floor, shadow, axes=0, camera=cam)

os.makedirs("Fig", exist_ok=True)
fig.screenshot("Fig/A_FULL_TREE_train.png")
fig.close()
