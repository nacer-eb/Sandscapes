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
data_run = np.load(runname+".npz")
Ms_saves_cpu_detail = data_run["Ms_saves_cpu_detail"]
t_saves = data_run["t_saves"]
t_id = -5
kmin, kmax = 50*2000, -1 #50*(t_saves[t_id])
Ms_saves_f = Ms_saves_cpu_detail[-1][kmin:kmax]
Ms_saves_m1k = Ms_saves_cpu_detail[t_id][kmin:kmax]
Ms_saves_m1k_nca = Ms_saves_m1k@NCA_mtx
Ms_saves_f_nca = Ms_saves_f@NCA_mtx



# Coloring based on marker
pca = data["pca_components"]
pca_inv = np.linalg.pinv(pca)
gene_names_high = data["gene_names_high"]
gene_markers = [
    "Kit", "Ly6a", "Cd34", "Fcgr3", "Itga2b", "Itgb3", "Slamf1", 
    "Il1rl1", "Itgb7", "Il3ra", "Itga2", "Il2rb", "Flt3", "Il7r", 
    "Ccr9", "Itgam", "Ly6g", "Ptprc", "Cd3g", "Gypa", "Fcer1a"
]

marker_mask = np.isin(np.char.upper(gene_names_high), np.char.upper(gene_markers))
matching_genes = gene_names_high[marker_mask]
marker_plotting_order = np.asarray([13, 2, 3, 6, 18, 16, 17, 1, 19, 4, 12, 14, 15, 7, 9, 11, 10, 8, 0, 5])
ordered_genes = matching_genes[marker_plotting_order].tolist()
data_matrix_markers = (Ms_saves_m1k @ pca_inv.T)[:, marker_mask][:, marker_plotting_order]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams.update({
    'font.size': 22,          # General text size
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 28,     # Axis label size (e.g., "Kit")
    'xtick.labelsize': 22,    # Tick number size
    'ytick.labelsize': 22,
    'legend.fontsize': 24
})


lin_bounds = {
    "Itgam": [-0.7923, -0.1243],
    "Cd3g": [-0.6321, 0.0455],
    "Gypa": [-0.2647, 0.1902],
    "Fcer1a": [-0.4509, -0.0226],
}

lin_labels = [
    "Itgam\n ~CD11b \n Myeloid",
    "Cd3g\n~CD3γ \n T-cell",
    "Gypa \n ~Ter119 \n Erythroid",
    "Fcer1a\n~FcεRIα \n Mast-Baso",
]


##### FIGURE 5 D (left) #####################
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
for i, marker in enumerate(lin_bounds):
    X = data_matrix_markers[:, ordered_genes.index(marker)]
    v1 = ax.violinplot(X[X<lin_bounds[marker][1]], positions=[i],
                       widths=0.6, showmeans=False, showmedians=False, showextrema=False)
    v2 = ax.violinplot(X[X>lin_bounds[marker][1]], positions=[i],
                       widths=0.6, showmeans=False, showmedians=False, showextrema=False)
    for pc in v1['bodies']:
        pc.set_facecolor('k')
        pc.set_edgecolor('k')
    for pc in v2['bodies']:
        pc.set_facecolor('k')
        pc.set_edgecolor('k')
        pc.set_alpha(0.1)

ax.set_xticks(np.arange(len(lin_labels)))
ax.set_xticklabels(list(lin_labels))
ax.set_ylabel("Genetic Expression residuals")
plt.subplots_adjust(left=0.15, bottom=0.115)
import os; os.makedirs("Fig", exist_ok=True)
plt.savefig("Fig/D_LIN_GATE_1.png")


##### FIGURE 5 D (right) #####################
lin_mask = data_matrix_markers[:, ordered_genes.index("Itgam")] <  lin_bounds["Itgam"][1]
for marker in lin_bounds:
    print(marker, lin_bounds[marker][1])
    lin_mask &= data_matrix_markers[:, ordered_genes.index(marker)] <  lin_bounds[marker][1]

kit_bound = -0.18 #-0.2063
fig, ax = plt.subplots(1, 1, figsize=(4, 9))
X = data_matrix_markers[lin_mask, ordered_genes.index("Kit")]
kit_mask = X>kit_bound
v1 = ax.violinplot(X[kit_mask], positions=[0],
                   widths=0.6, showmeans=False, showmedians=False, showextrema=False)
v2 = ax.violinplot(X[~kit_mask], positions=[0],
                   widths=0.6, showmeans=False, showmedians=False, showextrema=False)

for pc in v1['bodies']:
    pc.set_facecolor('k')
    pc.set_edgecolor('k')
for pc in v2['bodies']:
    pc.set_facecolor('k')
    pc.set_edgecolor('k')
    pc.set_alpha(0.1)

ax.set_xticklabels(["Kit \n ~Stem/Prog"])
ax.set_xticks([0])
ax.set_ylabel("")
plt.tight_layout()
#plt.show()
plt.savefig("Fig/D_LIN_GATE_2.png")




##### FIGURE 5 E #####################
fig = vedo.Plotter(axes=0, pos=(1366, 0), size=(3840, 2160), interactive=True, offscreen=False)
cells_ct = [0, 0, 0]
cells_ct[0] = vedo.Points(Ms_saves_m1k_nca[lin_mask, :3][kit_mask], r=7, alpha=1.0, c="k")
cells_ct[1] = vedo.Points(Ms_saves_m1k_nca[lin_mask, :3][~kit_mask], r=7, alpha=1.0, c=[0.9, 0.9, 0.9])
cells_ct[2] = vedo.Points(Ms_saves_m1k_nca[~lin_mask, :3], r=7, alpha=1.0, c=[0.9, 0.9, 0.9])

cam = dict(
    pos=(89.9062, 21.2696, -50.2680),
    focal_point=(8.34769, -3.25451, -16.7184),
    viewup=(-0.367635, -0.0479114, -0.928735),
    roll=92.8505,
    distance=91.5358,
    clipping_range=(0.304213, 304.213),
)

fig.show(*cells_ct, axes=0, camera=cam)
fig.screenshot("Fig/E_LIN_TREE.png")
fig.close()

#
