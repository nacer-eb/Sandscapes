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
    'axes.labelsize': 24,     # Axis label size (e.g., "Kit")
    'xtick.labelsize': 24,    # Tick number size
    'ytick.labelsize': 24,
    'legend.fontsize': 24
})


lin_bounds = {
    "Itgam": [-0.7923, -0.1243],
    "Cd3g": [-0.6321, 0.0455],
    "Gypa": [-0.2647, 0.1902],
    "Fcer1a": [-0.4509, -0.0226],
}

lin_mask = data_matrix_markers[:, ordered_genes.index("Itgam")] <  lin_bounds["Itgam"][1]
for marker in lin_bounds:
    print(marker, lin_bounds[marker][1])
    lin_mask &= data_matrix_markers[:, ordered_genes.index(marker)] <  lin_bounds[marker][1]
kit_bound = -0.18
X = data_matrix_markers[lin_mask, ordered_genes.index("Kit")]
kit_mask = X>kit_bound


###
LOW_SLAMF1 = [-0.0719, -0.039]
HIGH_SLAMF1 = [-0.025, -0.00]

NO_FCGR3 = [-0.4276, -0.3734]
LOW_FCGR3 = [-0.32, -0.29]
HIGH_FCGR3 = [-0.2085, -0.169]

prog_gates = {
    "MEP": {
        "Slamf1": HIGH_SLAMF1,
        "Fcgr3": [-0.35, -0.32], # in between -/low
        "color": "#D49A89"
    },
    
    "GMP": {
        "Slamf1": LOW_SLAMF1,
        "Fcgr3": HIGH_FCGR3,
        "color": "#A9D1B3"
    },
    "CMP": {
        "Slamf1": LOW_SLAMF1,
        "Fcgr3": LOW_FCGR3, #[-0.3204, -0.2673]
        "color": "#EBD88A"
    },
    "CLP": {
        "Slamf1": LOW_SLAMF1,
        "Fcgr3": NO_FCGR3,
        "color": "#A3C2E0"
    },
   
}


slamf1_idx = ordered_genes.index("Slamf1")
fcgr3_idx = ordered_genes.index("Fcgr3")
cd34_idx = ordered_genes.index("Cd34")

expr_LK_markers = data_matrix_markers[lin_mask][kit_mask]
X_LK_nca = Ms_saves_m1k_nca[lin_mask, :3][kit_mask]


fig_mpl, axs = plt.subplots(2, 1, figsize=(16, 28), dpi=200)
pop_masks = {}
for pop_name, params in prog_gates.items():
    mask_slamf1 = (expr_LK_markers[:, slamf1_idx] >= params["Slamf1"][0]) & (expr_LK_markers[:, slamf1_idx] <= params["Slamf1"][1])
    mask_fcgr3 = (expr_LK_markers[:, fcgr3_idx] >= params["Fcgr3"][0]) & (expr_LK_markers[:, fcgr3_idx] <= params["Fcgr3"][1])
    pop_mask = mask_slamf1 & mask_fcgr3
    pop_masks[pop_name] = pop_mask
    
    pop_markers = expr_LK_markers[:, [slamf1_idx, fcgr3_idx]][pop_mask]
    pop_markers_2 = expr_LK_markers[:, [cd34_idx, fcgr3_idx]][pop_mask]
    
    if pop_name == "CLP":
        axs[0].scatter(*pop_markers.T, s=10, c=params["color"])
        axs[1].scatter(*pop_markers_2.T, s=10, c=params["color"])
    else:
        axs[0].scatter(*pop_markers.T, s=1, c=params["color"])
        axs[1].scatter(*pop_markers_2.T, s=1, c=params["color"])


assigned_mask = np.zeros(len(expr_LK_markers), dtype=bool)
for mask in pop_masks.values():
    assigned_mask = assigned_mask | mask
pop_markers = expr_LK_markers[:, [slamf1_idx, fcgr3_idx]][~assigned_mask]
pop_markers_2 = expr_LK_markers[:, [cd34_idx, fcgr3_idx]][~assigned_mask]

axs[0].scatter(*pop_markers.T, s=1, c="k", alpha=0.3)
axs[1].scatter(*pop_markers_2.T, s=1, c="k", alpha=0.3)
axs[0].set_xlabel("Slamf1 (CD150)")
axs[0].set_ylabel("Fcgr3 (CD16)")
axs[0].set_xlim(-0.08, 0.02)
axs[0].set_ylim(-0.5, 0.1)

axs[1].set_xlabel("Cd34 (CD34)")
axs[1].set_ylabel("Fcgr3 (CD16)")
axs[1].set_xlim(0.05, 0.25)
axs[1].set_ylim(-0.5, 0.1)

plt.savefig("Fig/S_Slamf1Fcgr3_Expression.png")

