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

expr_LK_markers = data_matrix_markers[lin_mask][kit_mask]
X_LK_nca = Ms_saves_m1k_nca[lin_mask, :3][kit_mask]


clusters = ["MEP", "GMP", "CMP", "CLP"]
colors = [prog_gates[c]["color"] for c in clusters]
max_n = 500

figure_layouts = {
    "CMPCLP_genes": [
        ["Ly6a", "Elane"],
        ["Ighm", "Pf4"],
        ["H2afy", "Hbb-bt"]
    ],
    "MEPGMP_genes": [
        ["Spp1", "Car2"],
        ["Mpo", "Ppbp"],
        ["Gzmb", "Bex4"]
    ]
}


pop_masks = {}
for pop_name, params in prog_gates.items():
    mask_slamf1 = (expr_LK_markers[:, slamf1_idx] >= params["Slamf1"][0]) & (expr_LK_markers[:, slamf1_idx] <= params["Slamf1"][1])
    mask_fcgr3 = (expr_LK_markers[:, fcgr3_idx] >= params["Fcgr3"][0]) & (expr_LK_markers[:, fcgr3_idx] <= params["Fcgr3"][1])
    pop_mask = mask_slamf1 & mask_fcgr3
    pop_masks[pop_name] = pop_mask

expr_LK_full = (Ms_saves_m1k @ pca_inv.T)[lin_mask][kit_mask]
for plot_name, layout in figure_layouts.items():
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    for row in range(3):
        for col in range(2):
            gene = layout[row][col]
            gene_idx = list(gene_names_high).index(gene)
            
            gene_data = []
            weights = []
            
            for c in clusters:
                expr_pop = expr_LK_full[pop_masks[c]][:, gene_idx]
                if len(expr_pop) > max_n:
                    idx = np.random.choice(len(expr_pop), max_n, replace=False)
                    expr_pop = expr_pop[idx]
                gene_data.append(expr_pop)
                weights.append(np.ones_like(expr_pop) / len(expr_pop))
            
            axes[row, col].hist(gene_data, bins=30, color=colors, label=clusters, histtype='bar', stacked=True, weights=weights)
            
            axes[row, col].set_xlabel(f"{gene} Expression")
            axes[row, col].set_ylabel("Proportion")
            axes[row, col].set_title(f"Normalized {gene} Expression")
            
            if row == 0 and col == 1:
                axes[row, col].legend(title="Populations")

    plt.tight_layout()
    plt.savefig(f"Fig/S_{plot_name}.png")
    plt.close()

