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
LOW_SLAMF1 = [-0.0719, -0.035]
HIGH_SLAMF1 = [-0.031, -0.0]

NO_FCGR3 = [-0.4276, -0.3734]
LOW_FCGR3 = [-0.32, -0.29]
HIGH_FCGR3 = [-0.2085, -0.169]

prog_gates = {
    "MEP": {
        "Slamf1": HIGH_SLAMF1,
        "Fcgr3": LOW_FCGR3, # 2901
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





fig = vedo.Plotter(axes=0,  size=(3840, 2160), interactive=False, offscreen=True)
cells_ct = [0, 0]
cells_ct[0] = vedo.Points(Ms_saves_m1k_nca[lin_mask, :3][~kit_mask], r=7, alpha=1.0, c=[0.9, 0.9, 0.9])
cells_ct[1] = vedo.Points(Ms_saves_m1k_nca[~lin_mask, :3], r=7, alpha=1.0, c=[0.9, 0.9, 0.9])

gated_population_points = []
pop_masks = {}
for pop_name, params in prog_gates.items():
    mask_slamf1 = (expr_LK_markers[:, slamf1_idx] >= params["Slamf1"][0]) & (expr_LK_markers[:, slamf1_idx] <= params["Slamf1"][1])
    mask_fcgr3 = (expr_LK_markers[:, fcgr3_idx] >= params["Fcgr3"][0]) & (expr_LK_markers[:, fcgr3_idx] <= params["Fcgr3"][1])
    pop_mask = mask_slamf1 & mask_fcgr3
    pop_masks[pop_name] = pop_mask
    
    pop_coords = X_LK_nca[pop_mask]
    if len(pop_coords) > 0:
        pts = vedo.Points(pop_coords, r=8, c=params["color"])
        gated_population_points.append(pts)

assigned_mask = np.zeros(len(expr_LK_markers), dtype=bool)
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




## VOLCANO TIME
"""
import scipy.stats
import os
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

comparisons = [
    ("CMP", "CLP"),
    ("MEP", "GMP")
]

expr_LK_full = (Ms_saves_m1k @ pca_inv.T)[lin_mask][kit_mask]
for pop_A, pop_B in comparisons:      
    expr_A = expr_LK_full[pop_masks[pop_A]]
    expr_B = expr_LK_full[pop_masks[pop_B]]

    max_n = 100

    if len(expr_A) > max_n:
        idx_A = np.random.choice(len(expr_A), max_n, replace=False)
        expr_A = expr_A[idx_A]
    else:
        expr_A = expr_A

    if len(expr_B) > max_n:
        idx_B = np.random.choice(len(expr_B), max_n, replace=False)
        expr_B = expr_B[idx_B]
    else:
        expr_B = expr_B
    
    t_stat, p_array = scipy.stats.ttest_ind(expr_A, expr_B, equal_var=False)
    
    mean_diff = np.mean(expr_A, axis=0) - np.mean(expr_B, axis=0)
    
    p_array = np.clip(p_array, 1e-300, 1.0)
    n_p_value = -np.log10(p_array)
    
    cutoff = 4.2 * np.std(mean_diff)
    pvalue_min = 1.0
    
    mask_high_selected_genes = (n_p_value > pvalue_min) & (mean_diff > cutoff)
    mask_low_selected_genes = (n_p_value > pvalue_min) & (mean_diff < -cutoff)
    

    ax_w = 206 / 25.4
    ax_h = 199 / 25.4
    m_left = 0.5
    m_right = 1.2
    m_bottom = 0.8
    m_top = 0.5
    fig_w = ax_w + m_left + m_right
    fig_h = ax_h + m_bottom + m_top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([m_left / fig_w, m_bottom / fig_h, ax_w / fig_w, ax_h / fig_h])

    mask_selected_genes = mask_low_selected_genes | mask_high_selected_genes
    
    #ax.scatter(mean_diff, n_p_value, alpha=0.5)
    mask_not_selected = ~mask_selected_genes
    ax.scatter(mean_diff[mask_not_selected], n_p_value[mask_not_selected], color="lightgray", alpha=0.5)

    color_A=prog_gates[pop_A]["color"]
    color_B=prog_gates[pop_B]["color"]
    ax.scatter(mean_diff[mask_high_selected_genes], n_p_value[mask_high_selected_genes], color=color_A, s=150)
    ax.scatter(mean_diff[mask_low_selected_genes], n_p_value[mask_low_selected_genes], color=color_B, s=150)
    ax.set_xlim(-np.max(np.abs(mean_diff))*1.1, np.max(np.abs(mean_diff))*1.1)
    ax.set_ylim(40, np.max(np.abs(n_p_value))*1.1)
    
    texts = []
    for i in np.argwhere(mask_selected_genes).ravel():
        texts.append(ax.text(mean_diff[i], n_p_value[i], gene_names_high[i], fontsize=28))
        
    #adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), max_iterations=50000)
    adjust_text(
        texts,
        force_text=(1.2, 6.0),
        force_points=(1.2, 4.0),
        expand_points=(1.1, 1.5),
        expand_text=(1.1, 2.0),
        max_iterations=10000,
        arrowprops=dict(arrowstyle="-", color='gray', lw=1.5, alpha=0.7)
    )
    
    ax.axvline(cutoff, color="red", linestyle="--")
    ax.axvline(-cutoff, color="red", linestyle="--")
    ax.axhline(pvalue_min, color="red", linestyle="--")
    
    #ax.set_title(f"{pop_A} vs {pop_B}")
    ax.set_xlabel(f"Mean Difference") #  ({pop_A} - {pop_B})

    if pop_A == "CLP":
        ax.set_xlabel("")
        ax.set_xticks([])

    ax.set_yticks(np.arange(40, np.max(np.abs(n_p_value))*1.1, 25))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("-log10(p-value)")


    #plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    plt.savefig(f"Fig/GH_Volcano_pop_{pop_A}_{pop_B}.png")
    plt.close()
"""


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


