import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from loading_utils import load_analysis_data

matplotlib.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24
})

ctx = load_analysis_data()
ordered_genes = ctx["ordered_genes"]
expr_LK_markers = ctx["expr_LK_markers"]
prog_gates = ctx["prog_gates"]
pop_masks = ctx["pop_masks"]

slamf1_idx = ordered_genes.index("Slamf1")
fcgr3_idx = ordered_genes.index("Fcgr3")
cd34_idx = ordered_genes.index("Cd34")

fig_mpl, axs = plt.subplots(2, 1, figsize=(16, 28), dpi=200)
for pop_name, pop_mask in pop_masks.items():
    pop_markers = expr_LK_markers[:, [slamf1_idx, fcgr3_idx]][pop_mask]
    pop_markers_2 = expr_LK_markers[:, [cd34_idx, fcgr3_idx]][pop_mask]
    
    if pop_name == "CLP":
        axs[0].scatter(*pop_markers.T, s=10, c=prog_gates[pop_name]["color"])
        axs[1].scatter(*pop_markers_2.T, s=10, c=prog_gates[pop_name]["color"])
    else:
        axs[0].scatter(*pop_markers.T, s=1, c=prog_gates[pop_name]["color"])
        axs[1].scatter(*pop_markers_2.T, s=1, c=prog_gates[pop_name]["color"])


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

