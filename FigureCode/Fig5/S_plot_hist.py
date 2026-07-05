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
expr_LK_full = ctx["expr_LK_full"]
gene_names_high = ctx["gene_names_high"]
prog_gates = ctx["prog_gates"]
pop_masks = ctx["pop_masks"]

clusters = ["MEP", "GMP", "CMP", "CLP"]
colors = [prog_gates[c]["color"] for c in clusters]
max_n = 100

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

