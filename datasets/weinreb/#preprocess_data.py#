import sys
sys.path.append('../../common') 

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

import argparse
parser = argparse.ArgumentParser(description="Generate dataset and NCA from SEED")
parser.add_argument("--SEED", type=int, required=False, default=2, help="Run id (0, 1, ..) ")
args = parser.parse_args()
SEED = args.SEED #2 is dflt

def compute_pearson_residuals(data, gene_names, min_cells=3):
    """
    Filters out genes expressed in fewer than min_cells.
    Computes Negative Binomial Pearson residuals.
    """

    # Assumes data is of shape CELL x GENE
    
    # Filter out low-count genes
    cells_per_gene = np.sum(data > 0, axis=0)
    keep_mask = cells_per_gene >= min_cells
    data_filt = data[:, keep_mask]
    genes_filt = gene_names[keep_mask]
    
    # Calculate expected value matrix
    L = np.sum(data_filt, axis=1)
    G = np.sum(data_filt, axis=0)
    T = np.sum(data_filt) 
    E_value = (L[:, None] * G[None, :]) / T
    # G/T can be understood as the mean relative expression of each gene, L is the total genetic expression in a cell,
    # multiplied together we have for each cell a the expected expression of each gene
    # for more intuition, you can imagine there is a 1/N_cell and in G and in T that just cancels out, assuming each cell has equal probablity weight

    # Estimate raw Theta via Method of Moments
    mean_g = G / data_filt.shape[0] # genetic expression of each gene, averaged over cells
    var_g = np.var(data_filt, axis=0)
    valid = var_g > (mean_g + 1e-6) # if the mean is greater or equal to the variance, then this gene is roughly poisson and we don't fit theta
    raw_theta = np.full(G.shape, 100.0)
    raw_theta[valid] = (mean_g[valid]**2) / (var_g[valid] - mean_g[valid]) # we fit theta assuming that our data is s.t. var(gene) = mean((gene)) + mean(gene)**2/theta

    # We regularize, or smoothen, Theta via polynomial fit of the log # 
    fit_mask = valid & (mean_g > 1e-3)
    poly_coeffs = np.polyfit(np.log10(mean_g[fit_mask]), np.log10(raw_theta[fit_mask]), deg=2)
    
    log_mean_safe = np.log10(np.clip(mean_g, 1e-6, None))
    theta_reg = 10**(np.polyval(poly_coeffs, log_mean_safe))
    theta_reg = np.clip(theta_reg, 0.01, 1000.0)

    # Compute Variance and Residuals according the NB with our calculated theta
    E_variance = E_value + (E_value**2) / theta_reg[None, :] 
    E_variance = np.maximum(E_variance, 1e-4) # Prevent div by zero
    residuals = (data_filt - E_value) / np.sqrt(E_variance)

    # Outlier clipping
    vmax = np.sqrt(data_filt.shape[0])
    residuals = np.clip(residuals, -vmax, vmax)
    
    return residuals, data_filt, genes_filt

# =============================================================================
# DATASET SPECIFIC PORTION --- -
# =============================================================================

import scanpy as sc
import pandas as pd
adata = sc.read_mtx('./GSE140802_data/GSM4185642_stateFate_inVitro_normed_counts.mtx')

gene_names = np.asarray(pd.read_csv('./GSE140802_data/GSM4185642_stateFate_inVitro_gene_names.txt', header=None).values[:, 0], dtype=str)




cell_barcodes = pd.read_csv('./GSE140802_data/GSM4185642_stateFate_inVitro_cell_barcodes.txt', header=None).values[:, 0]

metadata = pd.read_csv('./GSE140802_data/GSM4185642_stateFate_inVitro_metadata.txt', sep='\t')
metadata_barcodes = metadata['Cell barcode'].values
time_labels = metadata['Time point'].values
ct_labels = metadata['Cell type annotation'].values

full_labels = ct_labels + "-" + np.asarray(time_labels, dtype=str)

# Clone Matrix
# This is unique to this dataset, it's a CELL x CLONE_FAMILY matrix that says which cell is part of which clonal family (at most 1)
clone_mtx_ = sc.read_mtx('./GSE140802_data/GSM4185642_stateFate_inVitro_clone_matrix.mtx').X.toarray()

# Unlabelled with the lentivirus cells are marked to the zeroth barcode and everything else is shifted up 1
clone_mtx = np.zeros((clone_mtx_.shape[0], clone_mtx_.shape[1]+1))
clone_mtx[:, 1:] = clone_mtx_
mask_unbarcoded = (clone_mtx.sum(axis=-1) == 0)
clone_mtx[mask_unbarcoded, 0] = 1
cell_clone_label = np.argmax(clone_mtx, axis=-1)

# Reorganizing the labels
L_broad = np.asarray(ct_labels)
biological_order = ['Undifferentiated', 'Baso', 'Eos', 'Mast', 'Neutrophil', 'Erythroid', 'Meg',
                    'Monocyte', 'pDC', 'Ccr7_DC', 'Lymphoid']

ct_to_idx = {ct: i for i, ct in enumerate(biological_order)}
Ls = np.array([ct_to_idx[ct] for ct in L_broad])
ct_names = np.asarray(biological_order)

# full_labels
biological_order_full = [
    # --- Time 2.0 ---
    'Undifferentiated-2.0', 'Baso-2.0', 'Eos-2.0', 'Mast-2.0', 'Neutrophil-2.0',
    'Erythroid-2.0', 'Meg-2.0', 'Monocyte-2.0', 'pDC-2.0', 'Ccr7_DC-2.0', 'Lymphoid-2.0',

    # --- Time 4.0 ---
    'Undifferentiated-4.0', 'Baso-4.0', 'Eos-4.0', 'Mast-4.0', 'Neutrophil-4.0',
    'Erythroid-4.0', 'Meg-4.0', 'Monocyte-4.0', 'pDC-4.0', 'Ccr7_DC-4.0', 'Lymphoid-4.0',

    # --- Time 6.0 ---
    'Undifferentiated-6.0', 'Baso-6.0', 'Eos-6.0', 'Mast-6.0', 'Neutrophil-6.0',
    'Erythroid-6.0', 'Meg-6.0', 'Monocyte-6.0', 'pDC-6.0', 'Ccr7_DC-6.0', 'Lymphoid-6.0']

ct_to_idx_full = {ct: i for i, ct in enumerate(biological_order_full)}
Ls_full = np.array([ct_to_idx_full[ct] for ct in full_labels])
ct_names_full = np.asarray(biological_order_full)


# Getting clonal statistics - how each clone family is biased to each fate
N_clones = clone_mtx.T.shape[0]
N_types = len(ct_names)
clone_stats = np.zeros((N_clones, N_types)) # the type distribution for each clone id
for t_i in range(N_types):
    clone_stats[:, t_i] = clone_mtx.T @ (Ls==t_i)


# In case the data is too dense, subsample (only if testing on limited hardware)
data = adata.X.toarray()

# Verbose debug to see cell distribution and time distribution
for t_i in np.unique(time_labels):
    print(t_i, "="*10)
    for i in np.unique(Ls):
        print(np.sum(Ls[time_labels==t_i]==i), ct_names[i], "CT - (", i, ")")


print("="*10)
for i in np.unique(time_labels):
    print(np.sum(time_labels==i), i, "TIME : (", i, ")")



# Filter Confounding Genes
mask_out_mt = ~np.strings.startswith(np.strings.upper(gene_names), 'MT-')
mask_out_rps = ~np.strings.startswith(np.strings.upper(gene_names), 'RPS')
mask_out_rpl = ~np.strings.startswith(np.strings.upper(gene_names), 'RPL')

exclude_genes = [
    "MALAT1", "NEAT1", "XIST", "KCNQ1OT1", "FOS", "FOSB", "JUN", "JUNB",
    "EGR1", "EGR2", "EGR3", "ATF3", "DUSP1", "DUSP2", "IER2", "IER3",
    "HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "HLA-A", "HLA-B", "HLA-C", 
    "B2M", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1"
]
mask_out_specifics = ~np.isin(np.strings.upper(gene_names), exclude_genes)
mask_out_confounding = mask_out_mt & mask_out_rps & mask_out_rpl & mask_out_specifics

data = data[:, mask_out_confounding]
gene_names = gene_names[mask_out_confounding]
print("Filtered out", np.sum(~mask_out_confounding), "Confounding genes, MT, RPS, RPL and MATLAT1, ...")



# Data partitioning, to construct a balanced CT partition
N_per_class = 200 # 20*3*len(ct_names) # 200 
selected_labels_full = np.asarray([0, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]) # take early hsc, late everything else
selected_labels = np.arange(np.max(Ls)+1, dtype=int) # for now sample from everytime point
indx = np.arange(len(data), dtype=int)

indx_partitionned, Ls_partitionned_full = get_data_partition(indx, Ls_full, selected_labels_full,
                                                             N_per_class, replace=True, verbose=False, seed=SEED)

data_partitionned = data[indx_partitionned]
Ls_time = time_labels[indx_partitionned]
Ls_partitionned = Ls[indx_partitionned]
cell_clone_label_partitionned = cell_clone_label[indx_partitionned]

# WARNING STATISTICS NOT UNIQUE is dangerous but EASIER TO DEAL WITH
clone_stats_partitionned = clone_stats[cell_clone_label_partitionned] 

# --- Normalize using Pearson Residuals ---
data_residuals, data_partitionned, gene_names_filtered = compute_pearson_residuals(data_partitionned, gene_names)

# Find Highly Variable Genes (HVGs) 
D = np.sum(data_residuals**2, axis=0)
genes_high = np.argsort(D)[::-1]

# Plot a few for debugging
N_genes_plot = 60
expr_per_type = np.zeros((len(selected_labels), N_genes_plot))
for i in range(len(selected_labels)):
    expr_per_type[i] = np.mean(data_residuals[Ls_partitionned==selected_labels[i]], axis=0)[genes_high[:N_genes_plot]]


fig, ax = plt.subplots(1, 1, figsize=(16, 9))
im = ax.imshow(expr_per_type, cmap="bwr")
ax.set_yticks(np.arange(len(selected_labels)))
ax.set_yticklabels(ct_names[selected_labels], fontsize=16)
ax.set_xticks(np.arange(N_genes_plot)-0.5)
ax.set_xticklabels(gene_names_filtered[genes_high[:N_genes_plot]], rotation=60, fontsize=16)
plt.tight_layout()
import os; os.makedirs("tmp", exist_ok=True);
plt.savefig(f"tmp/PlotHDGCellTypes_{SEED}.png")


# Subsetting to top genes for PCA
N_genes_kept = 2000
mask_keep_genes = np.zeros((len(gene_names_filtered)), dtype=bool)
mask_keep_genes[genes_high[:N_genes_kept]] = True # keep top genes
print(f"In total {mask_keep_genes.sum()} genes kept")


# Keep these genes because they are used to screen for progenitors in the analysis
gene_markers = [
    "Kit", "Ly6a", "Cd34", "Fcgr3", "Itga2b", "Itgb3", "Slamf1", 
    "Il1rl1", "Itgb7", "Il3ra", "Itga2", "Il2rb", "Flt3", "Il7r", 
    "Ccr9", "Itgam", "Ly6g", "Ptprc", "Cd3g", "Gypa", "Fcer1a"
]
marker_mask = np.isin(np.char.upper(gene_names_filtered), np.char.upper(gene_markers))
print(f"{np.sum(marker_mask)} out of {len(gene_markers)} screening gene markers found")
matching_genes = gene_names_filtered[marker_mask]
print("Keeping ", matching_genes)
mask_keep_genes[marker_mask] = True


genes_names_high = gene_names_filtered[mask_keep_genes]
data_preprocessed = data_residuals[:, mask_keep_genes]

# PCA & NCA (the latter for visualization only)
pca = PCA(n_components=100, random_state=SEED).fit(data_preprocessed)
pca_components = pca.components_
print(np.cumsum(pca.explained_variance_ratio_))

embed_pca = pca.transform(data_preprocessed)

from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
nca = NCA(n_components=3, verbose=True, max_iter=80, random_state=SEED).fit(embed_pca, Ls_partitionned)
NCA_mtx = nca.components_
embed_nca = nca.transform(embed_pca) # for speed i subsampled

biological_order_colors = {
    # --- 1. Progenitors (Kept the same) ---
    'Undifferentiated': (0.05, 0.05, 0.05, 1.0),  # Light Grey 

    # --- 2. Erythroid & Megakaryocytic (Kept the same) ---
    'Erythroid':        (0.85, 0.05, 0.05, 1.0),  # Bright Solid Red
    'Meg':              (0.60, 0.30, 0.10, 1.0),  # Deep Brown

    # --- 3. Granulocytes (Maximum Contrast) ---
    'Neutrophil':       (1.00, 0.55, 0.00, 1.0),  # Bright Orange (Perfect for massive populations)
    'Eos':              (0.00, 0.80, 0.80, 1.0),  # Cyan / Bright Teal
    'Baso':             (0.95, 0.85, 0.10, 1.0),  # Strong Gold / Yellow
    'Mast':             (0.20, 0.80, 0.20, 1.0),  # The single True Green allowed

    # --- 4. Monocytes & Dendritic Cells (Pinks & Purples) ---
    'Monocyte':         (0.60, 0.40, 0.80, 1.0),  # Soft Purple / Lavender
    'pDC':              (1.00, 0.10, 0.60, 1.0),  # Vibrant Hot Pink
    'Ccr7_DC':          (0.750, 0.00, 0.750, 1.0),  # Very Dark Purple

    # --- 5. Lymphoid Lineage ---
    'Lymphoid':         (0.00, 0.30, 0.90, 1.0)   # Strong Solid Blue
}


ordered_colors = [biological_order_colors[name] for name in ct_names]
cmap = ListedColormap(ordered_colors, name='hematopoiesis_cmap')

# Plot NCA
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection='3d')
im = ax.scatter(embed_nca[:, 0], embed_nca[:, 1], embed_nca[:, 2], s=4, c=Ls_partitionned, cmap=cmap, vmin=0, vmax=len(ct_names))
cb = plt.colorbar(im)
cb.set_ticks(np.arange(len(ct_names))+0.5)
cb.set_ticklabels(ct_names)
ax.set_title("NCA")
#plt.show()
plt.savefig(f"tmp/nca_{SEED}.png")

# Plot PCA
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection='3d')
mask = ~np.isin(Ls_partitionned, [-1])
im = ax.scatter(embed_pca[mask, 0], embed_pca[mask, 1], embed_pca[mask, 2], s=4, c=Ls_partitionned[mask], cmap=cmap, vmin=0, vmax=len(ct_names))
cb = plt.colorbar(im)
cb.set_ticks(np.arange(len(ct_names))+0.5)
cb.set_ticklabels(ct_names)
ax.set_title("PCA")
plt.savefig(f"tmp/pca_{SEED}.png")


for i in np.unique(Ls_time):
    print(np.sum(Ls_time==i), i, "TIME : (", i, ")")


# Save
np.savez(f"data_preprocessed_{SEED}.npz",
         XI=data_preprocessed, XI_pca=embed_pca, XI_NCA=embed_nca,
         NCA_mtx=NCA_mtx, pca_components=pca_components, 
         Ls=Ls_partitionned, Ls_time=Ls_time, clone_stats_partitionned=clone_stats_partitionned,
         clone_stats=clone_stats, cell_clone_label_partitionned=cell_clone_label_partitionned,
         cmap=cmap, ct_names=ct_names, gene_names_high=genes_names_high,
         indx_partitionned=indx_partitionned, allow_pickle=True)
