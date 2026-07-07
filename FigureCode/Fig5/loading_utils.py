import numpy as np

def load_analysis_data(datafile="../../datasets/weinreb/data_preprocessed_all_t_markers.npz",
                       runname="data_weinreb/worker_0_detail.npz",
                       runname_nca="data_weinreb/worker_0.npz", use_full_nca=True):
    
    # Load NCA_mtx, pca, pca_inv and gene_names from the dataset
    data = np.load(datafile, allow_pickle=True)
    NCA_mtx = data['NCA_mtx'].T
    pca = data["pca_components"]
    pca_inv = np.linalg.pinv(pca)
    gene_names_high = data["gene_names_high"]

    XI_NCA = data['XI_NCA']
    Ls = data['Ls']
    ct_names = data['ct_names']

    # Load saved units
    data_run = np.load(runname)
    Ms_saves_cpu_detail = data_run["Ms_saves_cpu_detail"]
    Ms_saves_m1k = Ms_saves_cpu_detail[-5][50*2000:-1]
    Ms_saves_m1k_nca = Ms_saves_m1k @ NCA_mtx

    if use_full_nca:
        Ms_saves_cpu_nca = np.load(runname_nca)['Ms_saves_cpu_nca']


    # Restructure markers
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
    

    # Lineage filter
    lin_bounds = {
        "Itgam": [-0.7923, -0.1243],
        "Cd3g": [-0.6321, 0.0455],
        "Gypa": [-0.2647, 0.1902],
        "Fcer1a": [-0.4509, -0.0226],
    }
    lin_mask = data_matrix_markers[:, ordered_genes.index("Itgam")] < lin_bounds["Itgam"][1]
    for marker, bounds in lin_bounds.items():
        lin_mask &= data_matrix_markers[:, ordered_genes.index(marker)] < bounds[1]

    # Kit filter
    kit_bound = -0.18
    kit_mask = data_matrix_markers[lin_mask, ordered_genes.index("Kit")] > kit_bound

    # Progenitor gates post LK
    prog_gates = {
        "MEP": {"Slamf1": [-0.025, -0.00], "Fcgr3": [-0.35, -0.32], "color": "#D49A89"},
        "GMP": {"Slamf1": [-0.0719, -0.039], "Fcgr3": [-0.2085, -0.169], "color": "#A9D1B3"},
        "CMP": {"Slamf1": [-0.0719, -0.039], "Fcgr3": [-0.32, -0.29], "color": "#EBD88A"},
        "CLP": {"Slamf1": [-0.0719, -0.039], "Fcgr3": [-0.4276, -0.3734], "color": "#A3C2E0"},
    }

    slamf1_idx = ordered_genes.index("Slamf1")
    fcgr3_idx = ordered_genes.index("Fcgr3")
    expr_LK_markers = data_matrix_markers[lin_mask][kit_mask]
    expr_LK_full = (Ms_saves_m1k @ pca_inv.T)[lin_mask][kit_mask]
    X_LK_nca = Ms_saves_m1k_nca[lin_mask, :3][kit_mask]

    pop_masks = {}
    for pop_name, params in prog_gates.items():
        mask_slamf1 = (expr_LK_markers[:, slamf1_idx] >= params["Slamf1"][0]) & (expr_LK_markers[:, slamf1_idx] <= params["Slamf1"][1])
        mask_fcgr3 = (expr_LK_markers[:, fcgr3_idx] >= params["Fcgr3"][0]) & (expr_LK_markers[:, fcgr3_idx] <= params["Fcgr3"][1])
        pop_masks[pop_name] = mask_slamf1 & mask_fcgr3

    return {
        "Ms_saves_cpu_nca": Ms_saves_cpu_nca,
        "Ms_saves_m1k": Ms_saves_m1k,
        "Ms_saves_m1k_nca": Ms_saves_m1k_nca,
        "pca_inv": pca_inv,
        "gene_names_high": gene_names_high,
        "ordered_genes": ordered_genes,
        "lin_mask": lin_mask,
        "kit_mask": kit_mask,
        "expr_LK_markers": expr_LK_markers,
        "expr_LK_full": expr_LK_full,
        "X_LK_nca": X_LK_nca,
        "prog_gates": prog_gates,
        "pop_masks": pop_masks,
        "XI_NCA": XI_NCA,
        "Ls": Ls,
        "ct_names": ct_names
    }
