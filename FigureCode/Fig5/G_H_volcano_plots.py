import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
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

comparisons = [
    ("CMP", "CLP"),
    ("MEP", "GMP")
]

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
    ax.set_ylim(10, np.max(np.abs(n_p_value))*1.1)
    
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

    ax.set_yticks(np.arange(10, np.max(np.abs(n_p_value))*1.1, 25))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("-log10(p-value)")


    #plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    plt.savefig(f"Fig/GH_Volcano_pop_{pop_A}_{pop_B}.png")
    plt.close()

