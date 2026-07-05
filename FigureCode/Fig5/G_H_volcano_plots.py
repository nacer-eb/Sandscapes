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

    if len(expr_A) > max_n:
        expr_A = expr_A[np.random.choice(len(expr_A), max_n, replace=False)]
    if len(expr_B) > max_n:
        expr_B = expr_B[np.random.choice(len(expr_B), max_n, replace=False)]

    _, p_array = scipy.stats.ttest_ind(expr_A, expr_B, equal_var=False)
    mean_diff = np.mean(expr_A, axis=0) - np.mean(expr_B, axis=0)
    n_p_value = -np.log10(np.clip(p_array, 1e-300, 1.0))

    cutoff = 4.2 * np.std(mean_diff)
    pvalue_min = 1.0

    mask_high = (n_p_value > pvalue_min) & (mean_diff > cutoff)
    mask_low = (n_p_value > pvalue_min) & (mean_diff < -cutoff)
    mask_selected = mask_high | mask_low

    fig = plt.figure(figsize=(9.8102, 9.1346))
    ax = fig.add_axes([0.05096, 0.08758, 0.82671, 0.85768])

    ax.scatter(mean_diff[~mask_selected], n_p_value[~mask_selected], color="lightgray", alpha=0.5)
    ax.scatter(mean_diff[mask_high], n_p_value[mask_high], color=prog_gates[pop_A]["color"], s=150)
    ax.scatter(mean_diff[mask_low], n_p_value[mask_low], color=prog_gates[pop_B]["color"], s=150)

    ax.set_xlim(-np.max(np.abs(mean_diff)) * 1.1, np.max(np.abs(mean_diff)) * 1.1)
    ax.set_ylim(10, np.max(np.abs(n_p_value)) * 1.1)

    texts = [
        ax.text(mean_diff[i], n_p_value[i], gene_names_high[i], fontsize=28)
        for i in np.argwhere(mask_selected).ravel()
    ]

    adjust_text(
        texts,
        force_text=(1.2, 6.0),
        force_points=(1.2, 4.0),
        expand_points=(1.1, 1.5),
        expand_text=(1.1, 2.0),
        max_iterations=10000,
        arrowprops=dict(arrowstyle="-", color="gray", lw=1.5, alpha=0.7)
    )

    ax.axvline(cutoff, color="red", linestyle="--")
    ax.axvline(-cutoff, color="red", linestyle="--")
    ax.axhline(pvalue_min, color="red", linestyle="--")

    ax.set_xlabel("Mean Difference" if pop_A != "CLP" else "")
    if pop_A == "CLP":
        ax.set_xticks([])

    ax.set_yticks(np.arange(10, np.max(np.abs(n_p_value)) * 1.1, 25))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("-log10(p-value)")

    plt.savefig(f"Fig/GH_Volcano_pop_{pop_A}_{pop_B}.png")
    plt.close()

    
