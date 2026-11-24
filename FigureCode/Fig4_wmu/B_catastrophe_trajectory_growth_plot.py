import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
#sns.set_theme(palette="pastel")




tmin = 0
#data_h = np.load("/localhome/nacer/SignalDiff_data/SLURM/run_save_growth_h_q4.0_seed4.npz")
data_h = np.load("run_save_growth.npz")
Ss_saves_h = data_h['Ss_saves'][tmin:70000]#100000
tmax = len(Ss_saves_h)

gamma_points_h = np.asarray([1500, 8200, tmax-1])
print(gamma_points_h)

z_h = Ss_saves_h/Ss_saves_h[:, -1][:, None]
print(z_h[gamma_points_h])

fig, ax = plt.subplots(figsize=(16, 9), dpi=200)

ts = np.arange(tmax)
ax.plot(ts, z_h[:, 1], color="white", lw=10, zorder=9)
ax.scatter(ts[gamma_points_h], z_h[gamma_points_h, 1], c=[(183/255.0, 183/255.0, 183/255.0)], s=400, zorder=10)

#gamma_vlines = np.unique(np.r_[np.arange(0, tmax-1, 10000, dtype=int)[[0, 1, 2, 4, 6]], tmax-1])
gamma_vlines = np.unique(np.r_[np.arange(0, tmax-1, 8000, dtype=int)[[0, 1, 2, 4, 6]], tmax-1])
print(gamma_vlines)
for vline in gamma_vlines[:-1]:
    ax.axvline(vline, color="grey", lw=1)

ax.set_ylabel(r"$\tilde{z}_2$", fontsize=34)

y_0 = np.mean(z_h[-10000:, 1])

#dy = 
ymin = y_0 - 50
ymax = y_0 + 60


ax.locator_params(axis='x', nbins=20)
ax.spines["bottom"].set_position(("data", y_0))
ax.tick_params(axis="x", bottom=False, labelbottom=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
#ax.grid(False)

for side in ["left", "bottom"]:
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(2.5)

ax.tick_params(axis="y", colors="black", width=2.5, length=8, labelsize=28)
ax.tick_params(axis="x", colors="black", width=2.5, length=8)


ax.yaxis.label.set_color("black")
ax.xaxis.label.set_color("black")

ax.set_xlim(0, tmax*1.05)
ax.set_ylim(ymin, ymax)

plt.tight_layout()
#plt.show()
plt.savefig("Fig/B_catastrophe_growth_plot.png", transparent=True)


