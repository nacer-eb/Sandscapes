import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(palette="pastel")

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=SMALL_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=SMALL_SIZE)    
plt.rc('ytick', labelsize=SMALL_SIZE)    
plt.rc('legend', fontsize=SMALL_SIZE)    
plt.rc('figure', titlesize=BIGGER_SIZE)



data = np.load("run_save.npz")
Xs_saves = data['Xs_saves']
K = data['K']
XIs = data['XIs']
beta = data['beta']
gamma = data['gamma']
tmax = len(Xs_saves)

data = np.load("energy_save.npz")
manifold_grads = data["manifold_grads"]
x = data["x"]
y = data["y"]

ts = np.asarray([0, 30, 70, 100, 150, 260, 580, 900], dtype=int)

size = len(x)
XIs_colors = np.asarray([[176.0, 213.0, 173.0, 255.0], [141, 211.0, 242.0, 255.0], [250.0, 205.0, 211.0, 255.0]])/255.0

fig, axs = plt.subplots(2, 4, figsize=(16, 8), dpi=200, sharex=True, sharey=True)

for i, t in enumerate(ts):
    axs.ravel()[i].scatter(Xs_saves[t, :, 0], Xs_saves[t, :, 1], s=80, edgecolor="k", facecolor="w", alpha=0.2, zorder=21)
    axs.ravel()[i].scatter(*XIs.T, marker="*", color=XIs_colors, s=600, edgecolor="k", lw=1.5, zorder=20)


    axs.ravel()[i].streamplot(x, y, manifold_grads[t, :, 0].reshape(size, size), manifold_grads[t, :, 1].reshape(size, size),
                              color=(0.1, 0.1, 0.1, 0.2))
    
    axs.ravel()[i].set_xlim(np.min(x), np.max(x))
    axs.ravel()[i].set_ylim(np.min(y), np.max(y))
    axs.ravel()[i].set_aspect(1.0)

axs.ravel()[0].set_ylabel(r"$y_n$"); axs.ravel()[4].set_ylabel(r"$y_n$")

for j in range(4, len(ts)):
    axs.ravel()[j].set_xlabel(r"$x_n$")


plt.tight_layout()
plt.savefig("fig/S_flow_snapshots.png")


