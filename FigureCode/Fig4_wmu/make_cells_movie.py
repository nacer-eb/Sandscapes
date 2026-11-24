import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

X = np.load("Xs_saves.mmap", mmap_mode="r")[-2000:]  # (t, K, 2)
sample = np.asarray(X[::100])

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(np.zeros(X.shape[1]), np.zeros(X.shape[1]), c=np.arange(X.shape[1]), cmap="viridis", s=1)
sc.set_offsets(np.full((X.shape[1], 2), np.nan))
ax.set_xlim(sample[..., 0].min() - .1, sample[..., 0].max() + .1)
ax.set_ylim(sample[..., 1].min() - .1, sample[..., 1].max() + .1)

def upd(t): 
    if t%100==0: print(t)
    sc.set_offsets(np.asarray(X[t])); 
    return sc,
FuncAnimation(fig, upd, frames=range(0, len(X), 10), blit=False).save("cells_movie.mp4", fps=30)
