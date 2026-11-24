import numpy as np, subprocess as sp, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
X = np.load("/localhome/nacer/data_4_wmu/Xs_saves.mmap", mmap_mode="r")[::10]; S = np.asarray(X[::100]); K = X.shape[1]

def init():  # runs once per worker: each worker builds its own figure
    global fig, sc
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    sc = ax.scatter(np.zeros(K), np.zeros(K), c=np.arange(K), cmap="viridis", s=1)
    ax.set_xlim(S[..., 0].min() - .1, S[..., 0].max() + .1); ax.set_ylim(S[..., 1].min() - .1, S[..., 1].max() + .1)

def render(t):
    sc.set_offsets(np.asarray(X[t])); fig.canvas.draw()
    return fig.canvas.buffer_rgba().tobytes()

if __name__ == "__main__":
    ff = sp.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "1000x800", "-r", "30", "-i", "-",
                   "-vcodec", "libx264", "-pix_fmt", "yuv420p", "cells_movie.mp4"], stdin=sp.PIPE)
    with Pool(30, init) as p:
        for i, f in enumerate(p.imap(render, range(0, len(X), 1), chunksize=2)):
            ff.stdin.write(f); i % 100 or print(i)
    ff.stdin.close(); ff.wait()