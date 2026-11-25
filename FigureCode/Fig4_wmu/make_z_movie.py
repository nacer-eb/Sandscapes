import numpy as np, subprocess as sp, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mc, seaborn as sns
from multiprocessing import Pool
from mpl_toolkits.axes_grid1 import make_axes_locatable
sns.set_theme(palette="pastel")

z = np.load("/localhome/nacer/data_4_wmu/run_save_growth.npz")["Ss_saves"][::1]; 
#z = np.load("../Fig4/run_save_growth.npz")["Ss_saves"][::10]; 
z = z / z[:, -1][:, None]

d = np.load("data_catastrophe_2/catastrophe_data_growth.npz")
zm, n = d["z_mus"], int(d["size_gamma"])
w_mu = d["w_mu"]
zm = zm


z = z / w_mu[None, :]

N_pp = np.ceil((d["N_fps_space"] - 1) / 2.0)  # keep only odd FP counts
cmap = mc.ListedColormap(["#D55E00", "#E69F00", "#56B4E9", "#009E73"])
DPI = 150  # 400 gives 6000x2400 frames (~58 MB each); raise it if you really need it

def init():  # runs once per worker: builds the full static figure
    global fig, ln
    fig, ax = plt.subplots(figsize=(15, 6), dpi=DPI)
    im = ax.pcolormesh(zm[:, 0].reshape(n, n), zm[:, 1].reshape(n, n), N_pp, cmap=cmap, norm=mc.Normalize(0, 4))
    ax.set_axisbelow(False); ax.grid(True, which="both", linewidth=0.8, alpha=0.25); ax.tick_params(labelsize=16)
    cb = fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05))
    cb.set_ticks(np.arange(4) + 0.5); cb.set_ticklabels([1, 3, 5, 7], fontsize=16); cb.set_label("Number of FPs", labelpad=20, fontsize=24)
    ax.set_xlabel(r"$\tilde{z}_1$", fontsize=24); ax.set_ylabel(r"$\tilde{z}_2$", fontsize=24)
    ax.set_xlim(z[:, 0].min(), z[:, 0].max()); ax.set_ylim(z[:, 1].min(), z[:, 1].max()); ax.set_aspect("auto")
    fig.tight_layout(); ln, = ax.plot([], [], ".", color="crimson")
    return fig

def render(t):  # trail of the last 100 points, same as your t>100 logic
    ln.set_data(z[max(0, t - 100):t, 0], z[max(0, t - 100):t, 1]); fig.canvas.draw()
    return fig.canvas.buffer_rgba().tobytes()

if __name__ == "__main__":
    print("Multiprocessed")
    W, H = init().canvas.get_width_height()
    ff = sp.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "z_movie.mp4"], stdin=sp.PIPE)
    with Pool(30, init) as p:
        for i, f in enumerate(p.imap(render, range(0, len(z), 10), chunksize=2)):
            ff.stdin.write(f); i % 100 or print(i)
    ff.stdin.close(); ff.wait()