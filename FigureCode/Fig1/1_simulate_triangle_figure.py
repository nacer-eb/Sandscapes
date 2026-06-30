import seaborn as sns
sns.set_theme(palette="pastel")

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance

data = np.load("run_save.npz")
Xs_saves = data['Xs_saves']
K = data['K']
XIs = data['XIs']
beta = data['beta']
gamma = data['gamma']

tmax = len(Xs_saves)

## More pop colors
XIs_colors = np.array([
    [120, 190, 120],   # greener, slightly darker
    [ 90, 170, 230],   # more saturated blue
    [240, 150, 170],   # stronger pink
]) / 255.0


# Plot #
import mpl_toolkits
def draw_rect(ax, w, h, center=[0, 0, 0], color = [0.95, 0.95, 0.95, 1.0], zorder=1):
    vert_rect = [np.asarray([[0, -w, -h], [0, +w, -h], [0, +w, +h], [0, -w, +h]]) + np.asarray(center)[None, :]]

    ecolor = np.asarray(color)
    ecolor[-1] = 1.0

    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(vert_rect, facecolors=color, edgecolor=ecolor, zorder=zorder)
    poly.set_zsort("average")
    ax.add_collection3d(poly)




fig = plt.figure(figsize=(16, 9), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)

xmin, xmax = -1.75, 1.75
ymin, ymax = -1.75, 1.75


k_s = 0.5
panel_color = np.asarray(ax.xaxis.pane.get_facecolor())
ax.yaxis.pane.set_facecolor((0.85, 0.85, 0.85, 0.5))
shadow_color = np.asarray([0.75, 0.75, 0.75, 1.0])*k_s + (1-k_s)*panel_color


particle_colors_coefs = np.exp(-beta*distance.cdist(Xs_saves.reshape(-1, 2), XIs)**2).reshape(tmax, -1, len(XIs))
particle_colors_coefs = particle_colors_coefs/np.sum(particle_colors_coefs, axis=-1)[:, :, None]
particle_colors = np.ones((tmax, Xs_saves.shape[1], 4))
particle_colors[:, :, :-1] = np.asarray(particle_colors_coefs@XIs_colors, copy=True)
print(particle_colors.shape)


ts = np.linspace(0, 1, len(Xs_saves))
ts_i = np.asarray([0, 260, 580, 900, len(Xs_saves)])# + 380+50

# Draw one surface per timepoint and zorder it properly
for j in range(4):
    draw_rect(ax, (xmax-xmin)/2.0, (xmax-xmin)/2.0, [ts[ts_i[j]], 0.0, 0.0], color=(0.35, 0.35, 0.35, 0.5), zorder=5.5-j)

for k in range(K):
    mem_traces = ax.plot(ts[:ts_i[0]], Xs_saves[:ts_i[0], k, 0], Xs_saves[:ts_i[0], k, 1],  lw=1.75, alpha=0.5,
                         color=particle_colors[-1, k], zorder=6, rasterized=True)
    for j in range(4):
        mem_traces = ax.plot(ts[ts_i[j]:ts_i[j+1]], Xs_saves[ts_i[j]:ts_i[j+1], k, 0], Xs_saves[ts_i[j]:ts_i[j+1], k, 1],  lw=1.75, alpha=0.5,
                             color=particle_colors[-1, k], zorder=5-j, rasterized=True)


    # Shadows
    mem_traces = ax.plot(ts, xmax, Xs_saves[:, k, 1],  lw=0.5, color=shadow_color, zorder=-1, rasterized=True)
    mem_traces = ax.plot(ts, Xs_saves[:, k, 0], ymin,  lw=0.5, color=shadow_color, zorder=-1, rasterized=True)


ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


ax.set_ylim(xmin, xmax)
ax.set_zlim(ymin, ymax)
ax.set_xlim(0, 1)

ax.view_init(elev= 23.11622534215244 , azim= -100.70988246540698 , roll= -2.590825529221398 )


ax.set_facecolor("None")
ax.set_box_aspect([4.0, 1.0, 1.2])
plt.tight_layout()

import os
os.makedirs("fig", exist_ok=True)

plt.savefig("fig/1_simulate_triangle_image_"+str(beta)+"_v2.svg")
#plt.show()
plt.close()

print("ax.view_init(elev=", ax.elev, ", azim=", ax.azim, ", roll=", ax.roll,")")
print("ax.set_xlim(",ax.get_xlim(),")")
print("ax.set_ylim(",ax.get_ylim(),")")
print("ax.set_zlim(",ax.get_zlim(),")")







