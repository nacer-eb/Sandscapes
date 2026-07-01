import sys
sys.path.append('../../highdim/common') # include the common folder

# SET GPU TO USE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tqdm

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(palette="pastel")


import numpy as np

import matplotlib.colors

colors = [    (0.8352941176, 0.3686274510, 0.0000000000, 1.0),  # 4: #D55E00 Vermilion
              (0.9019607843, 0.6235294118, 0.0000000000, 1.0),  # 1: #E69F00 Orange
              (0.3372549020, 0.7058823529, 0.9137254902, 1.0),  # 2: #56B4E9 Sky-blue
              (0.0000000000, 0.6196078431, 0.4509803922, 1.0),  # 3: #009E73 Bluish-green
          ]



common_cmap = matplotlib.colors.ListedColormap(colors)
common_norm = matplotlib.colors.Normalize(0, 4)




data = np.load("data_link/catastrophe_data.npz") #long_iso
N_fps_space = data["N_fps_space"]
gammas = data["gammas"]
size_gamma = data["size_gamma"]


data_inset = np.load("data_link/catastrophe_data_inset.npz") #
N_fps_space_inset = data_inset["N_fps_space"]
gammas_inset = data_inset["gammas"]
size_gamma_inset = data_inset["size_gamma"]



# gamma-Points of interest
gamma_points = np.asarray([[  0.01 ,   0.01 ],
                           [240.  ,  30. ],
                           [950.  ,  98],
                           [320.  , 240.  ],
                           [ 30.0 , 240.  ],
                           [ 98 , 950.  ],
                           [100.  , 100.  ],
                           [500, 15],
                           [15, 500]])
print(gamma_points)



# preprocess the FPs so that only odd numbers are kept
N_fps_space_pp = ((N_fps_space - 1)//2)
N_fps_space_pp_inset = ((N_fps_space_inset - 1)//2)





fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=400)

im=ax.pcolormesh(gammas[:, 0].reshape(size_gamma, size_gamma),
              gammas[:, 1].reshape(size_gamma, size_gamma),
                 N_fps_space_pp, cmap=common_cmap, norm=common_norm)

ax.set_axisbelow(False)
ax.grid(True, which="both", linewidth=0.8, alpha=0.25)

ax.scatter(*gamma_points.T, color="k", s=30)

# Increase fontsize for both x and y axes
ax.tick_params(axis='both', labelsize=16)



# inset Axes....
x1, x2, y1, y2 = 0.0, 50, 0.0, 50  # subregion of the original image
axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.set_xticks([])
axins.set_yticks([])

axins.pcolormesh(gammas_inset[:, 0].reshape(size_gamma_inset, size_gamma_inset),
                 gammas_inset[:, 1].reshape(size_gamma_inset, size_gamma_inset),
                 N_fps_space_pp_inset, cmap=common_cmap, norm=common_norm)

ax.indicate_inset_zoom(axins, edgecolor="black")

axins.scatter(*gamma_points[0], color="k", s=30)


# CBAR

cb = plt.colorbar(im, pad=0.005)
cb.set_ticks(np.arange(4)+0.5)
cb.set_ticklabels([1, 3, 5, 7], fontsize=16)
cb.set_label("Number of FPs", labelpad=20, fontsize=24)

ax.set_xlabel(r"$\tilde{z}_1$", fontsize=20, labelpad=-2)
ax.set_ylabel(r"$\tilde{z}_2$", fontsize=20, labelpad=-8)



ax.set_aspect(1.0)
plt.tight_layout()

import os
os.makedirs("Fig", exist_ok=True)

plt.savefig("Fig/catastrophe_plot_A_v2.png")
