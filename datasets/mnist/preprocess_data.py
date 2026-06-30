import sys
sys.path.append('../../common') 

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

from tensorflow.keras.datasets import mnist

(X_train, L_train), (X_test, L_test) = mnist.load_data()
X_train = np.asarray(X_train.reshape(-1, 784), dtype=float)#*2-1

Xs = X_train.reshape(-1, 784) * 1.0/np.max(X_train)

print(np.min(Xs), np.max(Xs)) # 1.0


Ls = L_train


# Initial data partitioning, to construct a balanced CT partition
N_per_class = 60  #2000 #
selected_labels = np.arange(np.max(L_train)+1, dtype=int) # for now sample from everytime point
data_partitionned, Ls_partitionned = get_data_partition(Xs, Ls, selected_labels,
                                                             N_per_class, replace=True, verbose=False, seed=42)


alpha_components = np.linalg.pinv(data_partitionned[:, :784], rcond=1E-5).reshape(784, 10, -1).sum(axis=-1)
embed_alpha = (data_partitionned[:, :784] @ alpha_components)

#embed_pca = data_partitionned
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
nca = NCA(n_components=2, verbose=True, max_iter=100, random_state=420, init='random', tol=1E-10).fit(embed_alpha, Ls_partitionned)
NCA_mtx = nca.components_
embed_nca = nca.transform(embed_alpha)

pca = PCA(n_components=3, random_state=41).fit(embed_alpha)
pca_components = pca.components_
print(np.cumsum(pca.explained_variance_ratio_))
embed_pca = pca.transform(embed_alpha)


# Plot NCA
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection='3d')
im = ax.scatter(embed_nca[:, 0], embed_nca[:, 1], embed_nca[:, -1], s=24, c=Ls_partitionned, cmap="tab10",
                vmin=0, vmax=10)
cb = plt.colorbar(im)
cb.set_ticks(np.arange(10)+0.5)
cb.set_ticklabels(np.arange(10))
ax.set_title("NCA")
plt.show()


# Plot NCA
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection='3d')
im = ax.scatter(embed_pca[:, 0], embed_pca[:, 1], embed_pca[:, -1], s=24, c=Ls_partitionned, cmap="tab10",
                vmin=0, vmax=10)
cb = plt.colorbar(im)
cb.set_ticks(np.arange(10)+0.5)
cb.set_ticklabels(np.arange(10))
ax.set_title("PCA")
plt.show()


# Save
np.savez("data_mnist_tmp.npz", XI=data_partitionned, XI_pca=embed_pca, XI_alpha=embed_alpha, XI_NCA=embed_nca,
         NCA_mtx=NCA_mtx, pca_components=pca_components, alpha_components=alpha_components, Ls=Ls_partitionned, allow_pickle=True)
