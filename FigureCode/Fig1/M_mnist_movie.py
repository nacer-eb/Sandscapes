import numpy as np
import matplotlib.pyplot as plt
import umap
import vedo

output_dir = "data_mnist"
run_id = 0
runname = f"{output_dir}/worker_{run_id}"

data_run = np.load(runname + ".npz", allow_pickle=True)
Ms_saves = data_run["Ms_saves"]
Ls_m = data_run["Ls_m"]

# Sort by final class just to make it easier visually
sorted_idx = np.argsort(Ls_m[-1])
Ms_saves = Ms_saves[:, sorted_idx]

data = np.load("../../datasets/mnist/data_mnist.npz", allow_pickle=True)
alpha_components = data["alpha_components"]
pca_components = data["pca_components"]

pca_inv = np.linalg.pinv(pca_components)
NCA_mtx = data['NCA_mtx'].T
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = int(np.sqrt(Ms_saves.shape[1]))
grid = (Ms_saves)[::5].reshape(-1, N, N, 28, 28).transpose(0, 1, 3, 2, 4).reshape(-1, N * 28, N * 28)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(grid[0], cmap="bwr", vmin=grid.min(), vmax=grid.max())
ax.axis("off")

def update(frame):
    print(frame)
    im.set_array(grid[frame])
    return [im]

ani = FuncAnimation(fig, update, frames=len(grid), interval=50, blit=True)
ani.save("mnist_movie_1.mp4")
plt.show()
