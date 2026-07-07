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

data = np.load("../../datasets/mnist/data_mnist.npz", allow_pickle=True)
alpha_components = data["alpha_components"]

alphas = Ms_saves @ alpha_components
dt = 10
start_t = 400
end_t = min(3000, alphas.shape[0])

alphas_sub = alphas[start_t:end_t:dt]
tmax_sub, K, N = alphas_sub.shape
alphas_flat = alphas_sub.reshape(tmax_sub * K, N)
labels_sub = Ls_m[start_t:end_t:dt].ravel()

reducer = umap.UMAP(n_components=2, n_neighbors=1000, verbose=True, low_memory=False) # random_state disallows multithreading
embed = reducer.fit_transform(alphas_flat)

fig = vedo.Plotter(axes=0, pos=(1366, 0), size=(1920, 1080), interactive=False, offscreen=False)
cells = vedo.Points(embed, r=4.0).cmap("tab10", labels_sub, vmin=0, vmax=9)
cells.add_scalarbar(title="Class", nlabels=10)

cam = dict(
    pos=(2.03635, 2.46198, 64.0612),
    focal_point=(2.03635, 2.46198, 0.0),
    viewup=(0, 1, 0),
    parallel_projection=True,
)

fig.show(cells, camera=cam)
import os; os.makedirs("fig/", exist_ok=True)
fig.screenshot("fig/D_mnist_umap.png")
fig.close()
