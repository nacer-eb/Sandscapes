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

np.random.seed(40)
final_labels = Ls_m[-1]
selected_indices = []

for digit in range(10):
    valid_indices = np.where(final_labels == digit)[0]
    if len(valid_indices) > 0:
        chosen_idx = np.random.choice(valid_indices)
        selected_indices.append(chosen_idx)
    else:
        selected_indices.append(0) # if the digit is not represented in the final population; just in case


tmax_total = Ms_saves.shape[0]
save_dt = 100

tab10 = plt.get_cmap("tab10")
dark_colors = [np.array(tab10(i)[:3]) * 0.85 for i in range(10)]

for t in range(0, tmax_total, save_dt):
    images_to_stack = []
    for digit, idx in enumerate(selected_indices):
        img = np.clip(Ms_saves[t, idx].reshape(28, 28), 0, 1)
        
        img_rgb = np.ones((28, 28, 3))
        for c in range(3):
            img_rgb[:, :, c] = 1.0 - img * (1.0 - dark_colors[digit][c])
            
        images_to_stack.append(img_rgb)
        
    stacked_image = np.vstack(images_to_stack)
    
    plt.figure(figsize=(3, 15))
    plt.imshow(stacked_image)
    plt.axis("off")

    import os; os.makedirs("mnist_snapshots", exist_ok=True)
    filename = f"mnist_snapshots/mnist_stacked_t{t:04d}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

