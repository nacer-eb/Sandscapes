# SET GPU TO USE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tqdm

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(palette="pastel")

import scipy.spatial.distance as distance

import jax
# DISABLED WARNING --- Enable float64 globally
jax.config.update("jax_enable_x64", True) # WARNING !!!!!!! WARNING !!!!!!!

import jax.numpy as jnp


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

d_x = 2.0
d_y = 2.6
Is = jnp.asarray([ [-d_x/2.0, 0], [d_x/2.0, 0], [0, d_y] ])

K = 100
def simulate_tanh_sys(X_0, d_x, d_y, beta, tmax, lr, noise_level, keys, tbeta_range):
    def iteration(x_vec, t):
        x_n, y_n = x_vec.T
        gamma_1 = jnp.sum(jnp.exp(-beta*(x_n-d_x/2.0)**2) * jnp.exp(-beta*(y_n)**2))
        gamma_2 = jnp.sum(jnp.exp(-beta*(x_n+d_x/2.0)**2) * jnp.exp(-beta*(y_n)**2))
        gamma_3 = jnp.sum(jnp.exp(-beta*(x_n)**2) * jnp.exp(-beta*(y_n-d_y)**2))


        theta_2 = 0.5*jnp.log(gamma_2/gamma_1)
        theta_3 =  0.5*jnp.log(gamma_3/gamma_1)

        phi_n = 0.5 * jnp.log( (jnp.exp(-beta*(x_n-d_x/2.0)**2) + (gamma_1/gamma_2)*jnp.exp(-beta*(x_n+d_x/2.0)**2) )/jnp.exp(-beta*(x_n)**2) )

        k_n = 1.0/(1.0 + (gamma_1/gamma_3)*(jnp.exp(-beta*(x_n)**2) * jnp.exp(-beta*(y_n-d_y)**2) / (jnp.exp(-beta*(x_n-d_x/2.0)**2) * jnp.exp(-beta*(y_n)**2) + jnp.exp(-beta*(x_n+d_x/2.0)**2) * jnp.exp(-beta*(y_n)**2)  )) )

        dt_x = k_n*jnp.tanh( beta*d_x*x_n + theta_2 )*0.5*d_x - x_n
        dt_y = jnp.tanh( beta*d_y*( y_n - 0.5*d_y) - theta_3 - phi_n  ) * 0.5*d_y - (y_n - 0.5*d_y)

        noise = jax.random.normal(keys[t], shape=X_0.shape) * (1.0/10.0)**noise_level
        x_n_new = x_n + lr * dt_x + noise[:, 0]
        y_n_new = y_n + lr * dt_y + noise[:, 1]

        x_vec_new = jnp.asarray([x_n_new, y_n_new]).T
        return x_vec_new, x_vec_new

    _, Xs_saves_tanh = jax.lax.scan(iteration, X_0, tbeta_range)

    return Xs_saves_tanh



size = 200
xs = np.linspace(-d_y/2.0-0.1, d_y/2.0+0.1, size)
ys = np.linspace(-0.1, d_y+0.1, size)
Ms_probes_x, Ms_probes_y = np.meshgrid(xs, ys)
Ms_probes = np.asarray([Ms_probes_x.ravel(), Ms_probes_y.ravel()]).T


#Is_colors = np.zeros((len(Is), 4))
cmap_tab10 = plt.get_cmap("tab10")
norm_tab10 = matplotlib.colors.Normalize(0, 10)
#Is_colors = cmap_tab10(norm_tab10([1, 4, 2])) # next would be 6
Is_colors = np.asarray([[176.0, 213.0, 173.0, 255.0], [141, 211.0, 242.0, 255.0], [250.0, 205.0, 211.0, 255.0]])/255.0

X_0 = jnp.zeros((K, 2)) + jnp.mean(Is, axis=0)

tmax = 800
lr= 0.03
noise_level = 7

seed = 42
key = jax.random.key(seed)
keys = jax.random.split(key, (tmax))
tbeta_range = jnp.arange(tmax, dtype=int)

def plot_ax(ax, X_0, d_x, d_y, beta, tmax, lr, noise_level, keys, tbeta_range, plot_y):
    Ms_saves = simulate_tanh_sys(X_0, d_x, d_y, beta, tmax, lr, noise_level, keys, tbeta_range)

    tmax, K, N = Ms_saves.shape

    ax.set_ylabel("y", labelpad=10)
    ax.set_xlabel("x", labelpad=5)
    ax.set_xlim(-d_y/2.0-0.1, d_y/2.0+0.1)
    ax.set_ylim(-0.1, d_y+0.1)

    ax.scatter(*Is.T, marker="*", color=Is_colors, s=600, edgecolor="k", lw=1.5, zorder=20)

    space_colors = np.exp(-beta*distance.cdist(Ms_probes, Is)**2)@Is_colors
    space_colors /= space_colors[:, -1][:, None]
    space_colors[:, -1] *= 0.9

    ax.pcolormesh(Ms_probes[:, 0].reshape(size, size), Ms_probes[:, 1].reshape(size, size), space_colors.reshape(size, size, -1), zorder=-2)

    # Setting grid to be lighter
    st = sns.axes_style()
    base = st["grid.color"]
    grid_rgba = matplotlib.colors.to_rgba(base, alpha=0.3)
    ax.grid(True, color=grid_rgba, linewidth=0.4)


    for k in range(K):
        ax.scatter(Ms_saves[0, k, 0], Ms_saves[0, k, 1], color="k", alpha=1.0, lw=0.2)
        ax.plot(Ms_saves[:, k, 0], Ms_saves[:, k, 1], color="k", alpha=0.5, lw=0.2)

    if not plot_y:
        ax.set_ylabel("")



fig, axs = plt.subplots(1, 3, figsize=(21, 7), sharey=True)

for i, beta in enumerate([1.5, 3.0, 11.0]):
    plot_ax(axs[i], X_0, d_x, d_y, beta, tmax, lr, noise_level, keys, tbeta_range, i==0)


plt.tight_layout()

import os
os.makedirs("Fig", exist_ok=True)

plt.savefig("Fig/C_approx_triplot.png", bbox_inches="tight")


