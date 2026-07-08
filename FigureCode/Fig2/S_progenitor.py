import sys
sys.path.append('../../common') # include the common folder


import tqdm

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import seaborn as sns
sns.set_theme(palette="pastel")

import scipy.spatial.distance as distance

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np
from sandscapes_main import *

import numpy as np
import tqdm

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(palette="pastel")

import scipy.spatial.distance as distance

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np



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


d_x = 2.0
d_y = 2.6
Is = jnp.asarray([ [-d_x/2.0, 0], [d_x/2.0, 0], [0, d_y] ])
K = 400

X_0 = jnp.zeros((K, 2)) + jnp.mean(Is, axis=0)
tmax = 800
lr= 0.03
noise_level = 7

seed = 42
key = jax.random.key(seed)
keys = jax.random.split(key, (tmax))
gamma = 0


def calc_y_tau_eps(beta, d_x, d_y, x_0=10**(-7), eps=0.1):

    t_A = -(1.0/(0.5*beta*(d_x**2)-1.0)) * jnp.log( 1.0/(beta*d_x*x_0) )
    t_B = -1.0 * jnp.log( (d_x/2.0-1.0/(beta*d_x)) / ( d_x/2.0 - eps ) )
    
    y_n_t_eps = (1.0/3.0) * d_y * jnp.exp(
        t_A + t_B
    )

    return y_n_t_eps #- d_y/2.0


betas = np.arange(1.5, 13.0, 1.0)
betas_lin = np.linspace(1.5, 13.0, 100)

print('TEST')
y_tau_eps = calc_y_tau_eps(betas_lin, d_x, d_y)

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(betas_lin, y_tau_eps, color="k", linewidth=6)

eps=0.1
for beta in betas:
    Ms_saves = training_loop(X_0, X_0, Is, jnp.asarray([beta]), gamma, tmax, lr, noise_level, keys)
    mask = np.abs(Ms_saves[-1, :, 0])>0.3
    ind = np.argmax(np.abs(Ms_saves[:,  mask, 0]) > eps, axis=0)
    y = np.mean(Ms_saves[:, mask][ind, np.arange(np.sum(mask)), 1])

    ax.scatter(beta, y, color="red", s=20)
    
ax.set_ylim(0, d_y/3.0)
ax.set_xlabel(r"$\beta$"); ax.set_ylabel("Intermediate state location (y)")
plt.savefig("Fig/S_progenitor.png")


#
