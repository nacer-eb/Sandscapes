import sys
sys.path.append("../../common")

import seaborn as sns
sns.set_theme(palette="pastel")

import tqdm

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from sandscapes_main import *


d_x = 2.0
d_y = 3.6

XI = jnp.asarray([[d_x/2.0, 0], [-d_x/2.0, 0], [0, d_y]])

N_c = 50
X_0 = jnp.zeros((N_c, 2)) + jnp.mean(XI, axis=0)

betas = jnp.asarray([[4.0], [13.0], [25.0]])
gamma = jnp.zeros((3))

tmax = 2000
noise_level = 7

seed = 43
key = jax.random.key(seed)
keys = jax.random.split(key, (tmax))
tbeta_range = jnp.arange(tmax, dtype=int)

lr= 0.03



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



def simulate_tanh_sys_approx(X_0, d_x, d_y, beta, tmax, lr, noise_level, keys, tbeta_range):


    #X_0 = X_0.at[:, -1].set(0.5*d_y -  1.0/10**noise_level)
    def iteration(x_vec, t):
        x_n, y_n = x_vec.T
        gamma_1 = jnp.sum(jnp.exp(-beta*(x_n-d_x/2.0)**2) * jnp.exp(-beta*(y_n)**2))
        gamma_2 = jnp.sum(jnp.exp(-beta*(x_n+d_x/2.0)**2) * jnp.exp(-beta*(y_n)**2))
        gamma_3 = jnp.sum(jnp.exp(-beta*(x_n)**2) * jnp.exp(-beta*(y_n-d_y)**2))

        theta_2 = 0.0
        theta_3 = 0

        phi_n = 0
        k_n = 1.0

        dt_x = k_n*jnp.tanh( beta*d_x*x_n + theta_2 )*0.5*d_x - x_n
        dt_y = jnp.tanh( beta*d_y*( y_n - 0.5*d_y) - theta_3 - phi_n  ) * 0.5*d_y - (y_n - 0.5*d_y)

        noise = jax.random.normal(keys[t], shape=X_0.shape) * (1.0/10.0)**noise_level
        x_n_new = x_n + lr * dt_x + noise[:, 0]
        y_n_new = y_n + lr * dt_y + noise[:, 1]

        x_vec_new = jnp.asarray([x_n_new, y_n_new]).T
        return x_vec_new, x_vec_new

    _, Xs_saves_tanh = jax.lax.scan(iteration, X_0, tbeta_range)

    return Xs_saves_tanh



def compute_y_n_tau_eps(beta, d_x, d_y, eps=0.1, x_0=10**(-7)):

    t_A = -(1.0/(0.5*beta*(d_x**2)-1.0)) * jnp.log( 1.0/(beta*d_x*x_0) )
    t_B = -1.0 * jnp.log( (d_x/2.0-1.0/(beta*d_x)) / ( d_x/2.0 - eps ) )
    
    y_n_t_eps = (1.0/3.0) * d_y * jnp.exp(
        t_A + t_B
    )

    return y_n_t_eps #- d_y/2.0


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(21, 6*4), dpi=200)

outer_most = gridspec.GridSpec(2, 1, height_ratios=[1, 1.0/3.0], hspace=0.15)

outer = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_most[0], height_ratios=[1, 1.0, 0.5], wspace=0.05, hspace=0.05)
#outer = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 0.5, 1], hspace=0.1)
inner_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.05)
inner_cent = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.05)
inner_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.05)

axs = [0, 0, 0]
axs[0] = fig.add_subplot(inner_top[0])
axs[1] = fig.add_subplot(inner_top[1], sharey=axs[0])
axs[2] = fig.add_subplot(inner_top[2], sharey=axs[0])
plt.setp(axs[0].get_xticklabels(), visible=False); plt.setp(axs[1].get_xticklabels(), visible=False); plt.setp(axs[2].get_xticklabels(), visible=False); 
plt.setp(axs[1].get_yticklabels(), visible=False)
plt.setp(axs[2].get_yticklabels(), visible=False)
for ax in axs:
    ax.tick_params(axis='both', labelsize=16)

axs_cent = [0, 0, 0]
axs_cent[0] = fig.add_subplot(inner_cent[0])
axs_cent[1] = fig.add_subplot(inner_cent[1], sharey=axs_cent[0])
axs_cent[2] = fig.add_subplot(inner_cent[2], sharey=axs_cent[0])
plt.setp(axs_cent[0].get_xticklabels(), visible=False); plt.setp(axs_cent[1].get_xticklabels(), visible=False); plt.setp(axs_cent[2].get_xticklabels(), visible=False); 
plt.setp(axs_cent[1].get_yticklabels(), visible=False);
plt.setp(axs_cent[2].get_yticklabels(), visible=False);
for ax in axs_cent:
    ax.tick_params(axis='both', labelsize=16)

axs_bot = [0, 0, 0]
axs_bot[0] = fig.add_subplot(inner_bot[0])
axs_bot[1] = fig.add_subplot(inner_bot[1], sharey=axs_bot[0])
axs_bot[2] = fig.add_subplot(inner_bot[2], sharey=axs_bot[0])
plt.setp(axs_bot[1].get_yticklabels(), visible=False);
plt.setp(axs_bot[2].get_yticklabels(), visible=False);
for ax in axs_bot:
    ax.tick_params(axis='both', labelsize=16)


ax_bot2 = fig.add_subplot(outer_most[1])
#gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.1)
ax_bot2.tick_params(axis='both', labelsize=16)

def calc_sig(X, XI, beta):
    D_2 = calculate_distances(X, XI)
    S = calculate_signals(beta, D_2)
    S_sum = calculate_aggregate_signals(S)

    return S_sum

calc_sig_all_t = jax.vmap(calc_sig, in_axes=(0, None, None))

eps = 0.2

axs[0].set_ylabel(r"$y_n$", fontsize=22)
axs_cent[0].set_ylabel(r"$y_n$", fontsize=22)
axs_bot[0].set_ylabel(r"$y_n$", fontsize=22)

for i in range(3):
    Xs_saves = training_loop(X_0, X_0, XI, jnp.asarray([betas[i]]), gamma, tmax, lr, noise_level, keys) # Just to get init cond.
    Xs_saves_tanh = simulate_tanh_sys(X_0, d_x, d_y, betas[i, 0], tmax, lr, noise_level, keys, tbeta_range)
    Xs_saves_tanh_approx = simulate_tanh_sys_approx(X_0, d_x, d_y, betas[i, 0], tmax, lr, noise_level, keys, tbeta_range)

    color = (0.04, 0.04, 0.04)
    axs[i].set_title(r"$\beta=$"+str(betas[i][0]), fontsize=22)
    axs[i].plot(Xs_saves[:, :, 0], Xs_saves[:, :, 1], color=color)
    axs[i].set_xlim(-d_x/2-0.2, d_x/2+0.2)
    axs[i].set_ylim(-0.1, d_y+0.25)

    color = (0.24, 0.24, 0.24)
    axs_cent[i].plot(Xs_saves_tanh[:, :, 0], Xs_saves_tanh[:, :, 1], color=color)
    axs_cent[i].set_xlim(-d_x/2-0.2, d_x/2+0.2)
    axs_cent[i].set_ylim(-0.1, d_y+0.25)

    axs_bot[i].plot(Xs_saves_tanh_approx[:, :, 0], Xs_saves_tanh_approx[:, :, 1], color=color)
    axs_bot[i].set_xlim(-d_x/2-0.2, d_x/2+0.2)
    axs_bot[i].set_ylim(-0.1, d_y/3.0+0.1)
    axs_bot[i].set_xlabel(r"$x_n$", fontsize=22)

    y_tau_eps = compute_y_n_tau_eps(betas[i][0], d_x, d_y, eps, 10**(-noise_level))
    color=(0.751, 0.251, 0.251)
    axs[i].scatter(0, y_tau_eps, color=color, s=80, zorder=4)
    axs_cent[i].scatter(0, y_tau_eps, color=color, s=80, zorder=4)
    axs_bot[i].scatter(0, y_tau_eps, color=color, s=80, zorder=4)

    for ax in [axs[i], axs_cent[i], axs_bot[i]]:
        ax.scatter(*jnp.mean(XI, axis=0), color="white", edgecolor="k", s=150, zorder=4)

        ax.text(XI[0, 0], XI[0, 1]+0.2, r"$\boldsymbol{\xi_1}$", color="k", fontsize=18, va="center", ha="center")
        ax.text(XI[1, 0], XI[1, 1]+0.2, r"$\boldsymbol{\xi_2}$", color="k", fontsize=18, va="center", ha="center")

        ax.scatter(XI[0, 0], XI[0, 1], marker="*", color="k", s=250)
        ax.scatter(XI[1, 0], XI[1, 1], marker="*", color="k", s=250)

        if not ax == axs_bot[i]:
            ax.text(XI[2, 0]+0.15, XI[2, 1], r"$\boldsymbol{\xi_3}$", color="k", fontsize=18, va="center", ha="center")
            ax.scatter(XI[2, 0], XI[2, 1], marker="*", color="k", s=250)



beta_lin = jnp.linspace(1.25, 25.0, 100)

y_tau_eps = compute_y_n_tau_eps(beta_lin, d_x, d_y, eps, 10**(-noise_level))

color=(0.751, 0.251, 0.251)
ax_bot2.plot(beta_lin, y_tau_eps, lw=10, color=color, alpha=0.6, label="Analytical Approximation")
beta_coarse = jnp.linspace(jnp.min(beta_lin), jnp.max(beta_lin), 20)
for i, b in enumerate(beta_coarse):
    Xs_saves_tanh_approx = simulate_tanh_sys(X_0, d_x, d_y, jnp.asarray([b]), tmax, lr, noise_level, keys, tbeta_range)# _approx
    Xs_saves_tanh_approx = Xs_saves_tanh_approx[:, Xs_saves_tanh_approx[-1, :, 0]>0.2]

    ts_tau = jnp.argmax(jnp.abs(Xs_saves_tanh_approx[:, :, 0]) > eps, axis=0)
    cell_choice = jnp.arange(Xs_saves_tanh_approx.shape[1], dtype=int)
    y_est = jnp.mean(Xs_saves_tanh_approx[ts_tau, cell_choice, 1])

    if i==0:
        ax_bot2.scatter(b, y_est, color=(0.04, 0.04, 0.04), s=80, zorder=4, label="Numerical Simulation")
    else:
        ax_bot2.scatter(b, y_est, color=(0.04, 0.04, 0.04), s=80, zorder=4)



ax_bot2.set_xlabel(r"$\beta$", fontsize=22)
ax_bot2.set_ylabel(r"$y(\tau^\epsilon)$", fontsize=22)
ax_bot2.legend(fontsize=22)

plt.savefig(f"Fig/S_progentior_estimate_d_y_{d_y}.png", bbox_inches='tight', pad_inches=0)


