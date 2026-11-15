from sandscapes_main import *
import jax
import jax.numpy as jnp

import numpy as np
import tqdm


def calculate_signals_all(Xs_saves, XIs, beta, gamma, mute_cells, batch_size=100, w_mu=None):

    
    if w_mu is None:
        w_mu = jnp.ones((len(XIs)))

    # Calculate the signal at each time point
    calc_D_2 = jax.vmap(calculate_distances, in_axes=(0, None))

    gammas_eff = np.zeros((Xs_saves.shape[0], XIs.shape[0]))

    for i in tqdm.tqdm(range(0, Xs_saves.shape[0], batch_size)):
        i_max = min(i + batch_size, Xs_saves.shape[0])
        D_2 = np.asarray(calc_D_2(Xs_saves[i:i_max], XIs), copy=True)
        gammas_eff[i:i_max] = gamma[None, :] +\
            w_mu[None, :] * (np.exp(-beta*D_2)*mute_cells[i:i_max, :, None]).sum(axis=1) # aggregated

    return gammas_eff



# For cells the gradient is not needed
def calculate_energy_over_cells_and_time(cells, gammas_eff, XIs, beta, gamma,
                                         mute_cells=None, probe_factor=1E-5, batch_size=100, w_mu=None):

    
    if w_mu is None:
        w_mu = jnp.ones((len(XIs)))

    calc_grads_and_Es_all_cells = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None, None))
    calc_grads_and_Es_all_cells_all_t = jax.vmap(calc_grads_and_Es_all_cells, in_axes=(0, None, None, 0, None, None))

    
    Es = np.zeros((cells.shape[0], cells.shape[1]))
    for i in tqdm.tqdm(range(0, cells.shape[0], batch_size)):
        i_max = min(i + batch_size, cells.shape[0])

        _, Es_ = calc_grads_and_Es_all_cells_all_t(cells[i:i_max, :, None, :], XIs, beta,
                                                        gammas_eff[i:i_max], jnp.asarray([probe_factor]), w_mu)

        Es[i:i_max] = np.asarray(Es_, copy=True)

    
    return -Es



def calculate_energy_over_probes_and_time(probes, gammas_eff, XIs, beta, gamma, probe_factor=1E-5, batch_size=100, w_mu=None):

    if w_mu is None:
        w_mu = jnp.ones((len(XIs)))

    calc_grads_and_Es_all_probes = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None, None))
    calc_grads_and_Es_all_probes_many_t = jax.vmap(calc_grads_and_Es_all_probes, in_axes=(None, None, None, 0, None, None))

    grads = np.zeros((gammas_eff.shape[0], probes.shape[0], probes.shape[2])) 
    Es = np.zeros((gammas_eff.shape[0], probes.shape[0]))

    for i in tqdm.tqdm(range(0, gammas_eff.shape[0], batch_size)):
        i_max = min(i + batch_size, gammas_eff.shape[0])
        grads_, Es_ = calc_grads_and_Es_all_probes_many_t(probes, XIs, beta,
                                                    gammas_eff[i:i_max], jnp.asarray([probe_factor]), w_mu)

        grads[i:i_max] = np.asarray(grads_, copy=True)
        Es[i:i_max] = np.asarray(Es_, copy=True)

    return (grads, -Es)

