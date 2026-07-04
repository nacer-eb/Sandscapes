from sandscapes_main import *
import jax
import jax.numpy as jnp

import numpy as np
import tqdm


def calculate_signals_all(Xs_saves, XIs, beta, gamma, mute_cells, batch_size=100):
    
    # Calculate the signal at each time point
    calc_D_2 = jax.vmap(calculate_distances, in_axes=(0, None))

    gammas_eff = np.zeros((Xs_saves.shape[0], XIs.shape[0]))
    for i in tqdm.tqdm(range(Xs_saves.shape[0]//batch_size)):
        D_2 = np.asarray(calc_D_2(Xs_saves[i*batch_size: (i+1)*batch_size], XIs), copy=True)
        gammas_eff[i*batch_size:(i+1)*batch_size] = gamma[None, :] +\
            (np.exp(-beta*D_2)*mute_cells[i*batch_size:(i+1)*batch_size, :, None]).sum(axis=1) # aggregated
        
    D_2 = np.asarray(calc_D_2(Xs_saves[-batch_size:], XIs), copy=True)   
    gammas_eff[-batch_size:] = gamma[None, :] + (np.exp(-beta*D_2)*mute_cells[-batch_size:, :, None]).sum(axis=1) # aggregated

    return gammas_eff



# For cells the gradient is not needed
def calculate_energy_over_cells_and_time(cells, gammas_eff, XIs, beta, gamma,
                                         mute_cells=None, probe_factor=1E-5, batch_size=100):
    
    calc_grads_and_Es_all_cells = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_cells_all_t = jax.vmap(calc_grads_and_Es_all_cells, in_axes=(0, None, None, 0, None))

    
    Es = np.zeros((cells.shape[0], cells.shape[1]))
    for i in tqdm.tqdm(range(cells.shape[0]//batch_size)):
        _, Es_ = calc_grads_and_Es_all_cells_all_t(cells[i*batch_size:(i+1)*batch_size, :, None, :], XIs, beta,
                                                        gammas_eff[i*batch_size:(i+1)*batch_size], jnp.asarray([probe_factor]))

        Es[i*batch_size:(i+1)*batch_size] = np.asarray(Es_, copy=True)

    _, Es_ = calc_grads_and_Es_all_cells_all_t(cells[-batch_size:, :, None, :], XIs, beta,
                                                        gammas_eff[-batch_size:], jnp.asarray([probe_factor]))
    Es[-batch_size:] = np.asarray(Es_, copy=True)

    
    return -Es



def calculate_energy_over_probes_and_time(probes, gammas_eff, XIs, beta, gamma, probe_factor=1E-5, batch_size=100):
    calc_grads_and_Es_all_probes = jax.vmap(calculate_grads_and_Es, in_axes=(0, None, None, None, None))
    calc_grads_and_Es_all_probes_many_t = jax.vmap(calc_grads_and_Es_all_probes, in_axes=(None, None, None, 0, None))

    grads = np.zeros((gammas_eff.shape[0], probes.shape[0], probes.shape[2])) 
    Es = np.zeros((gammas_eff.shape[0], probes.shape[0]))
    for i in tqdm.tqdm(range(gammas_eff.shape[0]//batch_size)):
        grads_, Es_ = calc_grads_and_Es_all_probes_many_t(probes, XIs, beta,
                                                    gammas_eff[i*batch_size:(i+1)*batch_size], jnp.asarray([probe_factor]))

        grads[i*batch_size:(i+1)*batch_size] = np.asarray(grads_, copy=True)
        Es[i*batch_size:(i+1)*batch_size] = np.asarray(Es_, copy=True)

    grads_, Es_ = calc_grads_and_Es_all_probes_many_t(probes, XIs, beta,
                                                    gammas_eff[-batch_size:], jnp.asarray([probe_factor]))
    grads[-batch_size:] = np.asarray(grads_, copy=True)
    Es[-batch_size:] = np.asarray(Es_, copy=True)

    return (grads, -Es)

