import jax
jax.config.update("jax_default_matmul_precision", "highest") # In case you choose to use float32

import jax.numpy as jnp


# Checks if #jax.config.update("jax_enable_x64", True)  was called and adapts to the precision
is_x64 = jax.config.jax_enable_x64
numerical_eps = 1e-14 if is_x64 else 1e-7
print("USING 64 PRECISION", is_x64, numerical_eps)

@jax.jit
def calculate_distances(X, XI):
    # Assumes X (N-Cells, N-dim)
    # Assumes XI (N-patterns, N-dim)

    # calculates all distances more efficiently
    norm_X_2 = jnp.sum(X**2, axis=1)[:, None]    # N-Cells, 1
    norm_XI_2 = jnp.sum(XI**2, axis=1)[None, :]  # 1, N-patterns
    mat_mult_X_XI = X@XI.T                       # N-Cells, N_patterns

    # Stops negative distances due to numerical imprecision. Yeah, crazy i know
    D_2 = jnp.maximum(norm_X_2 + norm_XI_2 - 2*mat_mult_X_XI, 0.0) 
    return D_2


@jax.jit
def calculate_signals(beta, D_2):
    S = jnp.exp(-beta*D_2)
    return S # (N-Cells, N-patterns)

@jax.jit
def calculate_aggregate_signals(S, mute_cells):
    S_sum = (S * mute_cells[:, None]).sum(axis=0)
    return S_sum #(N-patterns, )

@jax.jit
def calculate_gradient(X, XI, beta, gamma, mute_cells, w_mu):
    D_2 = calculate_distances(X, XI)
    S = calculate_signals(beta, D_2)
    S_sum = calculate_aggregate_signals(S, mute_cells)*w_mu +gamma

    coefs = S / (S_sum)[None, :] # (N-cells, N-patterns)
    N = coefs.sum(axis=-1)  + numerical_eps # (N-cells,)
    grad = (coefs/N[:, None])  @ XI - X # (N-cells, N-dim)
    return grad


@jax.jit
def calculate_Es(X, XI, beta, gamma, mute_cells, w_mu):
    D_2 = calculate_distances(X, XI)

    # more numerically stable
    log_S_sum_raw = jax.nn.logsumexp(-beta * D_2, b=mute_cells[:, None], axis=0) + jnp.log(w_mu) # log(ab) = log(a) + log(b)
    log_S_sum = (1.0/w_mu) *  jnp.logaddexp(log_S_sum_raw, jnp.log(gamma))
    Es = log_S_sum.sum(axis=0)
    
    return Es


@jax.jit
def calculate_grads_and_Es(X, XI, beta, gamma, mute_cells, w_mu):
    D_2 = calculate_distances(X, XI)
    S = calculate_signals(beta, D_2)
    S_sum = calculate_aggregate_signals(S, mute_cells)*w_mu +gamma
    

    coefs = S / (S_sum)[None, :] # (N-cells, N-patterns)
    N = coefs.sum(axis=-1)  + numerical_eps # (N-cells,)
    grad = (coefs/N[:, None])  @ XI - X # (N-cells, N-dim)


    # more numerically stable
    log_S_sum_raw = jax.nn.logsumexp(-beta * D_2, b=mute_cells[:, None], axis=0) + jnp.log(w_mu) # log(ab) = log(a) + log(b)
    log_S_sum = (1.0/w_mu) *  jnp.logaddexp(log_S_sum_raw, jnp.log(gamma))
    Es = log_S_sum.sum(axis=0)

    # assumes you have a probe so returns only the grad for last cell (i.e. the probe)
    return (grad[-1], Es)



@jax.jit
def _training_loop(X_0, X_STEM, XI, betas, gamma, w_mu, tmax, lr, noise_level, keys, mute_cells_ts, tbeta_range):
    # keys should be of shape len(betas)*tmax

    @jax.jit
    def training_iter(X, t):
        beta = betas[t//tmax]
        key = keys[t]

        grad = calculate_gradient(X, XI, beta, gamma, mute_cells_ts[t], w_mu)
        X_new = X + ( lr*grad + jax.random.normal(key, shape=X.shape) * (1.0/10.0)**noise_level )

        X_new_cond = (1-(mute_cells_ts[t][:, None]==0)) * X_new + (mute_cells_ts[t][:, None]==0) * X_STEM
        return X_new_cond, X_new_cond
        
    _, Xs_saves = jax.lax.scan(training_iter, X_0, tbeta_range)

    return Xs_saves


# Wrapper
def training_loop(X_0, X_STEM, XI, betas, gamma, tmax, lr, noise_level, keys, mute_cells_ts=None, w_mu=None):
    assert (not jnp.isscalar(betas))
    assert len(keys)==len(betas)*tmax

    tbeta_range = jnp.arange(tmax*len(betas), dtype=int)
    if mute_cells_ts is None:
        mute_cells_ts = jnp.ones((tmax*len(betas), len(X_0)))

    
    if w_mu is None:
        w_mu = jnp.ones((len(XI)))

    assert mute_cells_ts.shape[0] == tmax*len(betas)
    assert mute_cells_ts.shape[1] == len(X_0)
    assert X_0.shape[1] == XI.shape[1]

    return _training_loop(X_0, X_STEM, XI, betas, gamma, w_mu, tmax, lr, noise_level, keys, mute_cells_ts, tbeta_range)


@jax.jit
def _training_loop_linear_growth(X_0, X_STEM, XI, betas, gamma, w_mu, tmax, lr, noise_level, keys, growth_rate, t_offset, tbeta_range):
    # keys should be of shape len(betas)*tmax
    k = jnp.arange(len(X_0))

    @jax.jit
    def training_iter(X, t):
        mute_cells_ts = (k <= growth_rate * (t+t_offset))
        beta = betas[t//tmax]
        key = keys[t]

        grad = calculate_gradient(X, XI, beta, gamma, mute_cells_ts, w_mu)
        X_new = X + ( lr*grad + jax.random.normal(key, shape=X.shape) * (1.0/10.0)**noise_level )

        X_new_cond = (1-(mute_cells_ts[:, None]==0)) * X_new + (mute_cells_ts[:, None]==0) * X_STEM
        return X_new_cond, X_new_cond
        
    _, Xs_saves = jax.lax.scan(training_iter, X_0, tbeta_range)

    return Xs_saves


# Wrapper
def training_loop_linear_growth(X_0, X_STEM, XI, betas, gamma, tmax, lr, noise_level, keys, growth_rate, t_offset, w_mu=None):
    assert not jnp.isscalar(betas)
    assert len(keys)==(len(betas)*tmax)


    if w_mu is None:
        w_mu = jnp.ones((len(XI)))
        
    tbeta_range = jnp.arange(tmax*len(betas), dtype=int)
    assert X_0.shape[1] == XI.shape[1]

    return _training_loop_linear_growth(X_0, X_STEM, XI, betas, gamma, w_mu, tmax, lr, noise_level,
                                        keys, growth_rate, t_offset, tbeta_range)



@jax.jit
def training_loop_nosave(X_0, X_STEM, XI, betas, gamma, tmax, lr, noise_level, keys, mute_cells_ts, w_mu):
    # keys should be of shape len(betas)*tmax

    @jax.jit
    def training_iter(t, X):
        beta = betas[t//tmax]
        key = keys[t]

        grad = calculate_gradient(X, XI, beta, gamma, mute_cells_ts[t], w_mu)
        X_new = X + ( lr*grad + jax.random.normal(key, shape=X.shape) * (1.0/10.0)**noise_level )

        X_new_cond = (1-(mute_cells_ts[t][:, None]==0)) * X_new + (mute_cells_ts[t][:, None]==0) * X_STEM
        return X_new_cond

    
    Xs_final = jax.lax.fori_loop(0, tmax*len(betas), training_iter, X_0)
    return Xs_final
