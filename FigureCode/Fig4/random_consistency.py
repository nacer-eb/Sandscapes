import os

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(41)
key, subkey = jax.random.split(key)

random_integers = jax.random.randint(subkey, shape=(3,), minval=0, maxval=100, dtype=jnp.int64)
print(random_integers)

random_floats = jax.random.uniform(subkey, shape=(3,), dtype=jnp.float64)
print(random_floats)
