import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

random_integers = jax.random.randint(subkey, shape=(3,), minval=0, maxval=100)
print(random_integers)

random_floats = jax.random.uniform(subkey, shape=(3,))
print(random_floats)
