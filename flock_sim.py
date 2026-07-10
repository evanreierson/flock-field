import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

EPS = 1e-8


@jaxtyped(typechecker=beartype)
def normalize(vectors: Float[Array, "... 2"]) -> Float[Array, "... 2"]:
    return vectors / jnp.maximum(jnp.linalg.norm(vectors, axis=-1, keepdims=True), EPS)


@chex.dataclass
@jaxtyped(typechecker=beartype)
class Flock:
    positions: Float[Array, "bird 2"]
    headings: Float[Array, "bird 2"]


def initialize_flock(
    rng_key,
    population=100,
) -> Flock:
    k1, k2 = jax.random.split(rng_key, 2)

    positions = jax.random.uniform(
        k1,
        shape=(population, 2),
        minval=jnp.array([-1, -1]),
        maxval=jnp.array([1, 1]),
    )
    headings = normalize(
        jax.random.uniform(
            k2,
            shape=(population, 2),
            minval=jnp.array([-1, -1]),
            maxval=jnp.array([1, 1]),
        )
    )

    return Flock(positions=positions, headings=headings)
