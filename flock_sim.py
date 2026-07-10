import chex
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
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


@jaxtyped(typechecker=beartype)
def simulation_to_grid(
    positions: Float[Array, "... 2"],
    grid_size: int,
) -> Float[Array, "... 2"]:
    scale = (grid_size - 1) / 2
    return (jnp.clip(positions, -1, 1) + 1) * scale


@jaxtyped(typechecker=beartype)
def simulation_to_canvas(
    positions: Float[Array, "... 2"],
    render_grid_size: int,
) -> Float[Array, "... 2"]:
    scale = (render_grid_size - 1) / 2
    x = (jnp.clip(positions[..., 0], -1, 1) + 1) * scale
    y = (jnp.clip(positions[..., 1], -1, 1) + 1) * scale
    return jnp.stack([x, y], axis=-1)


@jaxtyped(typechecker=beartype)
def grid_coordinates(
    simulation_grid_size: int,
) -> Float[Array, "height width 2"]:
    axis = jnp.linspace(-1, 1, simulation_grid_size)
    grid_x, grid_y = jnp.meshgrid(axis, axis, indexing="xy")
    return jnp.stack([grid_x, grid_y], axis=-1)


@jaxtyped(typechecker=beartype)
def separation_field(
    flock: Flock,
    simulation_grid_size: int,
    sigma: float,
    strength: float,
) -> Float[Array, "height width 2"]:
    grid = grid_coordinates(simulation_grid_size)
    offset = grid[None, :, :, :] - flock.positions[:, None, None, :]
    distance_squared = jnp.sum(offset * offset, axis=-1, keepdims=True)
    distance = jnp.sqrt(distance_squared + EPS)
    sigma_squared = jnp.maximum(sigma * sigma, EPS)
    gaussian = jnp.exp(-distance_squared / (2 * sigma_squared))
    influence = strength * gaussian * offset / distance
    return jnp.sum(influence, axis=0)


@jaxtyped(typechecker=beartype)
def separation_density(
    flock: Flock,
    simulation_grid_size: int,
    sigma: float,
    strength: float,
) -> Float[Array, "height width"]:
    grid = grid_coordinates(simulation_grid_size)
    offset = grid[None, :, :, :] - flock.positions[:, None, None, :]
    distance_squared = jnp.sum(offset * offset, axis=-1)
    sigma_squared = jnp.maximum(sigma * sigma, EPS)
    gaussian = jnp.exp(-distance_squared / (2 * sigma_squared))
    return jnp.sum(strength * gaussian, axis=0)


@jaxtyped(typechecker=beartype)
def sample_field_bilinear(
    field: Float[Array, "height width 2"],
    positions: Float[Array, "bird 2"],
) -> Float[Array, "bird 2"]:
    grid_size = field.shape[0]
    grid_positions = simulation_to_grid(positions, grid_size)
    coords_yx = jnp.stack([grid_positions[:, 1], grid_positions[:, 0]], axis=0)
    x_values = map_coordinates(field[:, :, 0], coords_yx, order=1, mode="nearest")
    y_values = map_coordinates(field[:, :, 1], coords_yx, order=1, mode="nearest")
    return jnp.stack([x_values, y_values], axis=-1)


@jaxtyped(typechecker=beartype)
def clamp_vector_lengths(
    vectors: Float[Array, "... 2"],
    min_length: float,
    max_length: float,
) -> Float[Array, "... 2"]:
    lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    clamped_lengths = jnp.clip(lengths, min_length, max_length)
    return normalize(vectors) * clamped_lengths


@jaxtyped(typechecker=beartype)
def update_flock_with_separation_field(
    flock: Flock,
    simulation_grid_size: int,
    dt: float,
    min_velocity: float,
    max_velocity: float,
    separation_strength: float,
    turn_rate: float,
    sigma: float,
) -> Flock:
    velocity = clamp_vector_lengths(flock.headings, min_velocity, max_velocity)
    field = separation_field(
        flock=flock,
        simulation_grid_size=simulation_grid_size,
        sigma=sigma,
        strength=separation_strength,
    )
    separation = sample_field_bilinear(field, flock.positions)
    headings = normalize(velocity + turn_rate * separation)
    speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
    positions = jnp.clip(flock.positions + headings * speed * dt, -1, 1)
    return Flock(positions=positions, headings=headings)
