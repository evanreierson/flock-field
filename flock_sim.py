import math
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float, Int, jaxtyped

EPS = 1e-8


@jaxtyped(typechecker=beartype)
def normalize(vectors: Float[Array, "... 2"]) -> Float[Array, "... 2"]:
    return vectors / jnp.maximum(jnp.linalg.norm(vectors, axis=-1, keepdims=True), EPS)


@chex.dataclass
@jaxtyped(typechecker=beartype)
class Flock:
    positions: Float[Array, "bird 2"]
    headings: Float[Array, "bird 2"]
    generation: Int[Array, ""]


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

    return Flock(
        positions=positions,
        headings=headings,
        generation=jnp.array(0, dtype=jnp.int32),
    )


@jaxtyped(typechecker=beartype)
def simulation_to_grid(
    positions: Float[Array, "... 2"],
    grid_size: int,
) -> Float[Array, "... 2"]:
    scale = (grid_size - 1) / 2
    return (jnp.clip(positions, -1, 1) + 1) * scale


@jaxtyped(typechecker=beartype)
def grid_coordinates(
    simulation_grid_size: int,
) -> Float[Array, "height width 2"]:
    axis = jnp.linspace(-1, 1, simulation_grid_size)
    grid_x, grid_y = jnp.meshgrid(axis, axis, indexing="xy")
    return jnp.stack([grid_x, grid_y], axis=-1)


@jaxtyped(typechecker=beartype)
def local_splat_field(
    positions: Float[Array, "bird 2"],
    simulation_grid_size: int,
    kernel_radius: int,
    kernel: Callable[
        [Float[Array, "bird cell 2"]],
        Float[Array, "bird cell 2"],
    ],
) -> Float[Array, "height width 2"]:
    grid_positions = simulation_to_grid(positions, simulation_grid_size)
    centers = jnp.rint(grid_positions).astype(jnp.int32)

    axis = jnp.arange(-kernel_radius, kernel_radius + 1, dtype=jnp.int32)
    offset_x, offset_y = jnp.meshgrid(axis, axis, indexing="xy")
    index_offsets = jnp.stack([offset_x.ravel(), offset_y.ravel()], axis=-1)

    cell_indices = centers[:, None, :] + index_offsets[None, :, :]

    scale = (simulation_grid_size - 1) / 2
    cell_positions = cell_indices.astype(positions.dtype) / scale - 1
    simulation_offsets = cell_positions - positions[:, None, :]
    values = kernel(simulation_offsets)

    scatter_indices = jnp.where(cell_indices < 0, simulation_grid_size, cell_indices)
    field = jnp.zeros(
        (simulation_grid_size, simulation_grid_size, 2),
        dtype=positions.dtype,
    )
    return field.at[scatter_indices[:, :, 1], scatter_indices[:, :, 0]].add(
        values, mode="drop"
    )


@beartype
def default_kernel_radius(sigma: float, simulation_grid_size: int) -> int:
    sigma_cells = sigma * (simulation_grid_size - 1) / 2
    return max(1, math.ceil(3 * sigma_cells))


@jaxtyped(typechecker=beartype)
def radial_kernel(
    offset: Float[Array, "... 2"],
    sigma: float,
    strength: float,
) -> Float[Array, "... 2"]:
    distance_squared = jnp.sum(offset * offset, axis=-1, keepdims=True)
    distance = jnp.sqrt(distance_squared + EPS)
    sigma_squared = jnp.maximum(sigma * sigma, EPS)
    gaussian = jnp.exp(-distance_squared / (2 * sigma_squared))
    return strength * gaussian * offset / distance


@jaxtyped(typechecker=beartype)
def separation_field(
    flock: Flock,
    simulation_grid_size: int,
    sigma: float,
    strength: float,
) -> Float[Array, "height width 2"]:
    return local_splat_field(
        flock.positions,
        simulation_grid_size,
        default_kernel_radius(sigma, simulation_grid_size),
        lambda offset: radial_kernel(offset, sigma, strength),
    )


@jaxtyped(typechecker=beartype)
def cohesion_field(
    flock: Flock,
    simulation_grid_size: int,
    sigma: float,
    strength: float,
) -> Float[Array, "height width 2"]:
    return local_splat_field(
        flock.positions,
        simulation_grid_size,
        default_kernel_radius(sigma, simulation_grid_size),
        lambda offset: radial_kernel(offset, sigma, -strength),
    )


@jaxtyped(typechecker=beartype)
def alignment_kernel(
    offset: Float[Array, "bird cell 2"],
    headings: Float[Array, "bird 2"],
    sigma: float,
    strength: float,
) -> Float[Array, "bird cell 2"]:
    distance_squared = jnp.sum(offset * offset, axis=-1, keepdims=True)
    sigma_squared = jnp.maximum(sigma * sigma, EPS)
    gaussian = jnp.exp(-distance_squared / (2 * sigma_squared))
    return strength * gaussian * headings[:, None, :]


@jaxtyped(typechecker=beartype)
def alignment_field(
    flock: Flock,
    simulation_grid_size: int,
    sigma: float,
    strength: float,
) -> Float[Array, "height width 2"]:
    return local_splat_field(
        flock.positions,
        simulation_grid_size,
        default_kernel_radius(sigma, simulation_grid_size),
        lambda offset: alignment_kernel(offset, flock.headings, sigma, strength),
    )


@jaxtyped(typechecker=beartype)
def boundary_field(
    simulation_grid_size: int,
    margin: float,
    strength: float,
) -> Float[Array, "height width 2"]:
    grid = grid_coordinates(simulation_grid_size)
    radius = jnp.linalg.norm(grid, axis=-1, keepdims=True)
    margin = jnp.maximum(margin, EPS)

    boundary_radius = 1.0
    distance_to_boundary = boundary_radius - radius
    inward = jnp.clip((margin - distance_to_boundary) / margin, 0, 1)
    direction_to_center = -grid / jnp.maximum(radius, EPS)
    return strength * inward * inward * direction_to_center


@jaxtyped(typechecker=beartype)
def noise_field(
    generation: Int[Array, ""],
    simulation_grid_size: int,
    strength: float,
    noise_grid_size: int = 4,
    temporal_rate: float = 0.025,
) -> Float[Array, "height width 2"]:
    grid = grid_coordinates(simulation_grid_size)
    x = grid[:, :, 0]
    y = grid[:, :, 1]

    time = generation.astype(x.dtype) * temporal_rate
    frequency = jnp.pi * jnp.maximum(
        jnp.asarray(noise_grid_size, dtype=x.dtype),
        1.0,
    )

    waves = jnp.stack(
        [
            jnp.sin(frequency * (x + 0.35 * y) + 0.7 * time)
            + 0.5 * jnp.sin(1.7 * frequency * (-0.2 * x + y) + 1.3 * time),
            jnp.cos(frequency * (y - 0.25 * x) + 1.1 * time)
            + 0.5 * jnp.cos(1.5 * frequency * (x + 0.15 * y) + 0.9 * time),
        ],
        axis=-1,
    )
    magnitude = jnp.linalg.norm(waves, axis=-1, keepdims=True)
    mean_magnitude = jnp.maximum(jnp.mean(magnitude), EPS)
    return strength * waves / mean_magnitude


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
def update_flock(
    flock: Flock,
    simulation_grid_size: int,
    dt: float,
    speed: float,
    separation_strength: float,
    turn_rate: float,
    sigma: float,
    cohesion_strength: float = 0.25,
    cohesion_sigma: float = 0.2,
    alignment_strength: float = 0.5,
    alignment_sigma: float = 0.16,
    boundary_margin: float = 0.2,
    boundary_strength: float = 8.0,
    noise_strength: float = 0.0,
    noise_grid_size: int = 4,
    noise_temporal_rate: float = 0.025,
) -> Flock:
    next_generation = flock.generation + jnp.array(1, dtype=flock.generation.dtype)
    field = boundary_field(
        simulation_grid_size=simulation_grid_size,
        margin=boundary_margin,
        strength=boundary_strength,
    )
    if noise_strength != 0.0:
        field = field + noise_field(
            generation=flock.generation,
            simulation_grid_size=simulation_grid_size,
            strength=noise_strength,
            noise_grid_size=noise_grid_size,
            temporal_rate=noise_temporal_rate,
        )
    field = field + separation_field(
        flock=flock,
        simulation_grid_size=simulation_grid_size,
        sigma=sigma,
        strength=separation_strength,
    )
    field = field + cohesion_field(
        flock=flock,
        simulation_grid_size=simulation_grid_size,
        sigma=cohesion_sigma,
        strength=cohesion_strength,
    )
    field = field + alignment_field(
        flock=flock,
        simulation_grid_size=simulation_grid_size,
        sigma=alignment_sigma,
        strength=alignment_strength,
    )
    influence = sample_field_bilinear(field, flock.positions)
    headings = normalize(flock.headings + turn_rate * influence)
    positions = jnp.clip(flock.positions + headings * speed * dt, -1, 1)
    return Flock(positions=positions, headings=headings, generation=next_generation)


@beartype
def make_update_step(
    simulation_grid_size: int,
    dt: float,
    speed: float,
    separation_strength: float,
    turn_rate: float,
    sigma: float,
    cohesion_strength: float = 0.25,
    cohesion_sigma: float = 0.2,
    alignment_strength: float = 0.5,
    alignment_sigma: float = 0.16,
    boundary_margin: float = 0.2,
    boundary_strength: float = 8.0,
    noise_strength: float = 0.0,
    noise_grid_size: int = 4,
    noise_temporal_rate: float = 0.025,
) -> Callable[[Flock], Flock]:
    """Return a jitted flock update with static simulation configuration."""
    return jax.jit(
        lambda flock: update_flock(
            flock=flock,
            simulation_grid_size=simulation_grid_size,
            dt=dt,
            speed=speed,
            separation_strength=separation_strength,
            turn_rate=turn_rate,
            sigma=sigma,
            cohesion_strength=cohesion_strength,
            cohesion_sigma=cohesion_sigma,
            alignment_strength=alignment_strength,
            alignment_sigma=alignment_sigma,
            boundary_margin=boundary_margin,
            boundary_strength=boundary_strength,
            noise_strength=noise_strength,
            noise_grid_size=noise_grid_size,
            noise_temporal_rate=noise_temporal_rate,
        )
    )
