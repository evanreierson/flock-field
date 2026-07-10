import math
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jax.scipy.ndimage import map_coordinates
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
    in_bounds = jnp.all(
        (0 <= cell_indices) & (cell_indices < simulation_grid_size),
        axis=-1,
    )

    scale = (simulation_grid_size - 1) / 2
    cell_positions = cell_indices.astype(positions.dtype) / scale - 1
    simulation_offsets = cell_positions - positions[:, None, :]
    values = kernel(simulation_offsets)
    values = jnp.where(in_bounds[:, :, None], values, jnp.zeros_like(values))

    safe_indices = jnp.clip(cell_indices, 0, simulation_grid_size - 1)
    field = jnp.zeros(
        (simulation_grid_size, simulation_grid_size, 2),
        dtype=positions.dtype,
    )
    return field.at[safe_indices[:, :, 1], safe_indices[:, :, 0]].add(values)


@jaxtyped(typechecker=beartype)
def separation_kernel(
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
    kernel_radius: int | None = None,
) -> Float[Array, "height width 2"]:
    if kernel_radius is None:
        sigma_cells = sigma * (simulation_grid_size - 1) / 2
        kernel_radius = max(1, math.ceil(3 * sigma_cells))

    return local_splat_field(
        flock.positions,
        simulation_grid_size,
        kernel_radius,
        lambda offset: separation_kernel(offset, sigma, strength),
    )


@jaxtyped(typechecker=beartype)
def cohesion_kernel(
    offset: Float[Array, "... 2"],
    sigma: float,
    strength: float,
) -> Float[Array, "... 2"]:
    distance_squared = jnp.sum(offset * offset, axis=-1, keepdims=True)
    distance = jnp.sqrt(distance_squared + EPS)
    sigma_squared = jnp.maximum(sigma * sigma, EPS)
    gaussian = jnp.exp(-distance_squared / (2 * sigma_squared))
    return -strength * gaussian * offset / distance


@jaxtyped(typechecker=beartype)
def cohesion_field(
    flock: Flock,
    simulation_grid_size: int,
    sigma: float,
    strength: float,
    kernel_radius: int | None = None,
) -> Float[Array, "height width 2"]:
    if kernel_radius is None:
        sigma_cells = sigma * (simulation_grid_size - 1) / 2
        kernel_radius = max(1, math.ceil(3 * sigma_cells))

    return local_splat_field(
        flock.positions,
        simulation_grid_size,
        kernel_radius,
        lambda offset: cohesion_kernel(offset, sigma, strength),
    )


@jaxtyped(typechecker=beartype)
def boundary_field(
    simulation_grid_size: int,
    margin: float,
    strength: float,
) -> Float[Array, "height width 2"]:
    grid = grid_coordinates(simulation_grid_size)
    x = grid[:, :, 0]
    y = grid[:, :, 1]
    margin = jnp.maximum(margin, EPS)

    left = jnp.clip((margin - (x + 1)) / margin, 0, 1)
    right = jnp.clip((margin - (1 - x)) / margin, 0, 1)
    bottom = jnp.clip((margin - (y + 1)) / margin, 0, 1)
    top = jnp.clip((margin - (1 - y)) / margin, 0, 1)

    inward_x = left * left - right * right
    inward_y = bottom * bottom - top * top
    return strength * jnp.stack([inward_x, inward_y], axis=-1)


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
    boundary_margin: float = 0.2,
    boundary_strength: float = 8.0,
    separation_kernel_radius: int | None = None,
    cohesion_kernel_radius: int | None = None,
) -> Flock:
    field = boundary_field(
        simulation_grid_size=simulation_grid_size,
        margin=boundary_margin,
        strength=boundary_strength,
    )
    field = field + separation_field(
        flock=flock,
        simulation_grid_size=simulation_grid_size,
        sigma=sigma,
        strength=separation_strength,
        kernel_radius=separation_kernel_radius,
    )
    field = field + cohesion_field(
        flock=flock,
        simulation_grid_size=simulation_grid_size,
        sigma=cohesion_sigma,
        strength=cohesion_strength,
        kernel_radius=cohesion_kernel_radius,
    )
    influence = sample_field_bilinear(field, flock.positions)
    headings = normalize(flock.headings + turn_rate * influence)
    positions = jnp.clip(flock.positions + headings * speed * dt, -1, 1)
    return Flock(positions=positions, headings=headings)


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
    boundary_margin: float = 0.2,
    boundary_strength: float = 8.0,
    separation_kernel_radius: int | None = None,
    cohesion_kernel_radius: int | None = None,
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
            boundary_margin=boundary_margin,
            boundary_strength=boundary_strength,
            separation_kernel_radius=separation_kernel_radius,
            cohesion_kernel_radius=cohesion_kernel_radius,
        )
    )
