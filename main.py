from __future__ import annotations

from collections import deque

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

BIRD_COUNT = 300
TRAIL_WIDTH = 2
TRAIL_HISTORY_LENGTH = 200
MIN_TRAIL_ALPHA = 12
MAX_TRAIL_ALPHA = 220
BIRD_VELOCITY = 100.0
FPS = 60

# Separation field tuning. Each bird contributes a Gaussian "hill" to a
# scalar height field. Birds steer down the gradient of that field.
SEPARATION_SIGMA = 40.0
SEPARATION_STRENGTH = 20

# Cohesion is an attraction band: weak when birds are very close, strongest
# around COHESION_DISTANCE, then fades out.
COHESION_DISTANCE = 120.0
COHESION_SIGMA = 70.0
COHESION_STRENGTH = 2.0

# Alignment is a vector field. Each bird contributes its heading through an
# oriented Gaussian: longer in front/behind the bird, narrower sideways.
ALIGNMENT_LONG_SIGMA = 90.0
ALIGNMENT_SIDE_SIGMA = 35.0
ALIGNMENT_STRENGTH = 2.0

# Coherent wandering force. This is a smooth spatial vector field that slowly
# scrolls over time, rather than independent jitter per bird.
NOISE_SCALE = 160.0
NOISE_SCROLL_SPEED = 0.35
NOISE_STRENGTH = 1.5

MAX_TURN_FORCE = 8.0
EPS = 1e-8

BACKGROUND_COLOR = pygame.Color(48, 48, 48)
WHITE = pygame.Color(255, 255, 255)

# RGB color min and max to sample from (0-1.0)
COLOR_MIN = jnp.array([0.1, 0.2, 0.6])
COLOR_MAX = jnp.array([0.2, 0.5, 1.0])


@jaxtyped(typechecker=beartype)
def normalize(vectors: Float[Array, "... 2"]) -> Float[Array, "... 2"]:
    return vectors / jnp.maximum(jnp.linalg.norm(vectors, axis=-1, keepdims=True), EPS)


@chex.dataclass
@jaxtyped(typechecker=beartype)
class Flock:
    positions: Float[Array, "bird 2"]
    velocities: Float[Array, "bird 2"]
    colors: Float[Array, "bird 3"]


def initialize_flock(rng_key_1, rng_key_2, rng_key_3) -> Flock:
    ps = jax.random.uniform(
        rng_key_1,
        shape=(BIRD_COUNT, 2),
        minval=jnp.array([0, 0]),
        maxval=jnp.array([WINDOW_WIDTH, WINDOW_HEIGHT]),
    )
    vs = jax.random.uniform(
        rng_key_2,
        shape=(BIRD_COUNT, 2),
        minval=jnp.array([-1, -1]),
        maxval=jnp.array([1, 1]),
    )
    vs = normalize(vs)

    colors = jax.random.uniform(
        rng_key_3,
        shape=(BIRD_COUNT, 3),
        minval=COLOR_MIN,
        maxval=COLOR_MAX,
    )

    return Flock(positions=ps, velocities=vs, colors=colors)


def draw_flock_trails(
    surface: pygame.Surface,
    position_history: deque[np.ndarray],
    colors: np.ndarray,
) -> None:
    if len(position_history) < 2:
        return

    trail_layer = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    segment_count = len(position_history) - 1

    for segment_index, (previous_positions, positions) in enumerate(
        zip(position_history, list(position_history)[1:]), start=1
    ):
        freshness = segment_index / segment_count
        alpha = int(MIN_TRAIL_ALPHA + (MAX_TRAIL_ALPHA - MIN_TRAIL_ALPHA) * freshness)

        for previous_pos, bird_pos, bird_color in zip(
            previous_positions, positions, colors
        ):
            # Skip wrapped steps so toroidal screen edges do not create long lines
            # across the whole window.
            if (
                abs(float(bird_pos[0] - previous_pos[0])) > WINDOW_WIDTH / 2
                or abs(float(bird_pos[1] - previous_pos[1])) > WINDOW_HEIGHT / 2
            ):
                continue

            start = (int(previous_pos[0]), int(previous_pos[1]))
            end = (int(bird_pos[0]), int(bird_pos[1]))
            rgb = np.clip(bird_color * 255, 0, 255).astype(np.uint8)
            color = tuple(int(channel) for channel in rgb) + (alpha,)
            pygame.draw.line(trail_layer, color, start, end, TRAIL_WIDTH)

    surface.blit(trail_layer, (0, 0))


@jaxtyped(typechecker=beartype)
def gaussian_height_field(
    positions: Float[Array, "bird 2"],
) -> Float[Array, "height width"]:
    """Dense scalar field: sum of one Gaussian hill per bird.

    This is useful if you later want to draw/debug the field. The simulation
    below uses the analytic gradient at the bird positions instead of building
    this full 800x800 image every frame.
    """
    xs = jnp.arange(WINDOW_WIDTH, dtype=positions.dtype)
    ys = jnp.arange(WINDOW_HEIGHT, dtype=positions.dtype)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    grid = jnp.stack([grid_x, grid_y], axis=-1)  # height width 2

    offset = grid[:, :, None, :] - positions[None, None, :, :]
    dist_sq = jnp.sum(offset * offset, axis=-1)
    hills = jnp.exp(-dist_sq / (2.0 * SEPARATION_SIGMA**2))
    return jnp.sum(hills, axis=-1)


@jaxtyped(typechecker=beartype)
def pairwise_offsets(
    positions: Float[Array, "bird 2"],
) -> Float[Array, "bird bird 2"]:
    """Shortest wrapped offset from each neighbor to each sample bird.

    result[i, j] is positions[i] - positions[j], adjusted for the toroidal
    screen wrap used in update_flock.
    """
    offset = positions[:, None, :] - positions[None, :, :]
    world_size = jnp.array([WINDOW_WIDTH, WINDOW_HEIGHT], dtype=positions.dtype)
    return offset - world_size * jnp.round(offset / world_size)


@jaxtyped(typechecker=beartype)
def not_self_mask(count: int, dtype) -> Float[Array, "bird bird 1"]:
    return (1.0 - jnp.eye(count, dtype=dtype))[:, :, None]


@jaxtyped(typechecker=beartype)
def separation_field(
    positions: Float[Array, "bird 2"],
) -> Float[Array, "bird 2"]:
    """Return the downhill Gaussian-gradient vector at each bird.

    For a hill h_i(x) = exp(-||x - p_i||^2 / (2 sigma^2)),

        grad h_i(x) = -(x - p_i) / sigma^2 * h_i(x)

    The gradient points uphill, toward nearby birds, so separation uses
    -grad. The diagonal mask explicitly removes each bird's own hill.
    """
    offset = pairwise_offsets(positions)  # sample bird, hill bird, 2
    dist_sq = jnp.sum(offset * offset, axis=-1, keepdims=True)

    hills = jnp.exp(-dist_sq / (2.0 * SEPARATION_SIGMA**2))
    grad = -(offset / (SEPARATION_SIGMA**2)) * hills

    grad_without_self = grad * not_self_mask(positions.shape[0], positions.dtype)

    downhill = -jnp.sum(grad_without_self, axis=1)
    return downhill


@jaxtyped(typechecker=beartype)
def cohesion_field(
    positions: Float[Array, "bird 2"],
) -> Float[Array, "bird 2"]:
    """Attraction toward neighbors in a wider, ring-shaped influence band."""
    offset = pairwise_offsets(positions)
    distance = jnp.linalg.norm(offset, axis=-1, keepdims=True)

    # Peak attraction around COHESION_DISTANCE. Multiplying by a smooth
    # near-distance gate keeps cohesion from fighting separation up close.
    ring_weight = jnp.exp(
        -((distance - COHESION_DISTANCE) ** 2) / (2.0 * COHESION_SIGMA**2)
    )
    near_gate = 1.0 - jnp.exp(-(distance**2) / (2.0 * SEPARATION_SIGMA**2))
    weight = (
        ring_weight * near_gate * not_self_mask(positions.shape[0], positions.dtype)
    )

    direction_to_neighbor = -offset / jnp.maximum(distance, EPS)
    weighted_direction = jnp.sum(direction_to_neighbor * weight, axis=1)
    total_weight = jnp.sum(weight, axis=1)
    return weighted_direction / jnp.maximum(total_weight, EPS)


@jaxtyped(typechecker=beartype)
def alignment_field(
    positions: Float[Array, "bird 2"],
    velocities: Float[Array, "bird 2"],
) -> Float[Array, "bird 2"]:
    """Steer toward a locally weighted average heading.

    Each neighbor contributes its velocity through an oriented/elliptical
    Gaussian aligned to that neighbor's heading.
    """
    offset = pairwise_offsets(positions)
    neighbor_heading = velocities[None, :, :]

    along = jnp.sum(offset * neighbor_heading, axis=-1, keepdims=True)
    perpendicular = offset - along * neighbor_heading
    side_sq = jnp.sum(perpendicular * perpendicular, axis=-1, keepdims=True)

    weight = jnp.exp(
        -(along**2) / (2.0 * ALIGNMENT_LONG_SIGMA**2)
        - side_sq / (2.0 * ALIGNMENT_SIDE_SIGMA**2)
    )
    weight = weight * not_self_mask(positions.shape[0], positions.dtype)

    average_heading = jnp.sum(neighbor_heading * weight, axis=1)
    total_weight = jnp.sum(weight, axis=1)
    average_heading = average_heading / jnp.maximum(total_weight, EPS)

    # Return a steering vector, not an absolute velocity.
    return average_heading - velocities


@jaxtyped(typechecker=beartype)
def noise_field(
    positions: Float[Array, "bird 2"], time_seconds: float
) -> Float[Array, "bird 2"]:
    """Smooth pseudo-noise vector field for less perfectly settled flocks."""
    p = positions / NOISE_SCALE
    t = time_seconds * NOISE_SCROLL_SPEED

    # A few incommensurate sine/cosine waves make a cheap coherent vector
    # field. It is deterministic, smooth in space, and changes slowly in time.
    x = p[:, 0]
    y = p[:, 1]
    vx = jnp.sin(1.7 * x + 0.9 * y + t) + 0.5 * jnp.sin(-0.6 * x + 2.3 * y - 1.7 * t)
    vy = jnp.cos(1.1 * x - 1.5 * y - 0.8 * t) + 0.5 * jnp.cos(
        2.0 * x + 0.4 * y + 1.3 * t
    )

    return normalize(jnp.stack([vx, vy], axis=-1))


@jaxtyped(typechecker=beartype)
def limit_magnitude(
    vectors: Float[Array, "... 2"], max_magnitude: float
) -> Float[Array, "... 2"]:
    magnitudes = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, max_magnitude / jnp.maximum(magnitudes, EPS))
    return vectors * scale


def update_flock(dt: float, flock: Flock, time_seconds: float) -> Flock:
    separation = separation_field(flock.positions) * SEPARATION_STRENGTH
    cohesion = cohesion_field(flock.positions) * COHESION_STRENGTH
    alignment = alignment_field(flock.positions, flock.velocities) * ALIGNMENT_STRENGTH
    noise = noise_field(flock.positions, time_seconds) * NOISE_STRENGTH

    turn_force = separation + cohesion + alignment + noise
    turn_force = limit_magnitude(turn_force, MAX_TURN_FORCE)

    steering = flock.velocities + turn_force * dt
    velocities = normalize(steering)

    positions = flock.positions + velocities * dt * BIRD_VELOCITY
    positions = jnp.mod(positions, jnp.array([WINDOW_WIDTH, WINDOW_HEIGHT]))

    return Flock(positions=positions, velocities=velocities, colors=flock.colors)


def draw_fps(
    surface: pygame.Surface,
    font: pygame.Font,
    clock: pygame.time.Clock,
) -> None:
    fps_text = font.render(
        f"FPS: {clock.get_fps():.0f}",
        True,
        WHITE,
    )
    surface.blit(fps_text, (10, 10))


def main() -> None:
    pygame.init()

    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Flock Field")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)

        rng_key = jax.random.key(10)
        rng_key, k1 = jax.random.split(rng_key)
        rng_key, k2 = jax.random.split(rng_key)
        rng_key, k3 = jax.random.split(rng_key)

        flock = initialize_flock(k1, k2, k3)
        flock.positions.block_until_ready()
        position_history = deque(
            [np.asarray(flock.positions)], maxlen=TRAIL_HISTORY_LENGTH + 1
        )
        colors = np.asarray(flock.colors)

        screen.fill(BACKGROUND_COLOR)

        time_seconds = 0.0

        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0
            time_seconds += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            flock = update_flock(dt, flock, time_seconds)
            # JAX work can be asynchronous, so block here to wait for the
            # simulation step to finish before drawing its result.
            flock.positions.block_until_ready()
            position_history.append(np.asarray(flock.positions))

            screen.fill(BACKGROUND_COLOR)
            draw_flock_trails(screen, position_history, colors)
            draw_fps(screen, font, clock)
            pygame.display.flip()

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
