from __future__ import annotations

import time

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

BIRD_COUNT = 1000
BIRD_BASE_SIZE = 16.0
BIRD_VELOCITY = 100.0
FPS = 60
TIMING_REPORT_INTERVAL = 1.0

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
HEIGHT_FIELD_CONTRAST = 255.0
EPS = 1e-8

BACKGROUND_COLOR = pygame.Color(48, 48, 48)
WHITE = pygame.Color(255, 255, 255)


@jaxtyped(typechecker=beartype)
def normalize(vectors: Float[Array, "... 2"]) -> Float[Array, "... 2"]:
    return vectors / jnp.maximum(jnp.linalg.norm(vectors, axis=-1, keepdims=True), EPS)


@chex.dataclass
@jaxtyped(typechecker=beartype)
class Flock:
    positions: Float[Array, "bird 2"]
    velocities: Float[Array, "bird 2"]


def initialize_flock(rng_key_1, rng_key_2) -> Flock:
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

    return Flock(positions=ps, velocities=vs)


def draw_height_field(surface: pygame.Surface, flock: Flock) -> None:
    field = gaussian_height_field(flock.positions)

    # Normalize per-frame so the field is visible even when birds are sparse.
    field = field - jnp.min(field)
    field = field / jnp.maximum(jnp.max(field), EPS)

    # Pygame surfarray wants width x height x channels, while our field is
    # height x width. Blue hills keep red boids readable.
    brightness = np.asarray(field * HEIGHT_FIELD_CONTRAST, dtype=np.uint8).T
    pixels = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT, 3), dtype=np.uint8)
    pixels[:, :, 2] = brightness
    pixels[:, :, 1] = brightness // 3

    field_surface = pygame.surfarray.make_surface(pixels)
    surface.blit(field_surface, (0, 0))


def draw_flock(surface: pygame.Surface, flock: Flock) -> None:
    def triangle_points(
        position: pygame.Vector2, orientation: float, size: float
    ) -> list[pygame.Vector2]:
        local_points = [
            pygame.Vector2(size * 0.70, 0.0),
            pygame.Vector2(-size * 0.45, -size * 0.30),
            pygame.Vector2(-size * 0.45, size * 0.30),
        ]

        return [position + point.rotate(orientation) for point in local_points]

    for bird_pos, bird_velocity in zip(flock.positions, flock.velocities):
        position = pygame.Vector2(float(bird_pos[0]), float(bird_pos[1]))
        orientation = float(
            jnp.degrees(jnp.arctan2(bird_velocity[1], bird_velocity[0]))
        )

        pygame.draw.polygon(
            surface,
            (255, 0, 0),
            triangle_points(position, orientation, BIRD_BASE_SIZE),
        )


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

    # separation: steer to avoid crowding local flockmates
    # alignment: steer towards the average heading of local flockmates
    # cohesion: steer to move towards the average position of local flockmates
    return Flock(positions=positions, velocities=velocities)


def draw_fps(
    surface: pygame.Surface,
    font: pygame.Font,
    clock: pygame.time.Clock,
    update_ms: float,
    draw_ms: float,
) -> None:
    fps_text = font.render(
        f"FPS: {clock.get_fps():.0f}  update: {update_ms:.2f}ms  draw: {draw_ms:.2f}ms",
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
        # birds = [random_bird() for _ in range(BIRD_COUNT)]

        rng_key = jax.random.key(10)
        rng_key, k1 = jax.random.split(rng_key)
        rng_key, k2 = jax.random.split(rng_key)

        flock = initialize_flock(k1, k2)

        print(flock.positions)
        print()
        print(flock.velocities)

        time_seconds = 0.0
        update_ms = 0.0
        draw_ms = 0.0
        update_ms_samples: list[float] = []
        draw_ms_samples: list[float] = []
        last_timing_report = time.perf_counter()

        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0
            time_seconds += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            update_start = time.perf_counter()
            flock = update_flock(dt, flock, time_seconds)
            # JAX work can be asynchronous, so block here to measure actual
            # simulation time instead of just dispatch time.
            flock.positions.block_until_ready()
            update_ms = (time.perf_counter() - update_start) * 1000.0

            draw_start = time.perf_counter()
            screen.fill(BACKGROUND_COLOR)
            # draw_height_field(screen, flock)
            draw_flock(screen, flock)
            draw_fps(screen, font, clock, update_ms, draw_ms)
            pygame.display.flip()
            draw_ms = (time.perf_counter() - draw_start) * 1000.0

            update_ms_samples.append(update_ms)
            draw_ms_samples.append(draw_ms)
            now = time.perf_counter()
            if now - last_timing_report >= TIMING_REPORT_INTERVAL:
                avg_update_ms = sum(update_ms_samples) / len(update_ms_samples)
                avg_draw_ms = sum(draw_ms_samples) / len(draw_ms_samples)
                print(
                    f"avg over {len(update_ms_samples)} frames: "
                    f"update={avg_update_ms:.2f}ms, "
                    f"draw={avg_draw_ms:.2f}ms, "
                    f"fps={clock.get_fps():.0f}"
                )
                update_ms_samples.clear()
                draw_ms_samples.clear()
                last_timing_report = now
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
