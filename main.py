from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pygame
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

BIRD_COUNT = 100
BIRD_BASE_SIZE = 16.0
BIRD_VELOCITY = 100.0
FPS = 60

BACKGROUND_COLOR = pygame.Color(48, 48, 48)
WHITE = pygame.Color(255, 255, 255)


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
    eps = 1e-8
    vs = vs / jnp.maximum(jnp.linalg.norm(vs, axis=-1, keepdims=True), eps)

    return Flock(positions=ps, velocities=vs)


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


def update_flock(dt: float, flock: Flock) -> Flock:
    flock.positions = flock.positions + flock.velocities * dt * 100
    # separation: steer to avoid crowding local flockmates
    # alignment: steer towards the average heading of local flockmates
    # cohesion: steer to move towards the average position of local flockmates

    return flock


def draw_fps(
    surface: pygame.Surface, font: pygame.Font, clock: pygame.time.Clock
) -> None:
    fps_text = font.render(f"FPS: {clock.get_fps():.0f}", True, WHITE)
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

        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            flock = update_flock(dt, flock)

            screen.fill(BACKGROUND_COLOR)
            draw_flock(screen, flock)
            draw_fps(screen, font, clock)
            pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
