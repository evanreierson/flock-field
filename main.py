from __future__ import annotations

import random
from dataclasses import dataclass

import pygame

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

BIRD_COUNT = 100
BIRD_BASE_SIZE = 16.0
BIRD_VELOCITY = 100.0
FPS = 60

BACKGROUND_COLOR = pygame.Color(48, 48, 48)
WHITE = pygame.Color(255, 255, 255)


@dataclass
class Bird:
    position: pygame.Vector2
    orientation: float
    scale: float
    color: pygame.Color


def random_bird() -> Bird:
    return Bird(
        position=pygame.Vector2(
            random.uniform(0, WINDOW_WIDTH),
            random.uniform(0, WINDOW_HEIGHT),
        ),
        orientation=random.uniform(0, 360),
        scale=random.uniform(0.8, 1.2),
        color=pygame.Color(
            random.randrange(256),
            random.randrange(256),
            random.randrange(256),
        ),
    )


def triangle_points(
    position: pygame.Vector2, orientation: float, size: float
) -> list[pygame.Vector2]:
    local_points = [
        pygame.Vector2(size * 0.70, 0.0),
        pygame.Vector2(-size * 0.45, -size * 0.30),
        pygame.Vector2(-size * 0.45, size * 0.30),
    ]

    return [position + point.rotate(orientation) for point in local_points]


def draw_birds(surface: pygame.Surface, birds: list[Bird]) -> None:
    for bird in birds:
        size = BIRD_BASE_SIZE * bird.scale
        pygame.draw.polygon(
            surface,
            bird.color,
            triangle_points(bird.position, bird.orientation, size),
        )


def update_birds(dt: float, birds: list[Bird]) -> None:
    # separation: steer to avoid crowding local flockmates
    # alignment: steer towards the average heading of local flockmates
    # cohesion: steer to move towards the average position of local flockmates
    pass


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
        birds = [random_bird() for _ in range(BIRD_COUNT)]

        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            update_birds(dt, birds)

            screen.fill(BACKGROUND_COLOR)
            draw_birds(screen, birds)
            draw_fps(screen, font, clock)
            pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
