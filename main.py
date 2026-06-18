from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyray as rl

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

BIRD_COUNT = 100
BIRD_TEXTURE_SIZE = 64
BIRD_BASE_SIZE = 32.0
BIRD_VELOCITY = 100.0


@dataclass
class Bird:
    position: rl.Vector2
    orientation: float
    flap_phase: float
    scale: float
    color: rl.Color


def random_bird() -> Bird:
    return Bird(
        position=rl.Vector2(
            float(rl.get_random_value(0, WINDOW_WIDTH)),
            float(rl.get_random_value(0, WINDOW_HEIGHT)),
        ),
        orientation=float(rl.get_random_value(0, 359)),
        flap_phase=float(rl.get_random_value(0, 999)) / 1000.0,
        scale=float(rl.get_random_value(80, 120)) / 100.0,
        color=rl.Color(
            rl.get_random_value(0, 255),
            rl.get_random_value(0, 255),
            rl.get_random_value(0, 255),
            255,
        ),
    )


def draw_birds(
    birds: list[Bird],
    bird_texture: rl.Texture2D,
    bird_shader: rl.Shader,
    flap_phase_loc: int,
) -> None:
    for bird in birds:
        size = BIRD_BASE_SIZE * bird.scale

        if flap_phase_loc >= 0:
            rl.set_shader_value(
                bird_shader,
                flap_phase_loc,
                rl.ffi.new("float *", bird.flap_phase),
                rl.SHADER_UNIFORM_FLOAT,
            )

        rl.begin_shader_mode(bird_shader)
        rl.draw_texture_pro(
            bird_texture,
            rl.Rectangle(0, 0, float(bird_texture.width), float(bird_texture.height)),
            rl.Rectangle(bird.position.x, bird.position.y, size, size),
            rl.Vector2(size / 2, size / 2),
            bird.orientation,
            bird.color,
        )
        rl.end_shader_mode()


def update_birds(dt: float, birds: list[Bird]) -> None:
    # separation: steer to avoid crowding local flockmates
    # alignment: steer towards the average heading of local flockmates
    # cohesion: steer to move towards the average position of local flockmates
    pass


def main() -> None:
    rl.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Boids")

    bird_shader = None
    bird_texture = None
    bird_target = None

    try:
        shader_path = Path("bird.glsl")
        if not shader_path.exists():
            shader_path = Path("boids/bird.glsl")

        bird_shader = rl.load_shader(rl.ffi.NULL, str(shader_path))

        bird_image = rl.gen_image_color(BIRD_TEXTURE_SIZE, BIRD_TEXTURE_SIZE, rl.WHITE)
        bird_texture = rl.load_texture_from_image(bird_image)
        rl.unload_image(bird_image)

        bird_target = rl.load_render_texture(WINDOW_WIDTH, WINDOW_HEIGHT)

        birds = [random_bird() for _ in range(BIRD_COUNT)]

        time_loc = rl.get_shader_location(bird_shader, "time")
        flap_phase_loc = rl.get_shader_location(bird_shader, "flap_phase")

        while not rl.window_should_close():
            time = float(rl.get_time())
            if time_loc >= 0:
                rl.set_shader_value(
                    bird_shader,
                    time_loc,
                    rl.ffi.new("float *", time),
                    rl.SHADER_UNIFORM_FLOAT,
                )

            rl.begin_drawing()
            rl.clear_background(rl.DARKGRAY)

            rl.begin_texture_mode(bird_target)
            rl.clear_background(rl.BLANK)
            draw_birds(birds, bird_texture, bird_shader, flap_phase_loc)
            rl.end_texture_mode()

            rl.draw_texture_rec(
                bird_target.texture,
                rl.Rectangle(0, 0, float(WINDOW_WIDTH), -float(WINDOW_HEIGHT)),
                rl.Vector2(0, 0),
                rl.WHITE,
            )

            rl.draw_fps(10, 10)
            rl.end_drawing()
    finally:
        if bird_target is not None:
            rl.unload_render_texture(bird_target)
        if bird_texture is not None:
            rl.unload_texture(bird_texture)
        if bird_shader is not None:
            rl.unload_shader(bird_shader)
        rl.close_window()


if __name__ == "__main__":
    main()
