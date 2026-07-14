"""Offline artistic renderer: fading bird trails via an accumulation buffer.

Runs the flock simulation with jax.lax.scan and rasterizes every frame on
device. Each frame the trail buffer decays exponentially, then every bird
deposits a small Gaussian sprite colored by its heading. Frames are streamed
as raw RGB to ffmpeg, so the only external requirement is an ffmpeg binary.

    uv run python render_trails.py trails.mp4

All configuration lives in the constants below.
"""

import argparse
import math
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp

from flock_sim import initialize_flock, make_update_step, simulation_to_grid

SIM_PARAMS = dict(
    simulation_grid_size=128,
    dt=0.016,
    speed=0.4,
    separation_strength=0.5,
    turn_rate=0.3,
    sigma=0.09,
    cohesion_strength=0.10,
    cohesion_sigma=0.15,
    alignment_strength=0.05,
    alignment_sigma=0.08,
    boundary_margin=0.3,
    boundary_strength=4.0,
    noise_strength=0.4,
    noise_grid_size=1,
    noise_temporal_rate=0.025,
)

POPULATION = 500
SEED = 10
WARMUP_STEPS = 300  # simulation steps before recording

SECONDS = 60.0
FPS = 60
SIZE = 900  # output resolution (px); must be even for yuv420p

BACKGROUND = jnp.array([8.0, 10.0, 14.0]) / 255.0
TRAIL_SATURATION = 0.8
TRAIL_HALF_LIFE = 0.8  # seconds for a trail to fade to half brightness
TRAIL_SIGMA_PX = 1.6  # trail sprite radius (px std dev)
EXPOSURE = 0.4  # tonemap gain; higher saturates trails sooner
DEPOSITS_PER_FRAME = 4  # trail deposits along each bird's movement segment

CHUNK = 32  # frames per scan chunk


def hsv_to_rgb(hue, saturation, value):
    offsets = jnp.array([0.0, 4.0, 2.0])
    channels = jnp.abs((hue[..., None] * 6.0 + offsets) % 6.0 - 3.0) - 1.0
    rgb = jnp.clip(channels, 0.0, 1.0)
    return value[..., None] * (1.0 - saturation + saturation * rgb)


def splat_sprites(positions, colors, render_size, sigma_px):
    """Scatter-add a Gaussian sprite per bird into an RGB buffer."""
    radius = max(1, math.ceil(3 * sigma_px))
    grid_positions = simulation_to_grid(positions, render_size)
    centers = jnp.rint(grid_positions).astype(jnp.int32)

    axis = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    offset_x, offset_y = jnp.meshgrid(axis, axis, indexing="xy")
    index_offsets = jnp.stack([offset_x.ravel(), offset_y.ravel()], axis=-1)

    cell_indices = centers[:, None, :] + index_offsets[None, :, :]
    pixel_offsets = cell_indices.astype(positions.dtype) - grid_positions[:, None, :]
    distance_squared = jnp.sum(pixel_offsets * pixel_offsets, axis=-1)
    weights = jnp.exp(-distance_squared / (2 * sigma_px * sigma_px))
    values = weights[:, :, None] * colors[:, None, :]

    scatter_indices = jnp.where(cell_indices < 0, render_size, cell_indices)
    buffer = jnp.zeros((render_size, render_size, 3), dtype=positions.dtype)
    return buffer.at[scatter_indices[:, :, 1], scatter_indices[:, :, 0]].add(
        values, mode="drop"
    )


def heading_colors(headings):
    hue = (jnp.arctan2(headings[:, 1], headings[:, 0]) + jnp.pi) / (2 * jnp.pi)
    return hsv_to_rgb(hue, TRAIL_SATURATION, jnp.ones_like(hue))


def make_render_chunk(update_step):
    decay = 0.5 ** (1.0 / (FPS * TRAIL_HALF_LIFE))
    # Deposit along the segment travelled this frame (birds move several
    # pixels per frame, a single sprite per frame reads as beads).
    fractions = (
        jnp.arange(1, DEPOSITS_PER_FRAME + 1, dtype=jnp.float32) / DEPOSITS_PER_FRAME
    )

    def step(carry, _):
        flock, trail = carry
        previous_positions = flock.positions
        flock = update_step(flock)
        segment_positions = (
            previous_positions[None, :, :]
            + fractions[:, None, None]
            * (flock.positions - previous_positions)[None, :, :]
        ).reshape(-1, 2)
        segment_colors = jnp.tile(
            heading_colors(flock.headings), (DEPOSITS_PER_FRAME, 1)
        )
        deposit = splat_sprites(
            segment_positions,
            segment_colors / DEPOSITS_PER_FRAME,
            SIZE,
            TRAIL_SIGMA_PX,
        )
        trail = trail * decay + deposit

        glow = 1.0 - jnp.exp(-EXPOSURE * trail)
        rgb = jnp.clip(BACKGROUND + glow, 0.0, 1.0)
        frame = jnp.round(rgb * 255).astype(jnp.uint8)
        return (flock, trail), frame

    def run(flock, trail):
        return jax.lax.scan(step, (flock, trail), None, length=CHUNK)

    return jax.jit(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default="trails.mp4")
    args = parser.parse_args()

    if SIZE % 2:
        sys.exit("SIZE must be even (yuv420p requires even dimensions)")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")

    update_step = make_update_step(**SIM_PARAMS)
    flock = initialize_flock(jax.random.key(SEED), POPULATION)

    warmup = jax.jit(
        lambda flock: jax.lax.fori_loop(
            0, WARMUP_STEPS, lambda _, state: update_step(state), flock
        )
    )
    flock = warmup(flock)

    render_chunk = make_render_chunk(update_step)
    trail = jnp.zeros((SIZE, SIZE, 3), dtype=flock.positions.dtype)

    total_frames = round(SECONDS * FPS)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{SIZE}x{SIZE}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            args.output,
        ],
        stdin=subprocess.PIPE,
    )

    written = 0
    try:
        while written < total_frames:
            (flock, trail), frames = render_chunk(flock, trail)
            frames = jax.device_get(frames)[: total_frames - written]
            encoder.stdin.write(frames.tobytes())
            written += len(frames)
            print(f"\r{written}/{total_frames} frames", end="", flush=True)
    finally:
        encoder.stdin.close()
        encoder.wait()
    print()

    if encoder.returncode:
        sys.exit(f"ffmpeg exited with status {encoder.returncode}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
