# flock-field

A field-based flocking (boids) simulation in [JAX](https://github.com/jax-ml/jax).

Instead of computing pairwise interactions between birds, each steering
behavior is expressed as a 2D vector field on a shared grid: every bird splats
a local kernel (separation, cohesion, alignment) onto the grid, global fields
(a radial boundary, time-varying noise) are added on top, and each bird then
steers by sampling the combined field at its own position. The whole update step
is JIT-compiled and runs at hundreds of steps per second for a 1000-bird flock
on CPU, and faster still on GPU (see`benchmark.ipynb`).

The clip below shows the combined field (left) alongside an artistic view of
the flock, where each bird leaves a fading trail.

https://github.com/user-attachments/assets/f51b3d69-e78f-4637-bfc4-e0676e773c77

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run jupyter lab
```

- `flock_sim.py` — the simulation itself: `Flock` state, field/kernel functions, and `make_update_step` to build a jitted update
- `flock_field.ipynb` — live animated simulation rendered with ipycanvas (the
rendering step is horrendously slow, but useful for viewing the component fields individually)
- `render_trails.py` — configurable offline renderer that produced the clip above (requires ffmpeg)
- `benchmark.ipynb` — step-time benchmarks across flock size, grid size, and kernel radius
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanreierson/flock-field/blob/main/benchmark.ipynb)

## Tests

Golden-snapshot tests verify the simulation stays deterministic:

```sh
uv run pytest
```

## License

[MIT](LICENSE)
