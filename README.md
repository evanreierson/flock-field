# flock-field

A field-based flocking (boids) simulation in [JAX](https://github.com/jax-ml/jax).

Instead of computing pairwise interactions between birds, each steering
behavior is expressed as a 2D vector field on a shared grid: every bird splats
a local kernel (separation, cohesion, alignment) onto the grid, global fields
(a radial boundary, optional time-varying noise) are added on top, and each
bird then steers by bilinearly sampling the combined field at its own
position. The whole update step is JIT-compiled and runs at hundreds of steps
per second for a 1000-bird flock on CPU (see `benchmark.ipynb`).

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run jupyter lab
```

- `flock_field.ipynb` — live animated simulation rendered with ipycanvas
- `benchmark.ipynb` — step-time benchmarks across flock size, grid size, and kernel radius
- `flock_sim.py` — the simulation itself: `Flock` state, field/kernel functions, and `make_update_step` to build a jitted update

## Tests

Golden-snapshot tests verify the simulation stays deterministic:

```sh
uv run pytest
```

## License

[MIT](LICENSE)
