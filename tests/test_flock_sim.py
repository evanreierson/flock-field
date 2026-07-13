"""Characterization tests for the core simulation step.

These pin the output of `make_update_step` against golden snapshots so
optimizations can be checked for behavioral changes. Regenerate the
snapshots (only after an intentional behavior change) with:

    uv run python -m tests.test_flock_sim
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flock_sim import initialize_flock, make_update_step

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"

BASE_CONFIG = dict(
    simulation_grid_size=128,
    dt=0.016,
    speed=1.0,
    separation_strength=0.5,
    turn_rate=0.3,
    sigma=0.09,
    cohesion_strength=0.10,
    cohesion_sigma=0.15,
    alignment_strength=0.05,
    alignment_sigma=0.15,
    boundary_strength=8.0,
)

CONFIGS = {
    "base": BASE_CONFIG,
    "noise": {**BASE_CONFIG, "noise_strength": 0.3, "noise_grid_size": 1},
}


def run_simulation(config, population=100, steps=10, seed=10):
    flock = initialize_flock(jax.random.key(seed), population=population)
    step = make_update_step(**config)
    for _ in range(steps):
        flock = step(flock)
    return flock


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_update_step_matches_golden(name):
    flock = run_simulation(CONFIGS[name])
    golden = np.load(GOLDEN_DIR / f"{name}.npz")
    np.testing.assert_allclose(
        flock.positions, golden["positions"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(flock.headings, golden["headings"], rtol=1e-5, atol=1e-6)
    assert int(flock.generation) == int(golden["generation"])


def test_update_step_invariants():
    flock = run_simulation(CONFIGS["base"])
    assert jnp.all(jnp.abs(flock.positions) <= 1)
    np.testing.assert_allclose(
        jnp.linalg.norm(flock.headings, axis=-1), 1.0, rtol=1e-5
    )


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(exist_ok=True)
    for name, config in CONFIGS.items():
        flock = run_simulation(config)
        np.savez(
            GOLDEN_DIR / f"{name}.npz",
            positions=np.asarray(flock.positions),
            headings=np.asarray(flock.headings),
            generation=np.asarray(flock.generation),
        )
        print(f"wrote {GOLDEN_DIR / f'{name}.npz'}")
