#!/usr/bin/env python3
"""Offline forest-fire rescue ensemble with a small FPSO allocator."""

import json
import math

import numpy as np


PAPER = (
    "Ensemble framework by using nature inspired algorithms for the early-stage "
    "forest fire rescue — A case study of dynamic optimization problems.pdf"
)
ALGORITHM = "Rescue Ensemble with FPSO"
SEED = 202510

FOREST, BURNING, BURNT, RESCUED = 2, 3, 4, 5
DIRECTIONS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
)


def key_cells(state: np.ndarray) -> np.ndarray:
    """Burning edge cells: burning cells adjacent to still-unburned forest."""
    rows, cols = state.shape
    keys = []
    for r, c in np.argwhere(state == BURNING):
        for dr, dc in DIRECTIONS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and state[rr, cc] == FOREST:
                keys.append((r, c))
                break
    if not keys:
        keys = [tuple(v) for v in np.argwhere(state == BURNING)]
    return np.asarray(keys, dtype=float)


def decode_targets(vector: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Map each continuous PSO waypoint to a distinct nearest rescue cell."""
    selected = []
    available = list(range(len(keys)))
    for point in vector.reshape(-1, 2):
        if not available:
            break
        distance = np.sum((keys[available] - point) ** 2, axis=1)
        chosen_position = int(np.argmin(distance))
        selected.append(available.pop(chosen_position))
    return keys[selected].astype(int)


def assignment_score(
    vector: np.ndarray,
    keys: np.ndarray,
    aircraft: np.ndarray,
    state: np.ndarray,
    wind_direction: np.ndarray,
) -> float:
    targets = decode_targets(vector, keys)
    if len(targets) == 0:
        return 1e6
    active_aircraft = aircraft[: len(targets)]
    distances = np.linalg.norm(targets - active_aircraft, axis=1)
    max_time = float(np.max(distances / 2.5))

    # Firefighting weighted selection: reward reachability, wind-head alignment,
    # and burning continuity in the 3x3 neighborhood (paper Eqs. 13-15).
    weighted_risk = 0.0
    ignition = np.array(state.shape, dtype=float) / 2.0
    for target, distance in zip(targets, distances):
        r, c = target
        r0, r1 = max(0, r - 1), min(state.shape[0], r + 2)
        c0, c1 = max(0, c - 1), min(state.shape[1], c + 2)
        continuity = float(np.count_nonzero(state[r0:r1, c0:c1] == BURNING))
        radial = target.astype(float) - ignition
        radial_norm = np.linalg.norm(radial)
        alignment = (
            float(np.dot(radial / radial_norm, wind_direction)) if radial_norm > 0 else 0.0
        )
        reachable = 1.0 if distance <= 7.5 else 0.0
        weighted_risk += 1.4 * continuity + 1.2 * max(0.0, alignment) + reachable

    burnt_cost_proxy = max(0.0, 12.0 - weighted_risk)
    duplicate_penalty = 5.0 * (len(targets) - len(np.unique(targets, axis=0)))
    # Equal-weight form of paper Equation (1), with synthetic normalizers.
    return 0.5 * max_time / 8.0 + 0.5 * burnt_cost_proxy / 12.0 + duplicate_penalty


def fpso_allocate(
    rng: np.random.Generator,
    keys: np.ndarray,
    aircraft: np.ndarray,
    state: np.ndarray,
    wind_direction: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    n_aircraft = min(len(aircraft), len(keys))
    dims, swarm_size, generations = 2 * n_aircraft, 22, 28
    lower = np.zeros(dims)
    upper = np.tile(np.array(state.shape, dtype=float) - 1.0, n_aircraft)
    positions = rng.uniform(lower, upper, size=(swarm_size, dims))
    velocities = rng.uniform(-1.0, 1.0, size=(swarm_size, dims))
    values = np.array(
        [assignment_score(p, keys, aircraft, state, wind_direction) for p in positions]
    )
    initial = float(np.min(values))
    pbest, pbest_values = positions.copy(), values.copy()
    g_index = int(np.argmin(values))
    gbest, gbest_value = positions[g_index].copy(), float(values[g_index])

    for generation in range(generations):
        inertia = 1.0 - 0.8 * generation / max(1, generations - 1)
        r1 = rng.random((swarm_size, dims))
        r2 = rng.random((swarm_size, dims))
        velocities = (
            inertia * velocities
            + 2.0 * r1 * (pbest - positions)
            + 2.0 * r2 * (gbest - positions)
        )
        velocities = np.clip(velocities, -8.0, 8.0)
        positions = np.clip(positions + velocities, lower, upper)
        values = np.array(
            [assignment_score(p, keys, aircraft, state, wind_direction) for p in positions]
        )
        improved = values < pbest_values
        pbest[improved], pbest_values[improved] = positions[improved], values[improved]
        index = int(np.argmin(pbest_values))
        if float(pbest_values[index]) < gbest_value:
            gbest, gbest_value = pbest[index].copy(), float(pbest_values[index])
    return decode_targets(gbest, keys), initial, gbest_value


def spread_fire(
    rng: np.random.Generator,
    state: np.ndarray,
    density: np.ndarray,
    rescued_targets: np.ndarray,
    wind_speed: float,
    wind_angle: float,
) -> np.ndarray:
    old = state.copy()
    new = old.copy()
    rescued = {tuple(v) for v in rescued_targets.tolist()}
    rows, cols = old.shape

    for r, c in np.argwhere(old == BURNING):
        protected = (int(r), int(c)) in rescued
        new[r, c] = RESCUED if protected else BURNT  # Rules R2A/R2B.
        if protected:
            continue
        for dr, dc in DIRECTIONS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols) or old[rr, cc] != FOREST:
                continue
            propagation_angle = math.atan2(dc, dr)
            theta = propagation_angle - wind_angle
            p_den = {-1: -0.4, 0: 0.0, 1: 0.3}[int(density[rr, cc]) - 2]
            p_wind = math.exp(0.045 * wind_speed) * math.exp(
                wind_speed * 0.131 * (math.cos(theta) - 1.0)
            )
            p_burn = float(np.clip(0.58 * (1.0 + p_den) * p_wind, 0.0, 1.0))
            if rng.random() < p_burn:
                new[rr, cc] = BURNING
    return new


def main() -> None:
    rng = np.random.default_rng(SEED)
    shape = (15, 15)
    state = np.full(shape, FOREST, dtype=int)
    state[6:9, 6:9] = BURNING
    density = rng.integers(1, 4, size=shape)
    aircraft = np.array([[0.0, 2.0], [1.0, 7.0], [2.0, 12.0]])
    wind_angle = math.pi / 4.0
    wind_direction = np.array([math.cos(wind_angle), math.sin(wind_angle)])
    initial_burning = int(np.count_nonzero(state == BURNING))
    rescued_total = 0
    first_raw, last_optimized = 0.0, 0.0
    distinct_targets = True
    cycles_run = 0

    for cycle in range(9):
        keys = key_cells(state)
        if len(keys) == 0:
            break
        targets, raw_score, optimized_score = fpso_allocate(
            rng, keys, aircraft, state, wind_direction
        )
        if cycle == 0:
            first_raw = raw_score
        last_optimized = optimized_score
        distinct_targets &= len(targets) == len(np.unique(targets, axis=0))
        rescued_total += len(targets)
        state = spread_fire(rng, state, density, targets, 5.0, wind_angle)
        aircraft[: len(targets)] = targets
        cycles_run += 1

    result = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "grid_cells": int(np.prod(shape)),
            "cycles": cycles_run,
            "initial_burning_cells": initial_burning,
            "final_burning_cells": int(np.count_nonzero(state == BURNING)),
            "rescued_cells": int(np.count_nonzero(state == RESCUED)),
            "burned_cells": int(np.count_nonzero(state == BURNT)),
            "aircraft_assignments": rescued_total,
            "first_population_best_score": first_raw,
            "last_optimized_score": last_optimized,
        },
        "checks": {
            "finite": bool(np.isfinite(first_raw) and np.isfinite(last_optimized)),
            "valid_state_codes": bool(np.isin(state, [FOREST, BURNING, BURNT, RESCUED]).all()),
            "targets_distinct_within_cycle": bool(distinct_targets),
            "simulator_advanced": bool(cycles_run > 0),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
