#!/usr/bin/env python3
"""MCMOPSO-RL core reconstruction with per-particle Q tables."""

import json
import math

import numpy as np


PAPER = "Multi-objective particle swarm optimization with multi-mode.pdf"
ALGORITHM = "MCMOPSO-RL"
SEED = 202514


def objectives(position: np.ndarray) -> np.ndarray:
    """Path length and safety/smoothness cost for fixed-x 2-D UAV paths."""
    position = np.atleast_2d(position)
    count, dimensions = position.shape
    x = np.linspace(0.0, 1.0, dimensions + 2)
    y = np.column_stack(
        (np.full(count, 0.10), position, np.full(count, 0.90))
    )
    dx = np.diff(x)
    dy = np.diff(y, axis=1)
    length = np.sum(np.sqrt(dx[None, :] ** 2 + dy**2), axis=1)

    waypoints = np.stack(
        (
            np.broadcast_to(x[None, :], y.shape),
            y,
        ),
        axis=-1,
    )
    threats = np.array([[0.42, 0.43, 0.16], [0.69, 0.70, 0.14]])
    risk = np.zeros(count)
    for tx, ty, radius in threats:
        squared = (waypoints[:, :, 0] - tx) ** 2 + (waypoints[:, :, 1] - ty) ** 2
        risk += np.sum(np.exp(-squared / (2.0 * radius**2)), axis=1)
    smoothness = np.sum(np.abs(np.diff(y, n=2, axis=1)), axis=1)
    safety_cost = risk + 0.35 * smoothness
    return np.column_stack((length, safety_cost))


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def crowding(values: np.ndarray) -> np.ndarray:
    n = len(values)
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective])
        distance[order[0]] = distance[order[-1]] = np.inf
        extent = values[order[-1], objective] - values[order[0], objective]
        if extent > 1e-14:
            distance[order[1:-1]] += (
                values[order[2:], objective] - values[order[:-2], objective]
            ) / extent
    return distance


def update_archive(
    positions: list[np.ndarray],
    values: list[np.ndarray],
    candidates: np.ndarray,
    candidate_values: np.ndarray,
    capacity: int = 60,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    for candidate, value in zip(candidates, candidate_values):
        rejected = any(
            dominates(old, value) or np.allclose(old, value, atol=1e-12)
            for old in values
        )
        if rejected:
            continue
        keep = [not dominates(value, old) for old in values]
        positions = [x for x, flag in zip(positions, keep) if flag]
        values = [f for f, flag in zip(values, keep) if flag]
        positions.append(candidate.copy())
        values.append(value.copy())
        while len(values) > capacity:
            distances = crowding(np.asarray(values))
            finite = np.where(np.isfinite(distances))[0]
            remove = (
                int(finite[np.argmin(distances[finite])])
                if len(finite)
                else len(values) // 2
            )
            positions.pop(remove)
            values.pop(remove)
    return positions, values


def hypervolume_2d(values: np.ndarray, reference: tuple[float, float]) -> float:
    valid = values[(values[:, 0] <= reference[0]) & (values[:, 1] <= reference[1])]
    valid = valid[np.argsort(valid[:, 0])]
    area, previous_y = 0.0, reference[1]
    for x, y in valid:
        if y < previous_y:
            area += (reference[0] - x) * (previous_y - y)
            previous_y = y
    return float(area)


def main() -> None:
    rng = np.random.default_rng(SEED)
    particles, dimensions, iterations = 30, 6, 60
    positions = rng.uniform(0.0, 1.0, size=(particles, dimensions))
    velocities = rng.normal(0.0, 0.08, size=(particles, dimensions))
    values = objectives(positions)
    pbest, pbest_values = positions.copy(), values.copy()
    archive_positions: list[np.ndarray] = []
    archive_values: list[np.ndarray] = []
    archive_positions, archive_values = update_archive(
        archive_positions, archive_values, positions, values
    )
    initial_hv = hypervolume_2d(np.asarray(archive_values), (4.0, 10.0))

    q_tables = np.zeros((particles, 3, 3))
    states = np.full(particles, 2, dtype=int)
    mode_counts = np.zeros(3, dtype=int)
    positive_rewards = 0

    for iteration in range(iterations):
        new_positions = np.empty_like(positions)
        new_velocities = velocities.copy()
        mean_best = np.mean(pbest, axis=0)
        learning_rate = 0.15 + 0.65 * 0.5 * (
            1.0 - math.cos(math.pi * (1.0 - iteration / iterations))
        )

        actions = np.empty(particles, dtype=int)
        for i in range(particles):
            if rng.random() < 0.10:
                action = int(rng.integers(3))
            else:
                row = q_tables[i, states[i]]
                maxima = np.flatnonzero(np.isclose(row, np.max(row)))
                action = int(rng.choice(maxima))
            actions[i] = action
            mode_counts[action] += 1
            leader = archive_positions[int(rng.integers(len(archive_positions)))]

            if action in (0, 1):
                if action == 0:  # Equation (11): exploration.
                    c_self, c_archive = 2.5, 0.5
                else:  # Equation (12): exploitation.
                    c_self, c_archive = 0.5, 2.5
                gaussian = rng.normal(size=dimensions)
                new_velocities[i] = (
                    0.70 * velocities[i]
                    + c_self * rng.random(dimensions) * (pbest[i] - positions[i])
                    + c_archive * gaussian * (leader - positions[i])
                )
                new_velocities[i] = np.clip(new_velocities[i], -0.30, 0.30)
                candidate = positions[i] + new_velocities[i]
            else:
                alpha = 1.0 / (
                    1.0 + math.exp(4.0 - 20.0 * iteration / iterations)
                )
                if rng.random() < alpha:
                    phi = rng.random(dimensions)
                    attractor = phi * pbest[i] + (1.0 - phi) * leader
                    beta = 1.0 - 0.5 * iteration / iterations
                    sign = np.where(rng.random(dimensions) >= 0.5, 1.0, -1.0)
                    candidate = attractor + sign * beta * np.abs(
                        mean_best - positions[i]
                    ) * np.log(1.0 / np.maximum(rng.random(dimensions), 1e-12))
                else:
                    random_particle = positions[int(rng.integers(particles))]
                    candidate = positions[i] + 0.5 * (leader - random_particle)
                new_velocities[i] *= 0.5
            new_positions[i] = np.clip(candidate, 0.0, 1.0)

        new_values = objectives(new_positions)
        for i in range(particles):
            old_sum = float(np.sum(values[i]))
            new_sum = float(np.sum(new_values[i]))
            improved = dominates(new_values[i], values[i]) or (
                not dominates(values[i], new_values[i]) and new_sum < old_sum
            )
            reward = 1.0 if improved else -1.0
            positive_rewards += int(improved)
            next_state = int(actions[i])
            state, action = int(states[i]), int(actions[i])
            q_old = q_tables[i, state, action]
            q_tables[i, state, action] = q_old + learning_rate * (
                reward + 0.80 * np.max(q_tables[i, next_state]) - q_old
            )
            states[i] = next_state

            pbest_improved = dominates(new_values[i], pbest_values[i]) or (
                not dominates(pbest_values[i], new_values[i])
                and new_sum < float(np.sum(pbest_values[i]))
            )
            if pbest_improved:
                pbest[i] = new_positions[i]
                pbest_values[i] = new_values[i]

        positions, velocities, values = new_positions, new_velocities, new_values
        archive_positions, archive_values = update_archive(
            archive_positions, archive_values, positions, values
        )

    archive_array = np.asarray(archive_values)
    final_hv = hypervolume_2d(archive_array, (4.0, 10.0))
    archive_nondominated = all(
        not dominates(archive_array[j], archive_array[i])
        for i in range(len(archive_array))
        for j in range(len(archive_array))
        if i != j
    )
    result = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "particles": particles,
            "waypoints": dimensions,
            "iterations": iterations,
            "archive_size": len(archive_values),
            "initial_hypervolume": initial_hv,
            "final_hypervolume": final_hv,
            "minimum_path_length": float(np.min(archive_array[:, 0])),
            "minimum_safety_cost": float(np.min(archive_array[:, 1])),
            "exploration_updates": int(mode_counts[0]),
            "exploitation_updates": int(mode_counts[1]),
            "hybrid_updates": int(mode_counts[2]),
            "positive_rewards": positive_rewards,
            "mean_absolute_q": float(np.mean(np.abs(q_tables))),
        },
        "checks": {
            "finite": bool(
                np.isfinite(archive_array).all() and np.isfinite(q_tables).all()
            ),
            "archive_nondominated": bool(archive_nondominated),
            "all_modes_exercised": bool(np.all(mode_counts > 0)),
            "q_tables_updated": bool(np.any(np.abs(q_tables) > 0.0)),
            "bounds_respected": bool(np.all((positions >= 0.0) & (positions <= 1.0))),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
