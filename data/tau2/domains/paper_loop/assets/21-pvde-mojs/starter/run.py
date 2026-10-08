#!/usr/bin/env python3
"""CPU/NumPy reconstruction of the serial PVDE-MOJS numerical core."""

from __future__ import annotations

import json

import numpy as np


SEED = 20251021
RNG = np.random.default_rng(SEED)


def logistic_population(size: int, dimensions: int, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    chaos = RNG.uniform(0.05, 0.95, size=(size, dimensions))
    for _ in range(12):
        chaos = 4.0 * chaos * (1.0 - chaos)
    return low + chaos * (high - low)


START = np.array([0.0, 0.0, 2.0])
END = np.array([12.0, 12.0, 2.0])
OBSTACLES = (
    (np.array([4.0, 5.0, 2.0]), 1.8),
    (np.array([8.0, 7.0, 2.5]), 2.0),
    (np.array([6.0, 10.0, 2.0]), 1.2),
)


def path_points(flat: np.ndarray) -> np.ndarray:
    return np.vstack((START, flat.reshape(-1, 3), END))


def objectives(flat: np.ndarray) -> np.ndarray:
    points = path_points(flat)
    segments = np.diff(points, axis=0)
    length = float(np.linalg.norm(segments, axis=1).sum()) / 20.0
    samples = np.concatenate(
        [a + np.linspace(0.0, 1.0, 8)[:, None] * (b - a) for a, b in zip(points[:-1], points[1:])]
    )
    risk = 0.0
    for center, radius in OBSTACLES:
        clearance = np.linalg.norm(samples - center, axis=1) - radius
        risk += float(np.mean(np.exp(-np.maximum(clearance, -1.0))))
    risk /= len(OBSTACLES)
    unit = segments / np.maximum(np.linalg.norm(segments, axis=1, keepdims=True), 1e-12)
    smoothness = float(np.mean(1.0 - np.sum(unit[:-1] * unit[1:], axis=1)))
    return np.array([length, risk, smoothness])


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b + 1e-12) and np.any(a < b - 1e-12))


def front_indices(values: np.ndarray) -> np.ndarray:
    keep = []
    for i in range(len(values)):
        if not any(dominates(values[j], values[i]) for j in range(len(values)) if i != j):
            keep.append(i)
    return np.asarray(keep, dtype=int)


def crowding(values: np.ndarray) -> np.ndarray:
    if len(values) <= 2:
        return np.full(len(values), np.inf)
    distance = np.zeros(len(values))
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective])
        distance[order[[0, -1]]] = np.inf
        span = values[order[-1], objective] - values[order[0], objective]
        if span > 1e-12:
            distance[order[1:-1]] += (
                values[order[2:], objective] - values[order[:-2], objective]
            ) / span
    return distance


def update_archive(
    archive_x: np.ndarray, archive_f: np.ndarray, x: np.ndarray, f: np.ndarray, limit: int = 70
) -> tuple[np.ndarray, np.ndarray]:
    merged_x = np.vstack((archive_x, x)) if len(archive_x) else x.copy()
    merged_f = np.vstack((archive_f, f)) if len(archive_f) else f.copy()
    # Existing archive members are offered again every generation.  Remove
    # repeated decision vectors before Pareto filtering so crowding distance
    # measures distinct solutions rather than duplicated entries.
    _, unique_indices = np.unique(
        np.round(merged_x, decimals=12), axis=0, return_index=True
    )
    unique_indices = np.sort(unique_indices)
    merged_x, merged_f = merged_x[unique_indices], merged_f[unique_indices]
    keep = front_indices(merged_f)
    merged_x, merged_f = merged_x[keep], merged_f[keep]
    if len(merged_x) > limit:
        retain = np.argsort(crowding(merged_f))[::-1][:limit]
        merged_x, merged_f = merged_x[retain], merged_f[retain]
    return merged_x, merged_f


def run() -> dict[str, object]:
    waypoints, population_size, iterations = 6, 48, 75
    low = np.tile(np.array([0.0, 0.0, 1.0]), waypoints)
    high = np.tile(np.array([12.0, 12.0, 6.0]), waypoints)
    population = logistic_population(population_size, low.size, low, high)
    values = np.array([objectives(x) for x in population])
    weights = np.array([0.45, 0.40, 0.15])
    initial_score = float(np.min(values @ weights))
    archive_x, archive_f = update_archive(np.empty((0, low.size)), np.empty((0, 3)), population, values)
    mode_counts = {"current": 0, "active": 0, "passive": 0}

    for step in range(iterations):
        time_control = abs((1.0 - (step + 1) / iterations) * (2.0 * RNG.random() - 1.0))
        leader = archive_x[int(RNG.integers(len(archive_x)))]
        mean = population.mean(axis=0)
        next_population = np.empty_like(population)
        for i, current in enumerate(population):
            if RNG.random() < time_control:
                # Following the ocean current: food/leader attraction plus population drift.
                candidate = current + RNG.random(low.size) * (leader - 3.0 * RNG.random() * mean)
                mode_counts["current"] += 1
            elif RNG.random() < 0.5:
                # Passive motion within the local neighborhood.
                candidate = current + 0.1 * RNG.random(low.size) * (high - low)
                mode_counts["passive"] += 1
            else:
                # Active motion toward or away from another jellyfish based on quality.
                j = int(RNG.integers(population_size - 1))
                if j >= i:
                    j += 1
                direction = population[j] - current
                if values[j] @ weights > values[i] @ weights:
                    direction = -direction
                candidate = current + RNG.random() * direction
                mode_counts["active"] += 1

            # PVDE component: rand/1 differential mutation and binomial crossover.
            a, b, c = RNG.choice(population_size, size=3, replace=False)
            mutant = population[a] + 0.55 * (population[b] - population[c])
            mask = RNG.random(low.size) < 0.35
            mask[int(RNG.integers(low.size))] = True
            candidate = np.where(mask, mutant, candidate)
            # Spherical/wrap-around boundary behavior from the MOJS formulation.
            candidate = low + np.mod(candidate - low, high - low)
            candidate_value = objectives(candidate)
            if dominates(candidate_value, values[i]) or candidate_value @ weights < values[i] @ weights:
                next_population[i] = candidate
            else:
                next_population[i] = current
        population = next_population
        values = np.array([objectives(x) for x in population])
        archive_x, archive_f = update_archive(archive_x, archive_f, population, values)

    final_scores = archive_f @ weights
    best_index = int(np.argmin(final_scores))
    final_score = float(final_scores[best_index])
    best_objectives = archive_f[best_index]
    return {
        "paper": "UAV Path Planning for Forest Firefighting Using Optimized Multi-Objective Jellyfish Search Algorithm",
        "algorithm": "serial PVDE-MOJS core reconstruction",
        "seed": SEED,
        "metrics": {
            "population": population_size,
            "iterations": iterations,
            "archive_size": int(len(archive_f)),
            "initial_weighted_cost": round(initial_score, 6),
            "final_weighted_cost": round(final_score, 6),
            "best_objectives": [round(float(v), 6) for v in best_objectives],
            "behavior_counts": mode_counts,
        },
        "checks": {
            "archive_is_nondominated": bool(len(front_indices(archive_f)) == len(archive_f)),
            "archive_positions_unique": bool(
                len(np.unique(np.round(archive_x, decimals=12), axis=0)) == len(archive_x)
            ),
            "improved": bool(final_score < initial_score),
            "finite": bool(np.isfinite(archive_f).all()),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
