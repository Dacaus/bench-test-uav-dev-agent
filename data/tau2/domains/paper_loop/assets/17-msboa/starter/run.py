#!/usr/bin/env python3
"""Compact NumPy reconstruction of MSBOA on a synthetic 3-D UAV path."""

from __future__ import annotations

import json

import numpy as np


PAPER = "Multi-strategy secretary bird optimization algorithm for UAV path planning in complex environment.pdf"
ALGORITHM = "MSBOA"
SEED = 20261017

X_STATIONS = np.asarray([2.0, 4.0, 6.0, 8.0])
START = np.asarray([0.0, 0.0, 2.2])
GOAL = np.asarray([10.0, 0.0, 2.2])
OBSTACLE_CENTERS = np.asarray([[3.3, 0.0], [6.7, 0.15]])
OBSTACLE_RADII = np.asarray([0.85, 0.95])
UAV_CLEARANCE = 0.25


def decode(population: np.ndarray) -> np.ndarray:
    """Convert [y1,z1,...,y4,z4] vectors to full 3-D waypoint paths."""
    count = population.shape[0]
    controls = population.reshape(count, X_STATIONS.size, 2)
    middle = np.empty((count, X_STATIONS.size, 3), dtype=float)
    middle[:, :, 0] = X_STATIONS
    middle[:, :, 1:] = controls
    start = np.broadcast_to(START, (count, 1, 3))
    goal = np.broadcast_to(GOAL, (count, 1, 3))
    return np.concatenate((start, middle, goal), axis=1)


def path_components(population: np.ndarray) -> dict[str, np.ndarray]:
    paths = decode(population)
    segments = np.diff(paths, axis=1)
    segment_lengths = np.linalg.norm(segments, axis=2)
    length = np.sum(segment_lengths, axis=1)

    left = segments[:, :-1]
    right = segments[:, 1:]
    cosine = np.sum(left * right, axis=2) / np.maximum(
        np.linalg.norm(left, axis=2) * np.linalg.norm(right, axis=2), 1e-12
    )
    turn_angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    smoothness = 0.5 * np.sum(turn_angles, axis=1)

    altitude = np.sum(np.abs(paths[:, 1:-1, 2] - 2.2), axis=1)

    horizontal_a = paths[:, :-1, :2]
    horizontal_delta = np.diff(paths[:, :, :2], axis=1)
    denominator = np.maximum(np.sum(horizontal_delta**2, axis=2), 1e-12)
    all_distances = []
    for center in OBSTACLE_CENTERS:
        projection = np.sum((center - horizontal_a) * horizontal_delta, axis=2) / denominator
        projection = np.clip(projection, 0.0, 1.0)
        closest = horizontal_a + projection[:, :, None] * horizontal_delta
        all_distances.append(np.linalg.norm(closest - center, axis=2))
    distances = np.stack(all_distances, axis=1)
    required = OBSTACLE_RADII[None, :, None] + UAV_CLEARANCE
    penetration = np.maximum(required - distances, 0.0)
    threat = np.sum(penetration**2, axis=(1, 2))
    collisions = np.sum(distances < required, axis=(1, 2)).astype(int)

    # Paper equation (1): distance, threat, altitude, smoothness; finite hard penalties
    # replace mathematical infinity so every computed/output value remains inspectable.
    total = (
        length
        + 350.0 * threat
        + 1000.0 * collisions.astype(float)
        + 10.0 * altitude
        + smoothness
    )
    return {
        "total": total,
        "length": length,
        "threat": threat,
        "altitude": altitude,
        "smoothness": smoothness,
        "collisions": collisions,
    }


def objective(population: np.ndarray) -> np.ndarray:
    return path_components(population)["total"]


def greedy_update(
    population: np.ndarray,
    fitness: np.ndarray,
    candidate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> int:
    candidate = np.clip(candidate, lower, upper)
    candidate_fitness = objective(candidate)
    accepted = candidate_fitness < fitness
    population[accepted] = candidate[accepted]
    fitness[accepted] = candidate_fitness[accepted]
    return int(np.count_nonzero(accepted))


def normalized(values: np.ndarray, reverse: bool = False) -> np.ndarray:
    minimum = float(np.min(values))
    span = float(np.max(values) - minimum)
    result = (values - minimum) / max(span, 1e-12)
    return 1.0 - result if reverse else result


def main() -> None:
    rng = np.random.default_rng(SEED)
    population_size, iterations = 42, 100
    lower = np.tile(np.asarray([-3.0, 0.6]), X_STATIONS.size)
    upper = np.tile(np.asarray([3.0, 3.8]), X_STATIONS.size)
    dimension = lower.size
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    fitness = objective(population)
    initial_best = float(np.min(fitness))
    initial_collision_free = int(np.count_nonzero(path_components(population)["collisions"] == 0))
    best_history = [initial_best]

    pool_size = 8
    pool = population[np.argsort(fitness)[:pool_size]].copy()
    hunting_accepts = 0
    evasion_accepts = 0
    centroid_reverse_selected = 0
    dfdb_nonbest_selections = 0

    for iteration in range(1, iterations + 1):
        ratio = iteration / iterations
        order = np.argsort(fitness)
        best = population[order[0]].copy()
        worst = population[order[-1]].copy()

        # Equations (18)-(20): pool promising/worst dimensions for diversity.
        promising = np.clip(
            best + rng.uniform(-0.20, 0.20, dimension) * (upper - lower), lower, upper
        )
        binary = rng.integers(0, 2, size=dimension).astype(bool)
        pool_member = np.where(binary, promising, worst)
        pool[(iteration - 1) % pool_size] = pool_member

        # Equations (21)-(24): dynamic fitness-distance balance selects a guide.
        distances = np.linalg.norm(population - best, axis=1)
        period = max(1, round(iterations / 10))
        weight = 0.6 * (1.0 - ((iteration - 1) % period) / period)
        score = weight * normalized(fitness, reverse=True) + (1.0 - weight) * normalized(
            distances
        )
        guide_index = int(np.argmax(score))
        if guide_index != int(order[0]):
            dfdb_nonbest_selections += 1
        guide = population[guide_index].copy()

        # Equation (29): three hunting stages, with pool/Cauchy preferential search late.
        if ratio < 1.0 / 3.0:
            random_indices = rng.integers(0, population_size, size=population_size)
            hunting = population + rng.random((population_size, dimension)) * (
                guide - population[random_indices]
            )
        elif ratio < 2.0 / 3.0:
            normal = rng.normal(size=(population_size, dimension))
            hunting = best + np.exp(ratio**4) * (normal - 0.5) * (best - population)
        else:
            first = pool[rng.integers(0, pool_size, size=population_size)]
            second = pool[rng.integers(0, pool_size, size=population_size)]
            cauchy = 0.5 + 0.1 * np.tan(
                np.pi * (rng.random((population_size, 1)) - 0.5)
            )
            hunting = best + cauchy * (first - second)
        hunting_accepts += greedy_update(population, fitness, hunting, lower, upper)

        # Equation (30): pool-assisted flight/camouflage and random running evasion.
        pool_a = pool[rng.integers(0, pool_size, size=population_size)]
        pool_b = pool[rng.integers(0, pool_size, size=population_size)]
        rb = rng.normal(size=(population_size, dimension))
        r6 = 2.0 * rng.random((population_size, 1))
        flight = best + (2.0 * rb - 1.0) * (1.0 - ratio) ** 2 * (
            (r6 * pool_a + pool_b) * 0.5
        )
        random_peer = population[rng.integers(0, population_size, size=population_size)]
        multiplier = rng.integers(1, 3, size=(population_size, 1))
        running = population + rng.normal(size=(population_size, dimension)) * (
            random_peer - multiplier * population
        )
        choose_flight = rng.random((population_size, 1)) < 0.5
        evasion = np.where(choose_flight, flight, running)
        evasion_accepts += greedy_update(population, fitness, evasion, lower, upper)

        # Equations (25)-(26): centroid opposition followed by greedy N-of-2N survival.
        rank = np.argsort(fitness)
        top_k = int(rng.integers(1, population_size + 1))
        centroid = np.mean(population[rank[:top_k]], axis=0)
        reverse = np.clip(2.0 * centroid - population, lower, upper)
        reverse_fitness = objective(reverse)
        merged = np.vstack((population, reverse))
        merged_fitness = np.concatenate((fitness, reverse_fitness))
        survivors = np.argsort(merged_fitness, kind="stable")[:population_size]
        centroid_reverse_selected += int(np.count_nonzero(survivors >= population_size))
        population = merged[survivors]
        fitness = merged_fitness[survivors]
        best_history.append(float(np.min(fitness)))

    final_index = int(np.argmin(fitness))
    final_vector = population[final_index : final_index + 1]
    details = path_components(final_vector)
    final_best = float(details["total"][0])
    final_length = float(details["length"][0])
    final_threat = float(details["threat"][0])
    final_altitude = float(details["altitude"][0])
    final_smoothness = float(details["smoothness"][0])
    final_collisions = int(details["collisions"][0])
    finite_values = np.asarray(
        [
            initial_best,
            final_best,
            final_length,
            final_threat,
            final_altitude,
            final_smoothness,
            *final_vector.ravel().tolist(),
        ],
        dtype=float,
    )
    payload = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "decision_dimension": dimension,
            "population": population_size,
            "iterations": iterations,
            "initial_best_cost": initial_best,
            "final_cost": final_best,
            "improvement_ratio": float((initial_best - final_best) / max(initial_best, 1e-15)),
            "initial_collision_free_paths": initial_collision_free,
            "final_path_length": final_length,
            "final_threat_penalty": final_threat,
            "final_altitude_cost": final_altitude,
            "final_smoothness_cost": final_smoothness,
            "final_collision_segments": final_collisions,
            "hunting_accepts": hunting_accepts,
            "evasion_accepts": evasion_accepts,
            "dfdb_nonbest_guides": dfdb_nonbest_selections,
            "centroid_reverse_survivors": centroid_reverse_selected,
        },
        "checks": {
            "improved": final_best < initial_best,
            "collision_free": final_collisions == 0,
            "best_history_monotone": bool(np.all(np.diff(best_history) <= 1e-12)),
            "within_bounds": bool(np.all((population >= lower) & (population <= upper))),
            "finite": bool(np.all(np.isfinite(finite_values))),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
