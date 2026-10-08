#!/usr/bin/env python3
"""Deterministic NumPy core reconstruction of ECFDBO for two 3-D UAV paths."""

import json

import numpy as np


PAPER = "A Novel Improved Dung Beetle Optimization Algorithm for.pdf"
SEED = 20250202
UAVS = 2
WAYPOINTS = 4
LOWER = np.tile(np.array([-5.0, 0.4]), UAVS * WAYPOINTS)
UPPER = np.tile(np.array([5.0, 4.5]), UAVS * WAYPOINTS)
STARTS = np.array([[0.0, -3.0, 1.0], [0.0, 3.0, 1.0]])
GOALS = np.array([[12.0, 3.0, 1.0], [12.0, -3.0, 1.0]])
OBSTACLES = np.array([[4.8, 0.0, 1.2], [7.2, 0.0, 1.7]])
RADII = np.array([1.35, 1.20])


def decode(vector):
    values = np.asarray(vector, dtype=float).reshape(UAVS, WAYPOINTS, 2)
    x_values = np.linspace(0.0, 12.0, WAYPOINTS + 2)[1:-1]
    paths = []
    for uav in range(UAVS):
        middle = np.column_stack((x_values, values[uav, :, 0], values[uav, :, 1]))
        paths.append(np.vstack((STARTS[uav], middle, GOALS[uav])))
    return np.asarray(paths)


def point_segment_distance(point, start, end):
    """Exact Euclidean distance from a point to a finite line segment."""
    delta = end - start
    denominator = float(np.dot(delta, delta))
    fraction = 0.0 if denominator <= 1.0e-15 else float(
        np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0)
    )
    return float(np.linalg.norm(start + fraction * delta - point))


def synchronized_segment_distance(first_start, first_end, second_start, second_end):
    """Exact minimum separation while both UAVs traverse one indexed leg."""
    relative_start = first_start - second_start
    relative_delta = (first_end - first_start) - (second_end - second_start)
    denominator = float(np.dot(relative_delta, relative_delta))
    fraction = 0.0 if denominator <= 1.0e-15 else float(
        np.clip(-np.dot(relative_start, relative_delta) / denominator, 0.0, 1.0)
    )
    return float(np.linalg.norm(relative_start + fraction * relative_delta))


def path_statistics(vector):
    paths = decode(vector)
    segments = np.diff(paths, axis=1)
    lengths = np.sum(np.linalg.norm(segments, axis=2), axis=1)
    second_difference = paths[:, 2:, :] - 2.0 * paths[:, 1:-1, :] + paths[:, :-2, :]
    smoothness = float(np.sum(np.linalg.norm(second_difference, axis=2)))
    obstacle_clearances = []
    pair_distances = []
    for leg in range(paths.shape[1] - 1):
        for uav in range(UAVS):
            for obstacle, radius in zip(OBSTACLES, RADII):
                obstacle_clearances.append(
                    point_segment_distance(obstacle, paths[uav, leg], paths[uav, leg + 1])
                    - float(radius)
                )
        pair_distances.append(
            synchronized_segment_distance(
                paths[0, leg], paths[0, leg + 1], paths[1, leg], paths[1, leg + 1]
            )
        )
    return paths, lengths, smoothness, float(np.min(obstacle_clearances)), float(np.min(pair_distances))


def objective(vector):
    _, lengths, smoothness, min_clearance, min_separation = path_statistics(vector)
    obstacle_violation = max(0.0, 0.25 - min_clearance)
    separation_violation = max(0.0, 1.0 - min_separation)
    altitude = decode(vector)[:, 1:-1, 2]
    return float(
        np.sum(lengths)
        + 0.45 * np.max(lengths)
        + 0.16 * smoothness
        + 0.035 * np.sum(altitude)
        + 2500.0 * obstacle_violation
        + 900.0 * obstacle_violation * obstacle_violation
        + 2500.0 * separation_violation
        + 900.0 * separation_violation * separation_violation
    )


def evaluate(population):
    return np.asarray([objective(individual) for individual in population], dtype=float)


def smart_reflect(vector, eta=0.20, epsilon=1.0e-4):
    """Equation (12): rebound a violating coordinate using a finite-difference gradient."""
    result = vector.copy()
    violated = np.flatnonzero((result < LOWER) | (result > UPPER))
    for dimension in violated:
        boundary_point = np.clip(result, LOWER, UPPER)
        plus = boundary_point.copy()
        minus = boundary_point.copy()
        plus[dimension] += epsilon
        minus[dimension] -= epsilon
        gradient = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        direction = 1.0 if gradient >= 0.0 else -1.0
        if result[dimension] > UPPER[dimension]:
            result[dimension] = UPPER[dimension] - eta * abs(result[dimension] - UPPER[dimension]) * direction
        else:
            result[dimension] = LOWER[dimension] + eta * abs(LOWER[dimension] - result[dimension]) * direction
    return np.clip(result, LOWER, UPPER), int(len(violated))


def dbo_stage(population, previous, best, worst, progress, rng):
    """Compact realization of rolling, reproduction, foraging and stealing roles."""
    size = len(population)
    updated = population.copy()
    first = max(1, int(0.20 * size))
    second = first + max(1, int(0.20 * size))
    third = second + max(1, int(0.25 * size))
    contraction = 1.0 - progress
    # Multiplicative DBO bounds reverse whenever a best coordinate is
    # negative.  Sort both endpoints before intersecting with global bounds.
    raw_a = best * (1.0 - contraction)
    raw_b = best * (1.0 + contraction)
    local_low = np.maximum(np.minimum(raw_a, raw_b), LOWER)
    local_high = np.minimum(np.maximum(raw_a, raw_b), UPPER)
    for index in range(size):
        current = population[index]
        if index < first:  # ball rolling / dancing
            sign = -1.0 if rng.random() < 0.1 else 1.0
            trial = current + 0.08 * sign * previous[index] + 0.12 * rng.random() * np.abs(current - worst)
        elif index < second:  # brood-ball reproduction near current best
            trial = best + rng.normal(0.0, 0.35, current.shape) * (current - local_low) + rng.random(current.shape) * (current - local_high)
        elif index < third:  # foraging
            trial = current + rng.normal(0.0, 0.25, current.shape) * (current - local_low) + rng.random(current.shape) * (current - local_high)
        else:  # stealing around the global best
            trial = best + 0.18 * rng.normal(size=current.shape) * (np.abs(current - best) + np.abs(current - worst))
        updated[index] = trial
    return updated


def safe_seed():
    fractions = np.linspace(0.0, 1.0, WAYPOINTS + 2)[1:-1]
    values = np.empty((UAVS, WAYPOINTS, 2), dtype=float)
    for uav in range(UAVS):
        direct_y = STARTS[uav, 1] * (1.0 - fractions) + GOALS[uav, 1] * fractions
        side = 2.15 * np.sin(np.pi * fractions) * (1.0 if uav == 0 else -1.0)
        values[uav, :, 0] = direct_y + side
        values[uav, :, 1] = 1.0 + 2.45 * np.sin(np.pi * fractions)
    return values.reshape(-1)


def main():
    rng = np.random.default_rng(SEED)
    population_size = 24
    iterations = 55
    population = rng.uniform(LOWER, UPPER, size=(population_size, LOWER.size))
    population[0] = safe_seed()
    previous = population.copy()
    fitness = evaluate(population)
    initial_best = float(np.min(fitness))
    best_index = int(np.argmin(fitness))
    global_best = population[best_index].copy()
    global_best_cost = float(fitness[best_index])
    reflection_count = 0
    mutation_count = 0

    for iteration in range(iterations):
        progress = (iteration + 1) / iterations
        order = np.argsort(fitness)
        current_best = population[order[0]].copy()
        worst = population[order[-1]].copy()
        repulsors = population[order[1 : max(4, int(0.10 * population_size) + 1)]]
        proposals = np.empty_like(population)
        wg = 0.5 * (1.0 - progress)
        wr = 0.5 * progress
        for index, individual in enumerate(population):
            to_best = global_best - individual
            attraction = wg * to_best / max(np.linalg.norm(to_best), 0.45) ** 3
            repulsion = np.zeros_like(individual)
            for source in repulsors:
                difference = source - individual
                repulsion += difference / max(np.linalg.norm(difference), 0.45) ** 3
            force = attraction - wr * repulsion
            if rng.random() < 0.07:
                guided_step = 1.0 - np.exp(-np.linalg.norm(force))
                trial = individual + force * guided_step
                mutation_count += 1
            else:
                r1, r2 = rng.random(2)
                phi = (1.0 - np.cos(2.0 * np.pi * r1)) / (1.0 + np.log1p(r1))
                psi = np.tanh(2.0 * r2) * np.exp(-1.5 * r2 * r2)
                trial = global_best + phi * (global_best - individual) + psi * (current_best - previous[index])
            proposals[index], reflected = smart_reflect(trial)
            reflection_count += reflected

        proposals = dbo_stage(proposals, previous, current_best, worst, progress, rng)
        for index in range(population_size):
            proposals[index], reflected = smart_reflect(proposals[index])
            reflection_count += reflected
        proposal_fitness = evaluate(proposals)
        improved = proposal_fitness < fitness
        previous = population.copy()
        population[improved] = proposals[improved]
        fitness[improved] = proposal_fitness[improved]
        best_index = int(np.argmin(fitness))
        if fitness[best_index] < global_best_cost:
            global_best_cost = float(fitness[best_index])
            global_best = population[best_index].copy()
        # Explicit elitism keeps the best-so-far solution in the active population.
        worst_index = int(np.argmax(fitness))
        population[worst_index] = global_best
        fitness[worst_index] = global_best_cost

    paths, lengths, smoothness, min_clearance, min_separation = path_statistics(global_best)
    metrics = {
        "initial_best_cost": initial_best,
        "best_cost": global_best_cost,
        "relative_improvement": float((initial_best - global_best_cost) / max(initial_best, 1.0e-12)),
        "total_path_length": float(np.sum(lengths)),
        "maximum_path_length": float(np.max(lengths)),
        "smoothness_cost": smoothness,
        "minimum_obstacle_clearance": min_clearance,
        "minimum_uav_separation": min_separation,
        "force_field_mutations": int(mutation_count),
        "reflected_coordinates": int(reflection_count),
        "iterations": iterations,
    }
    numeric = np.asarray([value for value in metrics.values()], dtype=float)
    result = {
        "paper": PAPER,
        "algorithm": "ECFDBO with chaotic contraction, smart reflection and attraction-repulsion mutation",
        "seed": SEED,
        "metrics": metrics,
        "checks": {
            "finite_metrics": bool(np.all(np.isfinite(numeric))),
            "best_not_worse_than_initial": bool(global_best_cost <= initial_best + 1.0e-10),
            "decision_bounds_respected": bool(np.all(global_best >= LOWER) and np.all(global_best <= UPPER)),
            "obstacle_clearance_satisfied": bool(min_clearance >= 0.25),
            "uav_separation_satisfied": bool(min_separation >= 1.0),
            "endpoints_preserved": bool(np.allclose(paths[:, 0], STARTS) and np.allclose(paths[:, -1], GOALS)),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
