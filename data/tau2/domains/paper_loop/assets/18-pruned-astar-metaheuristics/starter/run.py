#!/usr/bin/env python3
"""Pruned A* seeding plus constrained Differential Evolution in NumPy."""

from __future__ import annotations

import heapq
import json
from typing import Iterable

import numpy as np


PAPER = "Optimized_Path-Planning_in_Continuous_Spaces_for_Unmanned_Aerial_Vehicles_Using_Meta-Heuristics.pdf"
ALGORITHM = "Pruned A* seeded constrained DE"
SEED = 20201018
GRID_SHAPE = (20, 16, 6)
START = (1, 1, 1)
GOAL = (18, 14, 4)
UAV_RADIUS = 0.35

# Two full-height walls create a compact back-and-forth planning scenario.
OBSTACLES = (
    (np.asarray([7.5, -0.5, -0.5]), np.asarray([8.5, 11.5, 5.5])),
    (np.asarray([12.5, 4.5, -0.5]), np.asarray([13.5, 15.5, 5.5])),
)


def make_occupancy() -> np.ndarray:
    coordinates = np.indices(GRID_SHAPE).reshape(3, -1).T.astype(float)
    occupied = np.zeros(coordinates.shape[0], dtype=bool)
    for lower, upper in OBSTACLES:
        inflated_lower = lower - UAV_RADIUS
        inflated_upper = upper + UAV_RADIUS
        occupied |= np.all(
            (coordinates >= inflated_lower) & (coordinates <= inflated_upper), axis=1
        )
    return occupied.reshape(GRID_SHAPE)


def neighbors(node: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    for axis in range(3):
        for direction in (-1, 1):
            candidate = list(node)
            candidate[axis] += direction
            result = tuple(candidate)
            if all(0 <= result[index] < GRID_SHAPE[index] for index in range(3)):
                yield result


def astar(occupied: np.ndarray) -> list[tuple[int, int, int]]:
    frontier: list[tuple[int, int, tuple[int, int, int]]] = []
    heapq.heappush(frontier, (sum(abs(START[i] - GOAL[i]) for i in range(3)), 0, START))
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    cost = {START: 0}
    while frontier:
        _, distance, current = heapq.heappop(frontier)
        if distance != cost[current]:
            continue
        if current == GOAL:
            route = [current]
            while route[-1] != START:
                route.append(parent[route[-1]])
            route.reverse()
            return route
        for adjacent in neighbors(current):
            if occupied[adjacent]:
                continue
            new_distance = distance + 1
            if new_distance < cost.get(adjacent, 10**9):
                cost[adjacent] = new_distance
                parent[adjacent] = current
                heuristic = sum(abs(adjacent[i] - GOAL[i]) for i in range(3))
                heapq.heappush(frontier, (new_distance + heuristic, new_distance, adjacent))
    raise RuntimeError("synthetic scenario unexpectedly has no path")


def prune_collinear(route: list[tuple[int, int, int]]) -> np.ndarray:
    """Remove successive cells that retain the same six-neighbor direction."""
    if len(route) <= 2:
        return np.asarray(route, dtype=float)
    kept = [route[0]]
    previous_direction = np.subtract(route[1], route[0])
    for index in range(1, len(route) - 1):
        next_direction = np.subtract(route[index + 1], route[index])
        if not np.array_equal(next_direction, previous_direction):
            kept.append(route[index])
        previous_direction = next_direction
    kept.append(route[-1])
    return np.asarray(kept, dtype=float)


def segment_intersects_box(
    first: np.ndarray, second: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> bool:
    """Exact segment/AABB slab test, with the box already UAV-inflated."""
    direction = second - first
    enter, leave = 0.0, 1.0
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-12:
            if first[axis] < lower[axis] or first[axis] > upper[axis]:
                return False
            continue
        inverse = 1.0 / direction[axis]
        near = (lower[axis] - first[axis]) * inverse
        far = (upper[axis] - first[axis]) * inverse
        if near > far:
            near, far = far, near
        enter = max(enter, float(near))
        leave = min(leave, float(far))
        if enter > leave:
            return False
    return True


def decode(vector: np.ndarray) -> np.ndarray:
    controls = vector.reshape(-1, 3)
    return np.vstack((np.asarray(START, dtype=float), controls, np.asarray(GOAL, dtype=float)))


def evaluate(vector: np.ndarray) -> tuple[int, float]:
    path = decode(vector)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    collisions = 0
    for first, second in zip(path[:-1], path[1:]):
        for lower, upper in OBSTACLES:
            if segment_intersects_box(
                first, second, lower - UAV_RADIUS, upper + UAV_RADIUS
            ):
                collisions += 1
    return collisions, length


def preferred(
    candidate: tuple[int, float],
    incumbent: tuple[int, float],
    rng: np.random.Generator,
) -> bool:
    """Paper's feasibility tournament for two path candidates."""
    candidate_collisions, candidate_length = candidate
    incumbent_collisions, incumbent_length = incumbent
    if candidate_collisions == 0 and incumbent_collisions != 0:
        return True
    if candidate_collisions != 0 and incumbent_collisions == 0:
        return False
    if candidate_collisions == 0:
        return candidate_length < incumbent_length
    if candidate_collisions != incumbent_collisions:
        return candidate_collisions < incumbent_collisions
    return bool(rng.random() < 0.5)


def best_index(scores: list[tuple[int, float]]) -> int:
    feasible = [index for index, score in enumerate(scores) if score[0] == 0]
    if feasible:
        return min(feasible, key=lambda index: scores[index][1])
    minimum_collisions = min(score[0] for score in scores)
    return next(index for index, score in enumerate(scores) if score[0] == minimum_collisions)


def main() -> None:
    rng = np.random.default_rng(SEED)
    occupied = make_occupancy()
    grid_route = astar(occupied)
    pruned_route = prune_collinear(grid_route)
    seed_vector = pruned_route[1:-1].reshape(-1)
    control_count = pruned_route.shape[0] - 2
    dimension = seed_vector.size
    lower = np.tile(np.zeros(3), control_count)
    upper = np.tile(np.asarray(GRID_SHAPE, dtype=float) - 1.0, control_count)

    population_size, generations = 48, 120
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    population[0] = seed_vector
    # Nearby stochastic candidates preserve the paper's single exact A* seed while
    # making a tiny deterministic demonstration informative in a few milliseconds.
    nearby = min(11, population_size - 1)
    population[1 : nearby + 1] = np.clip(
        seed_vector + rng.normal(0.0, 0.55, size=(nearby, dimension)), lower, upper
    )
    scores = [evaluate(vector) for vector in population]
    seed_collisions, seed_length = evaluate(seed_vector)
    initial_feasible = sum(score[0] == 0 for score in scores)

    mutation_factor, crossover_rate = 0.72, 0.88
    accepted_trials = 0
    feasible_trials = 0
    best_lengths = [scores[best_index(scores)][1]]
    evaluations = population_size

    all_indices = np.arange(population_size)
    for _ in range(generations):
        for index in range(population_size):
            choices = np.delete(all_indices, index)
            first, second, third = rng.choice(choices, size=3, replace=False)
            mutant = population[first] + mutation_factor * (
                population[second] - population[third]
            )
            mutant = np.clip(mutant, lower, upper)
            cross = rng.random(dimension) < crossover_rate
            cross[int(rng.integers(0, dimension))] = True
            trial = np.where(cross, mutant, population[index])
            trial_score = evaluate(trial)
            evaluations += 1
            feasible_trials += int(trial_score[0] == 0)
            if preferred(trial_score, scores[index], rng):
                population[index] = trial
                scores[index] = trial_score
                accepted_trials += 1
        best_lengths.append(scores[best_index(scores)][1])

    winner = best_index(scores)
    final_collisions, final_length = scores[winner]
    final_path = decode(population[winner])
    finite_values = np.asarray(
        [seed_length, final_length, *final_path.ravel().tolist(), *best_lengths], dtype=float
    )
    payload = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "grid_shape": list(GRID_SHAPE),
            "astar_nodes": len(grid_route),
            "pruned_nodes": int(pruned_route.shape[0]),
            "control_points": control_count,
            "population": population_size,
            "generations": generations,
            "evaluations": evaluations,
            "seed_path_length": seed_length,
            "optimized_path_length": final_length,
            "length_improvement_ratio": float(
                (seed_length - final_length) / max(seed_length, 1e-15)
            ),
            "seed_collisions": seed_collisions,
            "initial_feasible_candidates": initial_feasible,
            "feasible_trials": feasible_trials,
            "accepted_trials": accepted_trials,
            "final_collisions": final_collisions,
        },
        "checks": {
            "astar_found": len(grid_route) > 1,
            "pruned_reduces_nodes": int(pruned_route.shape[0]) < len(grid_route),
            "seed_feasible": seed_collisions == 0,
            "final_feasible": final_collisions == 0,
            "seed_retained_or_improved": final_length <= seed_length + 1e-12,
            "within_workspace": bool(np.all((population[winner] >= lower) & (population[winner] <= upper))),
            "finite": bool(np.all(np.isfinite(finite_values))),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
