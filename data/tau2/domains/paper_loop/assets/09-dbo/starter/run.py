#!/usr/bin/env python3
"""Deterministic NumPy reconstruction of the four-role DBO optimizer."""

import json

import numpy as np


PAPER = "Dung beetle optimizer a new meta-heuristic algorithm for global optimization.pdf"
ALGORITHM = "Dung Beetle Optimizer (DBO)"
SEED = 202509


def objective(x: np.ndarray) -> np.ndarray:
    shift = np.array([1.0, -0.75, 0.5, -0.25, 0.8, -1.1])
    z = x - shift
    return np.sum(z * z, axis=-1) + 0.15 * np.sum(1.0 - np.cos(5.0 * z), axis=-1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    n, d, iterations = 30, 6, 80
    lower, upper = -4.0, 4.0
    population = rng.uniform(lower, upper, size=(n, d))
    previous = population.copy()
    fitness = objective(population)
    initial_best = float(np.min(fitness))
    best = population[int(np.argmin(fitness))].copy()
    best_value = float(np.min(fitness))
    best_trace = [best_value]
    dance_updates = 0

    # The 6/6/7/11 role split is the N=30 example given in Remark 3.
    rollers = slice(0, 6)
    brood = slice(6, 12)
    foragers = slice(12, 19)
    thieves = slice(19, 30)

    for t in range(iterations):
        fitness = objective(population)
        local_best = population[int(np.argmin(fitness))].copy()
        worst = population[int(np.argmax(fitness))].copy()
        ratio = 1.0 - (t + 1) / iterations
        candidate = population.copy()

        # Ball rolling/dancing, Equations (1)-(2).
        for i in range(rollers.start, rollers.stop):
            if rng.random() < 0.9:
                direction = 1.0 if rng.random() < 0.9 else -1.0
                candidate[i] = (
                    population[i]
                    + direction * 0.1 * previous[i]
                    + 0.3 * np.abs(population[i] - worst)
                )
            else:
                theta = rng.uniform(0.05, np.pi - 0.05)
                if abs(theta - np.pi / 2.0) < 0.08:
                    theta += 0.16
                candidate[i] = population[i] + np.tan(theta) * np.abs(
                    population[i] - previous[i]
                )
                dance_updates += 1

        # Brood-ball spawning area, Equations (3)-(4).
        spawn_a = local_best * (1.0 - ratio)
        spawn_b = local_best * (1.0 + ratio)
        spawn_lower = np.maximum(np.minimum(spawn_a, spawn_b), lower)
        spawn_upper = np.minimum(np.maximum(spawn_a, spawn_b), upper)
        b1 = rng.random((brood.stop - brood.start, d))
        b2 = rng.random((brood.stop - brood.start, d))
        candidate[brood] = (
            local_best
            + b1 * (population[brood] - spawn_lower)
            + b2 * (population[brood] - spawn_upper)
        )
        candidate[brood] = np.clip(candidate[brood], spawn_lower, spawn_upper)

        # Small-beetle foraging area, Equations (5)-(6).
        forage_a = best * (1.0 - ratio)
        forage_b = best * (1.0 + ratio)
        forage_lower = np.maximum(np.minimum(forage_a, forage_b), lower)
        forage_upper = np.minimum(np.maximum(forage_a, forage_b), upper)
        c1 = rng.normal(size=(foragers.stop - foragers.start, 1))
        c2 = rng.random((foragers.stop - foragers.start, d))
        candidate[foragers] = (
            population[foragers]
            + c1 * (population[foragers] - forage_lower)
            + c2 * (population[foragers] - forage_upper)
        )

        # Thieves compete around the best food source, Equation (7).
        g = rng.normal(size=(thieves.stop - thieves.start, d))
        candidate[thieves] = best + 0.5 * g * (
            np.abs(population[thieves] - local_best)
            + np.abs(population[thieves] - best)
        )

        candidate = np.clip(candidate, lower, upper)
        candidate_fitness = objective(candidate)
        # Greedy replacement preserves DBO's stated best-so-far progression.
        improve = candidate_fitness < fitness
        old = population.copy()
        population[improve] = candidate[improve]
        fitness[improve] = candidate_fitness[improve]
        previous = old
        index = int(np.argmin(fitness))
        if float(fitness[index]) < best_value:
            best_value = float(fitness[index])
            best = population[index].copy()
        best_trace.append(best_value)

    result = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "dimensions": d,
            "population": n,
            "iterations": iterations,
            "initial_best": initial_best,
            "final_best": best_value,
            "improvement": initial_best - best_value,
            "distance_to_known_optimum": float(
                np.linalg.norm(best - np.array([1.0, -0.75, 0.5, -0.25, 0.8, -1.1]))
            ),
            "roller_count": 6,
            "brood_count": 6,
            "forager_count": 7,
            "thief_count": 11,
            "dance_updates": dance_updates,
        },
        "checks": {
            "finite": bool(np.isfinite(best_value)),
            "role_counts_sum_to_population": bool(6 + 6 + 7 + 11 == n),
            "best_so_far_nonincreasing": bool(np.all(np.diff(best_trace) <= 1e-12)),
            "bounds_respected": bool(np.all((population >= lower) & (population <= upper))),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
