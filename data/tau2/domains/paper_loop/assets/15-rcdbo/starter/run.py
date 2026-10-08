#!/usr/bin/env python3
"""Small deterministic NumPy reconstruction of the RCDBO update graph."""

from __future__ import annotations

import json

import numpy as np


PAPER = "Multi-strategy cooperative .pdf"
ALGORITHM = "RCDBO"
SEED = 20241015


def objective(x: np.ndarray) -> np.ndarray:
    """Shifted Rastrigin function; accepts (..., dimension) arrays."""
    shift = np.linspace(-0.35, 0.35, x.shape[-1])
    z = x - shift
    return 10.0 * z.shape[-1] + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z), axis=-1)


def bernoulli_population(
    rng: np.random.Generator, size: int, dimension: int, low: float, high: float
) -> np.ndarray:
    """Equation (8): Bernoulli-map samples scaled to the search interval."""
    control = 0.518
    state = float(rng.uniform(0.05, 0.95))
    values: list[float] = []
    for index in range(size * dimension + 32):
        if state <= 1.0 - control:
            state = state / (1.0 - control)
        else:
            state = (state - 1.0 + control) / control
        state = float(np.clip(state, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
        if index >= 32:
            values.append(state)
    unit = np.asarray(values, dtype=float).reshape(size, dimension)
    return low + (high - low) * unit


def mean_pairwise_distance(population: np.ndarray) -> float:
    delta = population[:, None, :] - population[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    upper = distances[np.triu_indices(population.shape[0], 1)]
    return float(np.mean(upper))


def greedy_replace(
    population: np.ndarray,
    fitness: np.ndarray,
    indices: np.ndarray,
    candidates: np.ndarray,
    low: float,
    high: float,
) -> int:
    candidates = np.clip(candidates, low, high)
    candidate_fitness = objective(candidates)
    accepted = candidate_fitness < fitness[indices]
    if np.any(accepted):
        chosen = indices[accepted]
        population[chosen] = candidates[accepted]
        fitness[chosen] = candidate_fitness[accepted]
    return int(np.count_nonzero(accepted))


def normalized_random_walk_candidates(
    rng: np.random.Generator,
    center: np.ndarray,
    span: float,
    count: int,
) -> np.ndarray:
    """Equations (9)-(14): cumulative binary walk normalized to local bounds."""
    steps = np.where(rng.random((count, center.size)) > 0.5, 1.0, -1.0)
    walk = np.vstack((np.zeros((1, center.size)), np.cumsum(steps, axis=0)))
    minimum = np.min(walk, axis=0)
    maximum = np.max(walk, axis=0)
    scaled = (walk - minimum) / np.maximum(maximum - minimum, 1.0)
    return center - span + 2.0 * span * scaled


def main() -> None:
    rng = np.random.default_rng(SEED)
    population_size = 36
    dimension = 10
    iterations = 100
    low, high = -5.12, 5.12

    population = bernoulli_population(rng, population_size, dimension, low, high)
    previous = population.copy()
    fitness = objective(population)
    initial_best = float(np.min(fitness))
    initial_diversity = mean_pairwise_distance(population)
    best_history = [initial_best]

    random_walk_triggers = 0
    random_walk_accepts = 0
    horizontal_accepts = 0
    vertical_accepts = 0
    behavior_accepts = 0
    stagnant = 0

    group_edges = np.asarray([0, 7, 14, 27, population_size])
    for iteration in range(1, iterations + 1):
        old_population = population.copy()
        best_index = int(np.argmin(fitness))
        worst_index = int(np.argmax(fitness))
        best = population[best_index].copy()
        worst = population[worst_index].copy()
        ratio = iteration / iterations
        radius = 1.0 - ratio

        # DBO's rolling, breeding, foraging, and stealing roles (Equations 1, 4, 6, 7).
        rolling = np.arange(group_edges[0], group_edges[1])
        alpha = rng.choice(np.asarray([-1.0, 1.0]), size=(rolling.size, 1))
        rolling_candidate = (
            population[rolling]
            + alpha * 0.10 * previous[rolling]
            + 0.30 * np.abs(population[rolling] - worst)
        )
        behavior_accepts += greedy_replace(
            population, fitness, rolling, rolling_candidate, low, high
        )

        breeding = np.arange(group_edges[1], group_edges[2])
        breed_a = best * (1.0 - radius)
        breed_b = best * (1.0 + radius)
        lower_breed = np.maximum(np.minimum(breed_a, breed_b), low)
        upper_breed = np.minimum(np.maximum(breed_a, breed_b), high)
        breed_candidate = (
            best
            + rng.random((breeding.size, 1)) * (population[breeding] - lower_breed)
            + rng.random((breeding.size, 1)) * (population[breeding] - upper_breed)
        )
        behavior_accepts += greedy_replace(
            population, fitness, breeding, breed_candidate, low, high
        )

        foraging = np.arange(group_edges[2], group_edges[3])
        forage_a = best * (1.0 - radius)
        forage_b = best * (1.0 + radius)
        lower_forage = np.maximum(np.minimum(forage_a, forage_b), low)
        upper_forage = np.minimum(np.maximum(forage_a, forage_b), high)
        forage_candidate = (
            population[foraging]
            + rng.normal(size=(foraging.size, 1)) * (population[foraging] - lower_forage)
            + rng.random((foraging.size, dimension)) * (population[foraging] - upper_forage)
        )
        behavior_accepts += greedy_replace(
            population, fitness, foraging, forage_candidate, low, high
        )

        stealing = np.arange(group_edges[3], group_edges[4])
        steal_candidate = best + 0.45 * rng.normal(size=(stealing.size, dimension)) * (
            np.abs(population[stealing] - best) + np.abs(population[stealing] - worst)
        )
        behavior_accepts += greedy_replace(
            population, fitness, stealing, steal_candidate, low, high
        )

        current_best = float(np.min(fitness))
        if current_best < best_history[-1] - 1e-14:
            stagnant = 0
        else:
            stagnant += 1

        # The paper activates random walking after five unchanged rounds.
        if stagnant >= 5:
            random_walk_triggers += 1
            beetle = int(rolling[rng.integers(0, rolling.size)])
            stage_scale = 1.0 + (10.0 ** min(6, 1 + int(5 * ratio))) * ratio
            span = (high - low) / stage_scale
            walks = np.clip(
                normalized_random_walk_candidates(rng, population[beetle], span, 24),
                low,
                high,
            )
            walk_fitness = objective(walks)
            walk_best = int(np.argmin(walk_fitness))
            if walk_fitness[walk_best] < fitness[beetle]:
                population[beetle] = walks[walk_best]
                fitness[beetle] = walk_fitness[walk_best]
                random_walk_accepts += 1
            stagnant = 0

        # Equations (15)-(17): late-stage horizontal and vertical crisscross.
        skip_probability = (2.0 / 3.0) * (1.0 - ratio**1.5)
        if rng.random() > skip_probability:
            order = rng.permutation(population_size)
            for first, second in order.reshape(-1, 2):
                parents = population[[first, second]].copy()
                a = rng.random((2, dimension))
                b = rng.uniform(-1.0, 1.0, size=(2, dimension))
                children = np.empty_like(parents)
                children[0] = a[0] * parents[0] + (1.0 - a[0]) * parents[1] + b[0] * (
                    parents[0] - parents[1]
                )
                children[1] = a[1] * parents[1] + (1.0 - a[1]) * parents[0] + b[1] * (
                    parents[1] - parents[0]
                )
                horizontal_accepts += greedy_replace(
                    population,
                    fitness,
                    np.asarray([first, second]),
                    children,
                    low,
                    high,
                )

            best_index = int(np.argmin(fitness))
            d1, d2 = rng.choice(dimension, size=2, replace=False)
            vertical = population[best_index].copy()
            mix = float(rng.random())
            vertical[d1] = mix * vertical[d1] + (1.0 - mix) * vertical[d2]
            vertical_accepts += greedy_replace(
                population,
                fitness,
                np.asarray([best_index]),
                vertical[None, :],
                low,
                high,
            )

        previous = old_population
        best_history.append(float(np.min(fitness)))

    final_index = int(np.argmin(fitness))
    final_best = float(fitness[final_index])
    final_position = population[final_index]
    finite_values = np.asarray(
        [initial_best, final_best, initial_diversity, *final_position.tolist()], dtype=float
    )
    payload = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "dimension": dimension,
            "population": population_size,
            "iterations": iterations,
            "initial_best": initial_best,
            "final_best": final_best,
            "improvement_ratio": float((initial_best - final_best) / max(initial_best, 1e-15)),
            "initial_mean_pairwise_distance": initial_diversity,
            "behavior_accepts": behavior_accepts,
            "random_walk_triggers": random_walk_triggers,
            "random_walk_accepts": random_walk_accepts,
            "horizontal_crossover_accepts": horizontal_accepts,
            "vertical_crossover_accepts": vertical_accepts,
            "best_position_norm": float(np.linalg.norm(final_position)),
        },
        "checks": {
            "improved": final_best < initial_best,
            "best_history_monotone": bool(np.all(np.diff(best_history) <= 1e-12)),
            "within_bounds": bool(np.all((population >= low) & (population <= high))),
            "finite": bool(np.all(np.isfinite(finite_values))),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
