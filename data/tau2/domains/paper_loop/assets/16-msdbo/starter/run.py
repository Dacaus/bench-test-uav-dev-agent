#!/usr/bin/env python3
"""Deterministic NumPy execution of the principal MSDBO strategies."""

from __future__ import annotations

import json

import numpy as np


PAPER = "Multi-strategy hybrid adaptive .pdf"
ALGORITHM = "MSDBO"
SEED = 20251016


def objective(x: np.ndarray) -> np.ndarray:
    """A shifted 12-D Rastrigin problem used as a compact multimodal test."""
    target = np.linspace(-0.4, 0.4, x.shape[-1])
    z = x - target
    return 10.0 * z.shape[-1] + np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z), axis=-1)


def piecewise_map(values: np.ndarray, parameter: float = 0.35) -> np.ndarray:
    """Paper equation (15), evaluated elementwise on [0, 1)."""
    x = np.asarray(values, dtype=float)
    result = np.empty_like(x)
    first = x < parameter
    second = (x >= parameter) & (x < 0.5)
    third = (x >= 0.5) & (x < 1.0 - parameter)
    fourth = ~(first | second | third)
    result[first] = x[first] / parameter
    result[second] = (x[second] - parameter) / (0.5 - parameter)
    result[third] = (1.0 - parameter - x[third]) / (0.5 - parameter)
    result[fourth] = (1.0 - x[fourth]) / parameter
    return np.clip(result, np.finfo(float).eps, 1.0 - np.finfo(float).eps)


def chaotic_population(
    rng: np.random.Generator, count: int, dimension: int, low: float, high: float
) -> np.ndarray:
    state = rng.uniform(0.01, 0.99, size=(count, dimension))
    for _ in range(12):
        state = piecewise_map(state)
    return low + (high - low) * state


def greedy_update(
    population: np.ndarray,
    fitness: np.ndarray,
    candidate: np.ndarray,
    low: float,
    high: float,
) -> int:
    candidate = np.clip(candidate, low, high)
    candidate_fitness = objective(candidate)
    accepted = candidate_fitness < fitness
    population[accepted] = candidate[accepted]
    fitness[accepted] = candidate_fitness[accepted]
    return int(np.count_nonzero(accepted))


def main() -> None:
    rng = np.random.default_rng(SEED)
    count, dimension, iterations = 40, 12, 110
    low, high = -5.12, 5.12
    population = chaotic_population(rng, count, dimension, low, high)
    fitness = objective(population)
    initial_best = float(np.min(fitness))
    global_best = population[int(np.argmin(fitness))].copy()
    global_best_value = initial_best
    best_history = [initial_best]

    osprey_accepts = 0
    t_mutation_accepts = 0
    weighted_thief_accepts = 0
    annealing_accepts = 0
    annealing_uphill_accepts = 0

    for iteration in range(1, iterations + 1):
        ratio = iteration / iterations
        best_index = int(np.argmin(fitness))
        best = population[best_index].copy()
        worst = population[int(np.argmax(fitness))].copy()

        # Equation (16): Osprey-style stochastic global exploration.
        integer_vector = rng.integers(1, 3, size=(count, dimension))
        osprey = population + rng.random((count, dimension)) * (
            best - integer_vector * population
        )
        osprey_accepts += greedy_update(population, fitness, osprey, low, high)

        # Equations (17)-(18): iteration-adaptive t mutation and sigmoid contraction.
        sigmoid = 1.0 / (1.0 + np.exp(10.0 * (ratio - 0.5)))
        degrees_of_freedom = float(np.exp(4.0 * ratio * ratio))
        current_best = population[int(np.argmin(fitness))].copy()
        t_noise = rng.standard_t(degrees_of_freedom, size=(count, dimension))
        t_candidate = current_best + sigmoid * t_noise * (np.abs(current_best) + 0.05)
        t_mutation_accepts += greedy_update(population, fitness, t_candidate, low, high)

        # Equations (19)-(20): dynamic weights shift emphasis as iterations advance.
        k1 = 1.0 - ratio**3
        k2 = ratio**3
        current_best = population[int(np.argmin(fitness))].copy()
        weighted_step = k1 * np.abs(population - current_best) + k2 * np.abs(
            population - worst
        )
        thief = current_best + rng.normal(0.0, 0.35, size=(count, dimension)) * weighted_step
        weighted_thief_accepts += greedy_update(population, fitness, thief, low, high)

        # Equations (21)-(22): late Gaussian local search filtered by simulated annealing.
        if ratio >= 0.55:
            temperature = max(1e-4, 0.30 * (1.0 - ratio) + 0.002)
            scale = 0.20 * sigmoid + 0.005
            local = np.clip(
                population + scale * rng.normal(size=(count, dimension)), low, high
            )
            local_fitness = objective(local)
            delta = local_fitness - fitness
            probability = np.exp(-np.maximum(delta, 0.0) / temperature)
            accepted = (delta < 0.0) | (rng.random(count) < probability)
            uphill = accepted & (delta > 0.0)
            annealing_accepts += int(np.count_nonzero(accepted))
            annealing_uphill_accepts += int(np.count_nonzero(uphill))
            population[accepted] = local[accepted]
            fitness[accepted] = local_fitness[accepted]

        iteration_best_index = int(np.argmin(fitness))
        iteration_best_value = float(fitness[iteration_best_index])
        if iteration_best_value < global_best_value:
            global_best_value = iteration_best_value
            global_best = population[iteration_best_index].copy()
        else:
            # Elitism keeps the best-so-far solution in the active population.
            replace = int(np.argmax(fitness))
            population[replace] = global_best
            fitness[replace] = global_best_value
        best_history.append(global_best_value)

    finite_values = np.asarray(
        [initial_best, global_best_value, float(np.linalg.norm(global_best)), *best_history],
        dtype=float,
    )
    payload = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "dimension": dimension,
            "population": count,
            "iterations": iterations,
            "initial_best": initial_best,
            "final_best": global_best_value,
            "improvement_ratio": float(
                (initial_best - global_best_value) / max(initial_best, 1e-15)
            ),
            "final_position_norm": float(np.linalg.norm(global_best)),
            "osprey_accepts": osprey_accepts,
            "adaptive_t_accepts": t_mutation_accepts,
            "dynamic_weight_accepts": weighted_thief_accepts,
            "annealing_accepts": annealing_accepts,
            "annealing_uphill_accepts": annealing_uphill_accepts,
        },
        "checks": {
            "improved": global_best_value < initial_best,
            "best_history_monotone": bool(np.all(np.diff(best_history) <= 1e-12)),
            "within_bounds": bool(np.all((population >= low) & (population <= high))),
            "finite": bool(np.all(np.isfinite(finite_values))),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
