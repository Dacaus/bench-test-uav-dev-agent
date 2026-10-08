#!/usr/bin/env python3
"""NumPy reconstruction of the AMISO population-update core."""

from __future__ import annotations

import json
import math

import numpy as np


SEED = 20251022
RNG = np.random.default_rng(SEED)


def sphere(population: np.ndarray) -> np.ndarray:
    return np.sum(population * population, axis=1)


def chaotic_initialization(size: int, dimensions: int, low: float, high: float) -> np.ndarray:
    z = RNG.uniform(0.01, 0.99, size=(size, dimensions))
    delta = 0.55
    logistic = 4.0 * z * (1.0 - z)
    sine = np.sin(np.pi * z)
    mixed = np.sin(np.pi * (4.0 * delta * logistic + (1.0 - delta) * sine))
    normalized = 0.5 * (mixed + 1.0)
    return low + normalized * (high - low)


def roulette_strategy(
    population: np.ndarray, index: int, best: np.ndarray, strategy: int
) -> np.ndarray:
    candidates = RNG.choice(len(population), size=3, replace=False)
    r1, r2, r3 = population[candidates]
    f, cr = 0.8, 0.95
    current = population[index]
    if strategy == 0:
        mutant = r1 + f * (r2 - r3)
        return np.where(RNG.random(current.size) < cr, mutant, current)
    if strategy == 1:
        mutant = current + f * (best - current + r1 - r2)
        return np.where(RNG.random(current.size) < cr, mutant, current)
    return current + RNG.random() * (r1 - current) + f * (r2 - r3)


def cascading_evolution(
    population: np.ndarray, fitness: np.ndarray, best: np.ndarray, iteration: int, total: int
) -> np.ndarray:
    order = np.argsort(fitness)
    top_count = max(3, int(0.2 * len(population)))
    top = order[:top_count]
    poor = order[top_count:]
    dimension = population.shape[1]
    factor = math.sin((2.0 * math.pi * iteration + math.pi * dimension) / dimension)
    factor *= (iteration + total) / (2.0 * total)
    candidate = population.copy()

    for idx in top:
        q, k = RNG.choice(top, size=2, replace=False)
        trial = population[q] + factor * (best - RNG.random() * population[k])
        if float(np.dot(trial, trial)) < fitness[idx]:
            candidate[idx] = trial
    RNG.shuffle(poor)
    split = len(poor) // 2
    for idx in poor[:split]:
        candidate[idx] = best + np.sign(RNG.random(dimension) - 0.5) * RNG.uniform(-1.0, 1.0, dimension)
    for idx in poor[split:]:
        candidate[idx] = population[idx] - 2.0 * np.sign(RNG.random(dimension) - 0.5) * RNG.uniform(
            -1.0, 1.0, dimension
        )
    return candidate


def run() -> dict[str, object]:
    size, dimensions, iterations = 60, 24, 120
    low, high = -30.0, 30.0
    population = chaotic_initialization(size, dimensions, low, high)
    fitness = sphere(population)
    initial_best = float(fitness.min())
    best = population[int(np.argmin(fitness))].copy()
    best_value = initial_best
    strategy_success = np.ones(3, dtype=float)
    strategy_calls = np.zeros(3, dtype=int)
    phase_counts = {"exploration": 0, "food": 0, "fight": 0, "mating": 0}

    for iteration in range(1, iterations + 1):
        old_population = population.copy()
        old_fitness = fitness.copy()
        temperature = math.exp(-iteration / iterations)
        food = 0.5 * math.exp((iteration - iterations) / iterations)
        candidate = population.copy()

        if food < 0.25:
            phase_counts["exploration"] += 1
            theta = math.pi * iteration / (10.0 * iterations)
            beta = 1.0 - round(5.0 * iteration / iterations) / 5.0
            adhesion = iteration / iterations
            for i in range(size):
                if RNG.random() < adhesion:
                    candidate[i] = best + RNG.random(dimensions) * math.cos(theta) * beta * RNG.uniform(
                        low, high, dimensions
                    )
                else:
                    peer = population[int(RNG.integers(size))]
                    candidate[i] = peer + RNG.random(dimensions) * (peer - population[i])
        elif temperature > 0.6:
            phase_counts["food"] += 1
            ability = np.exp(-best_value / np.maximum(fitness, 1e-12))[:, None]
            candidate = population + 2.0 * ability * RNG.random(population.shape) * food * (best - population)
        elif RNG.random() < 0.6:
            phase_counts["fight"] += 1
            probabilities = strategy_success / strategy_success.sum()
            for i in range(size):
                strategy = int(RNG.choice(3, p=probabilities))
                strategy_calls[strategy] += 1
                trial = roulette_strategy(population, i, best, strategy)
                if float(np.dot(trial, trial)) < fitness[i]:
                    candidate[i] = trial
                    strategy_success[strategy] += 1.0
        else:
            phase_counts["mating"] += 1
            half = size // 2
            male, female = population[:half], population[half:]
            male_fit, female_fit = fitness[:half], fitness[half:]
            male_best = male[int(np.argmin(male_fit))]
            female_best = female[int(np.argmin(female_fit))]
            candidate[:half] = male + 2.0 * RNG.random(male.shape) * food * (female_best - male)
            candidate[half:] = female + 2.0 * RNG.random(female.shape) * food * (male_best - female)
            candidate = cascading_evolution(candidate, sphere(candidate), best, iteration, iterations)

        candidate = np.clip(candidate, low, high)
        candidate_fitness = sphere(candidate)
        improved = candidate_fitness < old_fitness
        population = np.where(improved[:, None], candidate, old_population)
        fitness = np.where(improved, candidate_fitness, old_fitness)
        worst = int(np.argmax(fitness))
        population[worst] = best  # explicit elitism for the reconstructed finite demo
        fitness[worst] = best_value
        current = int(np.argmin(fitness))
        if fitness[current] < best_value:
            best, best_value = population[current].copy(), float(fitness[current])

    result = {
        "paper": "基于自适应多策略改进蛇优化算法的多无人机路径规划方法",
        "algorithm": "AMISO core reconstruction on the paper's Sphere benchmark",
        "seed": SEED,
        "metrics": {
            "dimensions": dimensions,
            "population": size,
            "iterations": iterations,
            "initial_best": round(initial_best, 10),
            "final_best": round(best_value, 10),
            "initial_to_final_ratio": round(initial_best / max(best_value, 1e-300), 6),
            "phase_counts": phase_counts,
            "roulette_strategy_calls": [int(x) for x in strategy_calls],
        },
        "checks": {
            "improved": bool(best_value < initial_best),
            "within_bounds": bool(np.all((population >= low) & (population <= high))),
            "finite": bool(np.isfinite(fitness).all()),
        },
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
