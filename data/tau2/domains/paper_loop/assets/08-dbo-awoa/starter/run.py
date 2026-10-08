#!/usr/bin/env python3
"""Small, deterministic NumPy reconstruction of DBO-AWOA."""

import json
import math

import numpy as np


PAPER = "DBO-AWOA An Adaptive Whale Optimization Algorithm for.pdf"
ALGORITHM = "DBO-AWOA"
SEED = 202508


def objective(x: np.ndarray) -> np.ndarray:
    """Shifted Ackley objective; the optimum is at the hidden shift."""
    shift = np.linspace(-1.25, 1.25, x.shape[-1])
    z = x - shift
    mean_square = np.mean(z * z, axis=-1)
    mean_cos = np.mean(np.cos(2.0 * np.pi * z), axis=-1)
    return -20.0 * np.exp(-0.2 * np.sqrt(mean_square)) - np.exp(mean_cos) + 20.0 + math.e


def icmic_population(
    rng: np.random.Generator, population: int, dimensions: int, lower: float, upper: float
) -> np.ndarray:
    """Equation (8), x[k+1] = sin(0.6/x[k]), mapped into the search box."""
    chaotic = np.empty((population, dimensions), dtype=float)
    state = rng.uniform(0.08, 0.92, size=population)
    for j in range(dimensions):
        state = np.sin(0.6 / np.where(np.abs(state) < 1e-9, 1e-9, state))
        chaotic[:, j] = state
    unit = 0.5 * (chaotic + 1.0)
    return lower + unit * (upper - lower)


def main() -> None:
    rng = np.random.default_rng(SEED)
    n, d, iterations = 30, 8, 70
    lower, upper = -5.0, 5.0
    whales = icmic_population(rng, n, d, lower, upper)
    fitness = objective(whales)
    initial_best = float(np.min(fitness))
    best_index = int(np.argmin(fitness))
    best = whales[best_index].copy()
    best_value = float(fitness[best_index])
    best_trace = [best_value]
    branch_counts = {"dbo_encircle": 0, "random_search": 0, "spiral": 0}

    for t in range(iterations):
        a = 1.0 + math.cos(math.pi * (t + 1) / iterations)  # Equation (9)
        inertia = abs(math.cos(math.pi * t / iterations))  # Equation (10)
        a2 = -1.0 - t / iterations
        updated = np.empty_like(whales)

        for i, position in enumerate(whales):
            coefficient_a = 2.0 * a * rng.random() - a
            coefficient_c = 2.0 * rng.random()
            p = rng.random()
            l_value = rng.random() * a2 + 1.0  # Equations (12)-(13)

            if p < 0.5 and abs(coefficient_a) <= 1.0:
                # DBO reproductive safety-zone fusion, Equations (14)-(18).
                distance = np.abs(coefficient_c * best - position)
                u = best - coefficient_a * distance
                r1, r2 = rng.random(2)
                candidate = best * inertia + (u - lower) * r1 + (u - upper) * r2
                branch_counts["dbo_encircle"] += 1
            elif p < 0.5:
                random_whale = whales[int(rng.integers(n))]
                distance = np.abs(coefficient_c * random_whale - position)
                candidate = random_whale - coefficient_a * distance
                branch_counts["random_search"] += 1
            else:
                distance = np.abs(best - position)
                candidate = (
                    distance * np.exp(l_value) * np.cos(2.0 * np.pi * l_value)
                    + best * inertia
                )
                branch_counts["spiral"] += 1

            updated[i] = np.clip(candidate, lower, upper)

        whales = updated
        fitness = objective(whales)
        current_index = int(np.argmin(fitness))
        current_value = float(fitness[current_index])
        if current_value < best_value:
            best_value = current_value
            best = whales[current_index].copy()
        best_trace.append(best_value)

    values = np.asarray(list(branch_counts.values()), dtype=float)
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
            "best_distance_to_optimum": float(
                np.linalg.norm(best - np.linspace(-1.25, 1.25, d))
            ),
            "dbo_encircle_updates": branch_counts["dbo_encircle"],
            "random_search_updates": branch_counts["random_search"],
            "spiral_updates": branch_counts["spiral"],
        },
        "checks": {
            "finite": bool(np.isfinite(best_value) and np.isfinite(values).all()),
            "best_so_far_nonincreasing": bool(np.all(np.diff(best_trace) <= 1e-12)),
            "all_branches_exercised": bool(np.all(values > 0)),
            "bounds_respected": bool(np.all((whales >= lower) & (whales <= upper))),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
