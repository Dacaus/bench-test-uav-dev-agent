#!/usr/bin/env python3
"""Compact MOEA/D-Tchebycheff reconstruction on a synthetic ZDT1 task."""

import json

import numpy as np


PAPER = "MOEAD A Multiobjective Evolutionary Algorithm.pdf"
ALGORITHM = "MOEA/D with Tchebycheff decomposition"
SEED = 202512


def objectives(x: np.ndarray) -> np.ndarray:
    f1 = x[..., 0]
    g = 1.0 + 9.0 * np.mean(x[..., 1:], axis=-1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.stack((f1, f2), axis=-1)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def crowding(values: np.ndarray) -> np.ndarray:
    count, objectives_count = values.shape
    distance = np.zeros(count)
    if count <= 2:
        return np.full(count, np.inf)
    for k in range(objectives_count):
        order = np.argsort(values[:, k])
        distance[order[0]] = distance[order[-1]] = np.inf
        extent = values[order[-1], k] - values[order[0], k]
        if extent > 1e-14:
            distance[order[1:-1]] += (
                values[order[2:], k] - values[order[:-2], k]
            ) / extent
    return distance


def update_archive(
    archive: list[tuple[np.ndarray, np.ndarray]],
    position: np.ndarray,
    value: np.ndarray,
    capacity: int = 100,
) -> list[tuple[np.ndarray, np.ndarray]]:
    for _, old_value in archive:
        if dominates(old_value, value) or np.allclose(old_value, value, atol=1e-12):
            return archive
    archive = [(x, f) for x, f in archive if not dominates(value, f)]
    archive.append((position.copy(), value.copy()))
    while len(archive) > capacity:
        values = np.asarray([f for _, f in archive])
        distances = crowding(values)
        finite = np.where(np.isfinite(distances))[0]
        remove = int(finite[np.argmin(distances[finite])]) if len(finite) else len(archive) // 2
        archive.pop(remove)
    return archive


def tchebycheff(value: np.ndarray, weight: np.ndarray, ideal: np.ndarray) -> float:
    safe_weight = np.where(weight > 0.0, weight, 1e-6)
    return float(np.max(safe_weight * np.abs(value - ideal)))


def igd(values: np.ndarray) -> float:
    parameter = np.linspace(0.0, 1.0, 201)
    reference = np.column_stack((parameter, 1.0 - np.sqrt(parameter)))
    distances = np.sqrt(np.sum((reference[:, None, :] - values[None, :, :]) ** 2, axis=2))
    return float(np.mean(np.min(distances, axis=1)))


def hypervolume_2d(values: np.ndarray, reference: tuple[float, float]) -> float:
    valid = values[(values[:, 0] <= reference[0]) & (values[:, 1] <= reference[1])]
    if len(valid) == 0:
        return 0.0
    valid = valid[np.argsort(valid[:, 0])]
    area, previous_y = 0.0, reference[1]
    for x, y in valid:
        if y < previous_y:
            area += max(0.0, reference[0] - x) * (previous_y - y)
            previous_y = y
    return float(area)


def main() -> None:
    rng = np.random.default_rng(SEED)
    subproblems, dimensions, neighborhood_size, generations = 41, 10, 7, 90
    parameter = np.linspace(0.0, 1.0, subproblems)
    weights = np.column_stack((parameter, 1.0 - parameter))
    weight_distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighborhoods = np.argsort(weight_distances, axis=1)[:, :neighborhood_size]

    population = rng.random((subproblems, dimensions))
    values = objectives(population)
    ideal = np.min(values, axis=0)
    initial_ideal = ideal.copy()
    archive: list[tuple[np.ndarray, np.ndarray]] = []
    for position, value in zip(population, values):
        archive = update_archive(archive, position, value)
    initial_values = np.asarray([f for _, f in archive])
    initial_igd = igd(initial_values)
    initial_hv = hypervolume_2d(initial_values, (1.2, 1.2))
    replacements = 0

    for _ in range(generations):
        for i in range(subproblems):
            pool = neighborhoods[i]
            r1, r2 = rng.choice(pool, size=2, replace=False)
            mutant = population[i] + 0.65 * (population[r1] - population[r2])
            crossover = rng.random(dimensions) < 0.9
            crossover[int(rng.integers(dimensions))] = True
            child = np.where(crossover, mutant, population[i])
            mutation = rng.random(dimensions) < 1.0 / dimensions
            child[mutation] += rng.normal(0.0, 0.06, size=np.count_nonzero(mutation))
            child = np.clip(child, 0.0, 1.0)
            child_value = objectives(child)
            ideal = np.minimum(ideal, child_value)

            for neighbor in pool:
                if tchebycheff(child_value, weights[neighbor], ideal) <= tchebycheff(
                    values[neighbor], weights[neighbor], ideal
                ):
                    population[neighbor] = child
                    values[neighbor] = child_value
                    replacements += 1
            archive = update_archive(archive, child, child_value)

    archive_values = np.asarray([f for _, f in archive])
    nondominated = all(
        not dominates(archive_values[j], archive_values[i])
        for i in range(len(archive_values))
        for j in range(len(archive_values))
        if i != j
    )
    final_igd = igd(archive_values)
    final_hv = hypervolume_2d(archive_values, (1.2, 1.2))
    result = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "decision_dimensions": dimensions,
            "subproblems": subproblems,
            "neighborhood_size": neighborhood_size,
            "generations": generations,
            "archive_size": len(archive),
            "initial_igd": initial_igd,
            "final_igd": final_igd,
            "initial_hypervolume": initial_hv,
            "final_hypervolume": final_hv,
            "neighbor_replacements": replacements,
            "ideal_f1": float(ideal[0]),
            "ideal_f2": float(ideal[1]),
        },
        "checks": {
            "finite": bool(np.isfinite(archive_values).all()),
            "archive_nondominated": bool(nondominated),
            "ideal_componentwise_nonincreasing": bool(np.all(ideal <= initial_ideal + 1e-12)),
            "bounds_respected": bool(np.all((population >= 0.0) & (population <= 1.0))),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
