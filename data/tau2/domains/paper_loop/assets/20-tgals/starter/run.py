#!/usr/bin/env python3
"""NumPy reconstruction of TGALS: Hungarian seed plus repaired genetic search."""

from __future__ import annotations

import json

import numpy as np


SEED = 20251020
RNG = np.random.default_rng(SEED)


def hungarian(cost: np.ndarray) -> np.ndarray:
    """O(n^3) minimum-cost assignment, using 1-indexed dual arrays."""
    n, m = cost.shape
    if n != m:
        raise ValueError("demo expects a square padded matrix")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.empty(n, dtype=int)
    for task_col in range(1, m + 1):
        assignment[task_col - 1] = p[task_col] - 1
    return assignment


def repair(chromosome: np.ndarray, n_uavs: int) -> np.ndarray:
    """Conflict resolution for N=M: retain one task and fill duplicates with idle UAVs."""
    child = chromosome.copy()
    counts = np.bincount(child, minlength=n_uavs)
    idle = list(np.flatnonzero(counts == 0))
    for uav in np.flatnonzero(counts > 1):
        duplicate_positions = np.flatnonzero(child == uav)[1:]
        for position in duplicate_positions:
            child[position] = idle.pop(0)
    return child


def fitness(chromosome: np.ndarray, profit: np.ndarray, route_synergy: np.ndarray) -> float:
    base = float(profit[chromosome, np.arange(len(chromosome))].sum())
    ordering_bonus = 0.0
    # A small adjacent-task term models the paper's path-cost coupling; it makes the
    # second-stage search non-trivial while preserving one-UAV-per-task feasibility.
    for task in range(len(chromosome) - 1):
        ordering_bonus += route_synergy[chromosome[task], chromosome[task + 1]]
    return base + 0.12 * float(ordering_bonus)


def make_problem(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    uav = RNG.uniform(0.0, 20.0, size=(n, 2))
    pickup = RNG.uniform(0.0, 20.0, size=(n, 2))
    delivery = RNG.uniform(0.0, 20.0, size=(n, 2))
    reward = RNG.uniform(80.0, 150.0, size=n)
    distance = np.linalg.norm(uav[:, None, :] - pickup[None, :, :], axis=2)
    distance += np.linalg.norm(pickup - delivery, axis=1)[None, :]
    profit = reward[None, :] - 1.8 * distance
    route_synergy = -np.linalg.norm(uav[:, None, :] - uav[None, :, :], axis=2)
    return profit, route_synergy


def tgals() -> dict[str, object]:
    profit, synergy = make_problem()
    n = len(profit)
    ha = hungarian(-profit)
    ha_fitness = fitness(ha, profit, synergy)

    population_size, generations = 80, 100
    population = np.array([RNG.permutation(n) for _ in range(population_size)])
    population[0] = ha
    initial_random_best = max(fitness(x, profit, synergy) for x in population[1:])
    history = []
    feasible_repairs = 0

    for generation in range(generations):
        scores = np.array([fitness(x, profit, synergy) for x in population])
        elite = population[int(np.argmax(scores))].copy()
        history.append(float(scores.max()))
        selected = []
        for _ in range(population_size):
            candidates = RNG.choice(population_size, size=3, replace=False)
            selected.append(population[candidates[np.argmax(scores[candidates])]].copy())
        offspring = [elite]
        while len(offspring) < population_size:
            a, b = selected[int(RNG.integers(population_size))], selected[int(RNG.integers(population_size))]
            if RNG.random() < 0.8:
                cut = int(RNG.integers(1, n))
                child = np.r_[a[:cut], b[cut:]]
                before = len(np.unique(child))
                child = repair(child, n)
                feasible_repairs += int(before != n)
            else:
                child = a.copy()
            if RNG.random() < 0.2:
                i, j = RNG.choice(n, size=2, replace=False)
                child[i], child[j] = child[j], child[i]
            offspring.append(child)
        population = np.asarray(offspring)

    final_scores = np.array([fitness(x, profit, synergy) for x in population])
    best_idx = int(np.argmax(final_scores))
    best = population[best_idx]
    best_score = float(final_scores[best_idx])
    return {
        "paper": "Two-stage heuristic genetic optimization algorithm for multi-UAV",
        "algorithm": "TGALS core reconstruction",
        "seed": SEED,
        "metrics": {
            "uavs": n,
            "tasks": n,
            "hungarian_seed_revenue": round(ha_fitness, 6),
            "initial_random_best_revenue": round(float(initial_random_best), 6),
            "final_revenue": round(best_score, 6),
            "conflict_repairs": int(feasible_repairs),
            "best_assignment_by_task": [int(x) for x in best],
        },
        "checks": {
            "all_tasks_assigned_once": bool(len(np.unique(best)) == n),
            "elite_seed_preserved_or_improved": bool(best_score >= ha_fitness - 1e-10),
            "finite": bool(np.isfinite(final_scores).all()),
        },
    }


if __name__ == "__main__":
    print(json.dumps(tgals(), ensure_ascii=False, sort_keys=True))
