#!/usr/bin/env python3
"""Two-stage NumPy reconstruction of BACOHBA tasking plus HBAFOA paths."""

import json
import math

import numpy as np


PAPER = "Multi-mechanism swarm optimization for multi-UAV task assignment and.pdf"
ALGORITHM = "BACOHBA + HBAFOA"
SEED = 202513
CLEARANCE_MARGIN = 0.08
CLEARANCE_SAMPLE_SPACING = 0.01


def area_of(point: np.ndarray) -> int:
    return int(point[0] >= 0.5) + 2 * int(point[1] >= 0.5)


def ground_speed(start: np.ndarray, end: np.ndarray) -> float:
    midpoint = 0.5 * (start[:2] + end[:2])
    area = area_of(midpoint)
    wind_speeds = np.array([0.11, 0.07, 0.13, 0.09])
    wind_angles = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    delta = end[:2] - start[:2]
    heading = math.atan2(float(delta[1]), float(delta[0]))
    theta = wind_angles[area] - heading
    wind = wind_speeds[area]
    airspeed = 1.0
    return float(
        max(
            0.2,
            wind * math.cos(theta)
            + math.sqrt(max(0.0, airspeed**2 - (wind * math.sin(theta)) ** 2)),
        )
    )


def edge_cost(start: np.ndarray, end: np.ndarray) -> float:
    distance = float(np.linalg.norm(end[:2] - start[:2]))
    time = distance / ground_speed(start, end)
    changed = float(area_of(start) != area_of(end))
    return 0.45 * distance + 0.40 * time + 0.15 * changed


def split_routes(order: np.ndarray, uavs: int = 2) -> list[np.ndarray]:
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(order, uavs)]


def assignment_cost(order: np.ndarray, tasks: np.ndarray) -> float:
    total = 0.0
    for route in split_routes(order):
        indices = np.concatenate(([0], route, [0]))
        total += sum(edge_cost(tasks[a], tasks[b]) for a, b in zip(indices[:-1], indices[1:]))
    return float(total)


def route_edges(order: np.ndarray) -> list[tuple[int, int]]:
    edges = []
    for route in split_routes(order):
        indices = np.concatenate(([0], route, [0]))
        edges.extend((int(a), int(b)) for a, b in zip(indices[:-1], indices[1:]))
    return edges


def construct_ant(
    rng: np.random.Generator,
    pheromone: np.ndarray,
    costs: np.ndarray,
    reverse: bool,
) -> np.ndarray:
    unvisited = list(range(1, len(costs)))
    order = []
    current = 0
    while unvisited:
        indices = np.asarray(unvisited)
        edge = costs[indices, current] if reverse else costs[current, indices]
        desirability = pheromone[current, indices] ** 1.0 * (1.0 / (edge + 1e-9)) ** 3.0
        probabilities = desirability / np.sum(desirability)
        chosen = int(rng.choice(indices, p=probabilities))
        order.append(chosen)
        unvisited.remove(chosen)
        current = chosen
    if reverse:
        order.reverse()
    return np.asarray(order, dtype=int)


def swap_sequence(source: np.ndarray, target: np.ndarray) -> list[tuple[int, int]]:
    working = source.copy()
    swaps = []
    for i in range(len(working)):
        if working[i] != target[i]:
            j = int(np.where(working == target[i])[0][0])
            swaps.append((i, j))
            working[i], working[j] = working[j], working[i]
    return swaps


def discrete_hba(
    rng: np.random.Generator,
    order: np.ndarray,
    prey: np.ndarray,
    density: float,
) -> np.ndarray:
    candidate = order.copy()
    swaps = swap_sequence(candidate, prey)
    # Discrete counterparts of the digging/honey update: apply a density-
    # controlled subset of swaps toward prey, then a small random exchange.
    probability = min(0.95, 0.2 + 0.35 * density)
    for i, j in swaps:
        if rng.random() < probability:
            candidate[i], candidate[j] = candidate[j], candidate[i]
    if rng.random() < 0.35:
        i, j = rng.choice(len(candidate), size=2, replace=False)
        candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def bacohba(rng: np.random.Generator, tasks: np.ndarray) -> tuple[np.ndarray, float, float]:
    n = len(tasks)
    costs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            costs[i, j] = edge_cost(tasks[i], tasks[j]) if i != j else 1e-9
    pheromone = np.ones((n, n))
    baseline = np.arange(1, n)
    initial_cost = assignment_cost(baseline, tasks)
    best, best_cost = baseline.copy(), initial_cost

    for iteration in range(38):
        candidates = []
        for _ in range(24):
            forward = construct_ant(rng, pheromone, costs, reverse=False)
            backward = construct_ant(rng, pheromone, costs, reverse=True)
            order = min((forward, backward), key=lambda x: assignment_cost(x, tasks))
            density = 2.0 * math.exp(-iteration / 38.0)
            refined = discrete_hba(rng, order, best, density)
            if assignment_cost(refined, tasks) < assignment_cost(order, tasks):
                order = refined
            candidates.append(order)
        candidate_costs = np.array([assignment_cost(c, tasks) for c in candidates])
        index = int(np.argmin(candidate_costs))
        if float(candidate_costs[index]) < best_cost:
            best, best_cost = candidates[index].copy(), float(candidate_costs[index])
        pheromone *= 0.78
        deposit = 1.0 / max(best_cost, 1e-9)
        for a, b in route_edges(best):
            pheromone[a, b] += deposit
            pheromone[b, a] += 0.5 * deposit
    return best, initial_cost, best_cost


def terrain_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.10 + 0.33 * np.exp(-((x - 0.52) ** 2 + (y - 0.48) ** 2) / 0.025)


def make_edge_pairs(order: np.ndarray) -> list[tuple[int, int]]:
    return route_edges(order)


def path_cost(
    vector: np.ndarray,
    edges: list[tuple[int, int]],
    tasks: np.ndarray,
    detailed: bool = False,
):
    mids = vector.reshape(-1, 3)
    total = 0.0
    minimum_clearance = math.inf
    for midpoint, (a, b) in zip(mids, edges):
        start = np.array([tasks[a, 0], tasks[a, 1], 0.30])
        end = np.array([tasks[b, 0], tasks[b, 1], 0.30])
        segments = ((start, midpoint), (midpoint, end))
        for p, q in segments:
            distance = float(np.linalg.norm(q - p))
            time = distance / ground_speed(p, q)
            wind_weight = 1.0 + 0.25 * abs(area_of(p) - area_of(q))
            energy = distance * wind_weight
            height_change = abs(float(q[2] - p[2]))
            total += 0.35 * distance + 0.30 * time + 0.20 * energy + 0.15 * height_change

            # A safe midpoint does not imply safe connecting segments.  Sample
            # every complete segment so a terrain crossing between control
            # points is both penalized and exposed to the final safety check.
            horizontal_distance = float(np.linalg.norm(q[:2] - p[:2]))
            sample_count = max(2, int(math.ceil(horizontal_distance / CLEARANCE_SAMPLE_SPACING)) + 1)
            fraction = np.linspace(0.0, 1.0, sample_count)[:, None]
            samples = p + fraction * (q - p)
            clearances = (
                samples[:, 2]
                - terrain_height(samples[:, 0], samples[:, 1])
                - CLEARANCE_MARGIN
            )
            minimum_clearance = min(minimum_clearance, float(np.min(clearances)))
            violations = np.maximum(-clearances, 0.0)
            if np.any(violations > 0.0):
                total += 1200.0 * float(np.mean(violations**2))
                total += 20.0 * float(np.mean(violations > 0.0))
    result = float(total)
    if detailed:
        return result, float(minimum_clearance)
    return result


def hbafoa(
    rng: np.random.Generator, order: np.ndarray, tasks: np.ndarray
) -> tuple[float, float, int, float, float]:
    edges = make_edge_pairs(order)
    base = []
    for a, b in edges:
        midpoint_xy = 0.5 * (tasks[a, :2] + tasks[b, :2])
        base.extend([midpoint_xy[0], midpoint_xy[1], 0.30])
    base = np.asarray(base)
    dimensions = len(base)
    lower = np.tile([0.0, 0.0, 0.15], len(edges))
    upper = np.tile([1.0, 1.0, 0.80], len(edges))
    step = np.tile([0.08, 0.08, 0.06], len(edges))
    population_size, iterations = 22, 48
    population = np.clip(
        base + rng.normal(0.0, step, size=(population_size, dimensions)), lower, upper
    )
    # Include one auditable, elevated route so feasibility is represented in
    # the initial population and cannot be lost by greedy replacement.
    safe_route = base.copy()
    safe_route[2::3] = 0.55
    population[0] = safe_route
    values = np.array([path_cost(x, edges, tasks) for x in population])
    baseline_cost, baseline_clearance = path_cost(base, edges, tasks, detailed=True)
    best_index = int(np.argmin(values))
    best, best_cost = population[best_index].copy(), float(values[best_index])
    foa_repairs = 0

    for iteration in range(iterations):
        density = 2.0 * math.exp(-iteration / iterations)  # Equation (37).
        candidates = np.empty_like(population)
        for i, position in enumerate(population):
            distance = best - position
            source = (position - population[(i + 1) % population_size]) ** 2
            intensity = rng.random(dimensions) * source / (
                4.0 * np.pi * (distance * distance + 1e-8)
            )
            intensity = np.clip(intensity, 0.0, 2.5)
            flag = 1.0 if rng.random() <= 0.5 else -1.0
            if rng.random() < 0.5:
                r1, r2, r3 = rng.random(3)
                candidate = (
                    best
                    + flag * 6.0 * intensity * best
                    + flag
                    * r1
                    * density
                    * distance
                    * abs(math.cos(2.0 * np.pi * r2) * (1.0 - math.cos(2.0 * np.pi * r3)))
                )
            else:
                candidate = best + flag * rng.random() * density * distance

            invalid = bool(
                np.any(candidate < lower)
                or np.any(candidate > upper)
                or np.linalg.norm(candidate - position) > 1.8
            )
            if invalid:
                # Equation (38), symmetrized to search both directions.
                candidate = position + rng.uniform(-1.0, 1.0, dimensions) * step * rng.random(
                    dimensions
                )
                foa_repairs += 1
            candidates[i] = np.clip(candidate, lower, upper)

        candidate_values = np.array([path_cost(x, edges, tasks) for x in candidates])
        improve = candidate_values < values
        population[improve] = candidates[improve]
        values[improve] = candidate_values[improve]
        index = int(np.argmin(values))
        if float(values[index]) < best_cost:
            best, best_cost = population[index].copy(), float(values[index])
    best_cost, best_clearance = path_cost(best, edges, tasks, detailed=True)
    return baseline_cost, best_cost, foa_repairs, baseline_clearance, best_clearance


def main() -> None:
    rng = np.random.default_rng(SEED)
    tasks = np.array(
        [
            [0.08, 0.08],
            [0.18, 0.32],
            [0.31, 0.82],
            [0.45, 0.20],
            [0.58, 0.73],
            [0.69, 0.29],
            [0.82, 0.88],
            [0.91, 0.46],
            [0.38, 0.58],
        ],
        dtype=float,
    )
    order, assignment_initial, assignment_final = bacohba(rng, tasks)
    path_initial, path_final, repairs, initial_clearance, final_clearance = hbafoa(
        rng, order, tasks
    )
    assigned = np.concatenate(split_routes(order))
    result = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "uavs": 2,
            "task_points_excluding_depot": len(tasks) - 1,
            "assignment_initial_cost": assignment_initial,
            "assignment_final_cost": assignment_final,
            "assignment_improvement": assignment_initial - assignment_final,
            "straight_midpoint_path_cost": path_initial,
            "hbafoa_path_cost": path_final,
            "path_improvement": path_initial - path_final,
            "straight_midpoint_minimum_clearance": initial_clearance,
            "minimum_terrain_clearance": final_clearance,
            "maximum_clearance_sample_spacing": CLEARANCE_SAMPLE_SPACING,
            "foa_replans": repairs,
            "wind_regions": 4,
        },
        "checks": {
            "finite": bool(
                np.isfinite(
                    [
                        assignment_initial,
                        assignment_final,
                        path_initial,
                        path_final,
                        initial_clearance,
                        final_clearance,
                    ]
                ).all()
            ),
            "each_task_assigned_once": bool(
                np.array_equal(np.sort(assigned), np.arange(1, len(tasks)))
            ),
            "assignment_not_worse_than_baseline": bool(
                assignment_final <= assignment_initial + 1e-12
            ),
            "foa_repair_exercised": bool(repairs > 0),
            "terrain_clearance_satisfied": bool(final_clearance >= -1.0e-12),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
