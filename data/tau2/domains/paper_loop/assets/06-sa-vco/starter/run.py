#!/usr/bin/env python3
"""Small deterministic Sa-VCO reconstruction for min-max multi-UAV inspection."""

import json
import math
from collections import Counter

import numpy as np


PAPER = "An adaptive heuristic algorithm with a collaborative search framework for .pdf"
SEED = 20250206
DEPOT = np.array([5.0, 5.0])
TARGETS = np.array(
    [
        [1.0, 1.4], [2.0, 2.2], [1.2, 4.6], [2.7, 6.1], [1.3, 8.6],
        [4.0, 8.8], [5.5, 7.8], [7.4, 8.9], [8.8, 7.1], [8.3, 4.7],
        [9.0, 2.0], [6.8, 1.2], [5.1, 2.7], [3.8, 4.1], [6.4, 5.2],
    ],
    dtype=float,
)
UAVS = 3


def clone(individual):
    return [route.copy() for route in individual]


def route_length(route):
    if len(route) == 0:
        return 0.0
    points = TARGETS[np.asarray(route, dtype=int)]
    chain = np.vstack((DEPOT, points, DEPOT))
    return float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))


def route_lengths(individual):
    return np.asarray([route_length(route) for route in individual], dtype=float)


def fitness(individual):
    lengths = route_lengths(individual)
    return float(np.max(lengths) + 0.08 * np.sum(lengths))


def edge_set(individual):
    edges = set()
    for route in individual:
        sequence = [-1] + [int(value) for value in route] + [-1]
        edges.update(zip(sequence[:-1], sequence[1:]))
    return edges


def similarity(individual, best):
    best_edges = edge_set(best)
    current_edges = edge_set(individual)
    return len(best_edges & current_edges) / max(len(best_edges), 1)


def random_greedy(indices, rng, candidate_size=3, deterministic=False):
    remaining = [int(value) for value in indices]
    route = []
    current = DEPOT
    while remaining:
        distances = np.asarray([np.linalg.norm(TARGETS[index] - current) for index in remaining])
        nearest = np.argsort(distances)[: min(candidate_size, len(remaining))]
        choice_position = int(nearest[0] if deterministic else rng.choice(nearest))
        chosen = remaining.pop(choice_position)
        route.append(chosen)
        current = TARGETS[chosen]
    return np.asarray(route, dtype=int)


def initial_population(rng, population_size):
    relative = TARGETS - DEPOT
    angle = np.arctan2(relative[:, 1], relative[:, 0])
    ordered = np.argsort(angle)
    base_clusters = [np.asarray(chunk, dtype=int) for chunk in np.array_split(ordered, UAVS)]
    population = []
    for index in range(population_size):
        clusters = [cluster.copy() for cluster in base_clusters]
        if index > 0:
            for _ in range(1 + index % 3):
                left, right = rng.choice(UAVS, size=2, replace=False)
                left_pos = int(rng.integers(len(clusters[left])))
                right_pos = int(rng.integers(len(clusters[right])))
                clusters[left][left_pos], clusters[right][right_pos] = clusters[right][right_pos], clusters[left][left_pos]
        population.append(
            [random_greedy(cluster, rng, candidate_size=3, deterministic=index == 0) for cluster in clusters]
        )
    return population


def two_opt(route):
    best = route.copy()
    best_length = route_length(best)
    improved = True
    passes = 0
    while improved and passes < 3:
        improved = False
        passes += 1
        for left in range(len(best) - 1):
            for right in range(left + 2, len(best) + 1):
                candidate = best.copy()
                candidate[left:right] = candidate[left:right][::-1]
                candidate_length = route_length(candidate)
                if candidate_length + 1.0e-12 < best_length:
                    best = candidate
                    best_length = candidate_length
                    improved = True
    return best


def segment_memories(elite, conventional):
    elite_counter = Counter(edge for individual in elite for edge in edge_set(individual) if edge[0] >= 0 and edge[1] >= 0)
    conventional_counter = Counter(edge for individual in conventional for edge in edge_set(individual) if edge[0] >= 0 and edge[1] >= 0)
    good = {edge for edge, _ in elite_counter.most_common(10)}
    bad = {edge for edge, _ in conventional_counter.most_common(10)}
    overlap = good & bad
    return good - overlap, bad - overlap


def repair(proposed, good_edges=frozenset(), bad_edges=frozenset()):
    used = set()
    routes = []
    for route in proposed:
        clean = []
        for target in route:
            value = int(target)
            if 0 <= value < len(TARGETS) and value not in used:
                clean.append(value)
                used.add(value)
        routes.append(clean)
    missing = [target for target in range(len(TARGETS)) if target not in used]
    for route_index in range(UAVS):
        if not routes[route_index] and missing:
            nearest = min(missing, key=lambda target: np.linalg.norm(TARGETS[target] - DEPOT))
            routes[route_index].append(nearest)
            missing.remove(nearest)
    for target in missing:
        best_choice = None
        best_delta = np.inf
        for route_index, route in enumerate(routes):
            augmented = [-1] + route + [-1]
            for position in range(len(route) + 1):
                previous_target = augmented[position]
                next_target = augmented[position + 1]
                previous_point = DEPOT if previous_target == -1 else TARGETS[previous_target]
                next_point = DEPOT if next_target == -1 else TARGETS[next_target]
                delta = (
                    np.linalg.norm(previous_point - TARGETS[target])
                    + np.linalg.norm(TARGETS[target] - next_point)
                    - np.linalg.norm(previous_point - next_point)
                )
                new_edges = {(previous_target, target), (target, next_target)}
                delta -= 0.18 * len(new_edges & good_edges)
                delta += 0.18 * len(new_edges & bad_edges)
                delta += 0.04 * route_length(np.asarray(route, dtype=int))
                if delta < best_delta:
                    best_delta = float(delta)
                    best_choice = (route_index, position)
        route_index, position = best_choice
        routes[route_index].insert(position, target)
    # Empty routes can only remain if malformed input contained every target elsewhere.
    for route_index, route in enumerate(routes):
        if not route:
            donor = int(np.argmax([len(candidate) for candidate in routes]))
            route.append(routes[donor].pop())
    return [np.asarray(route, dtype=int) for route in routes]


def elite_database_offspring(elite):
    selected_routes = []
    for route_index in range(UAVS):
        source = min(elite, key=lambda individual: route_length(individual[route_index]))
        selected_routes.append(source[route_index].copy())
    return repair(selected_routes)


def conventional_offspring(parent, teacher, good_edges, bad_edges):
    proposed = []
    for route_index in range(UAVS):
        parent_route = parent[route_index]
        teacher_route = teacher[route_index]
        proposed.append((teacher_route if route_length(teacher_route) < route_length(parent_route) else parent_route).copy())
    return repair(proposed, good_edges, bad_edges)


def poor_offspring(parent, partner, good_edges, bad_edges):
    proposed = [
        (parent[route_index] if route_index % 2 == 0 else partner[route_index]).copy()
        for route_index in range(UAVS)
    ]
    return repair(proposed, good_edges, bad_edges)


def all_targets_once(individual):
    flattened = [int(target) for route in individual for target in route]
    return sorted(flattened) == list(range(len(TARGETS))) and all(len(route) > 0 for route in individual)


def main():
    rng = np.random.default_rng(SEED)
    population_size = 30
    iterations = 70
    population = initial_population(rng, population_size)
    costs = np.asarray([fitness(individual) for individual in population])
    initial_best = float(np.min(costs))
    best = clone(population[int(np.argmin(costs))])
    best_cost = float(np.min(costs))
    accepted_offspring = 0
    database_uses = 0

    for iteration in range(iterations):
        progress = (iteration + 1) / iterations
        logistic = 1.0 / (1.0 + math.exp(-6.0 * progress)) - 0.5
        pd = 0.40 + 0.40 * logistic
        pb = 0.40 + 0.30 * logistic
        costs = np.asarray([fitness(individual) for individual in population])
        order = np.argsort(costs)
        if costs[order[0]] < best_cost:
            best = clone(population[int(order[0])])
            best_cost = float(costs[order[0]])
        similarities = np.asarray([similarity(individual, best) for individual in population])
        similarity_order = np.argsort(-similarities)
        quality_rank = np.empty(population_size, dtype=int)
        quality_rank[order] = np.arange(population_size)
        assumed_size = max(4, int(round(pd * population_size)))
        cutoff = max(5, int(round(pb * population_size)))
        elite_indices = [int(index) for index in similarity_order[:assumed_size] if quality_rank[index] < cutoff]
        for index in order[:4]:
            if int(index) not in elite_indices:
                elite_indices.append(int(index))
        elite_indices = sorted(set(elite_indices), key=lambda index: costs[index])
        remaining = [int(index) for index in order if int(index) not in elite_indices]
        conventional_count = max(4, len(remaining) // 2)
        conventional_indices = remaining[:conventional_count]
        poor_indices = remaining[conventional_count:]
        elite = [population[index] for index in elite_indices]
        conventional = [population[index] for index in conventional_indices]
        good_edges, bad_edges = segment_memories(elite, conventional if conventional else elite)
        new_population = [clone(individual) for individual in population]
        change_probability = progress**2

        for index in elite_indices:
            if rng.random() < change_probability:
                candidate = clone(population[index])
                route_index = int(rng.integers(UAVS))
                if len(candidate[route_index]) >= 3:
                    left, right = sorted(rng.choice(len(candidate[route_index]), size=2, replace=False))
                    candidate[route_index][left : right + 1] = candidate[route_index][left : right + 1][::-1]
            else:
                candidate = elite_database_offspring(elite)
                database_uses += 1
            candidate = [two_opt(route) for route in repair(candidate, good_edges, bad_edges)]
            if fitness(candidate) < costs[index]:
                new_population[index] = candidate
                accepted_offspring += 1

        for index in conventional_indices:
            teacher = elite[int(rng.integers(len(elite)))]
            candidate = conventional_offspring(population[index], teacher, good_edges, bad_edges)
            candidate = [two_opt(route) for route in candidate]
            if fitness(candidate) < costs[index]:
                new_population[index] = candidate
                accepted_offspring += 1

        for index in poor_indices:
            partner_index = int(rng.choice(poor_indices)) if poor_indices else index
            candidate = poor_offspring(population[index], population[partner_index], good_edges, bad_edges)
            candidate = [two_opt(route) for route in candidate]
            if fitness(candidate) < costs[index]:
                new_population[index] = candidate
                accepted_offspring += 1
        population = new_population

    final_costs = np.asarray([fitness(individual) for individual in population])
    final_index = int(np.argmin(final_costs))
    if final_costs[final_index] < best_cost:
        best = clone(population[final_index])
        best_cost = float(final_costs[final_index])
    lengths = route_lengths(best)
    metrics = {
        "initial_best_cost": initial_best,
        "best_cost": best_cost,
        "relative_improvement": float((initial_best - best_cost) / max(initial_best, 1.0e-12)),
        "maximum_route_length": float(np.max(lengths)),
        "total_route_length": float(np.sum(lengths)),
        "route_length_std": float(np.std(lengths)),
        "route_lengths": [float(value) for value in lengths],
        "targets": int(len(TARGETS)),
        "uavs": UAVS,
        "accepted_offspring": int(accepted_offspring),
        "elite_database_uses": int(database_uses),
        "iterations": iterations,
    }
    scalar_values = [value for value in metrics.values() if not isinstance(value, list)] + metrics["route_lengths"]
    result = {
        "paper": PAPER,
        "algorithm": "Sa-VCO adaptive initialization and three-subpopulation collaborative search",
        "seed": SEED,
        "metrics": metrics,
        "checks": {
            "finite_metrics": bool(np.all(np.isfinite(np.asarray(scalar_values, dtype=float)))),
            "best_not_worse_than_initial": bool(best_cost <= initial_best + 1.0e-10),
            "every_target_visited_once": bool(all_targets_once(best)),
            "all_uavs_nonempty": bool(all(len(route) > 0 for route in best)),
            "route_count_correct": bool(len(best) == UAVS),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
