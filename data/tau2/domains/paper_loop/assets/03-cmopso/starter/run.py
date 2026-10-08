#!/usr/bin/env python3
"""Compact NumPy reconstruction of competitive multi-objective PSO."""

import json

import numpy as np


PAPER = "A competitive mechanism based multi-objective.pdf"
SEED = 20171003


def zdt1(population):
    first = population[:, 0]
    g_value = 1.0 + 9.0 * np.mean(population[:, 1:], axis=1)
    second = g_value * (1.0 - np.sqrt(first / g_value))
    return np.column_stack((first, second))


def non_dominated_sort(values):
    size = len(values)
    dominates = [[] for _ in range(size)]
    dominated_count = np.zeros(size, dtype=int)
    fronts = [[]]
    for left in range(size):
        for right in range(size):
            if left == right:
                continue
            left_dominates = np.all(values[left] <= values[right]) and np.any(values[left] < values[right])
            right_dominates = np.all(values[right] <= values[left]) and np.any(values[right] < values[left])
            if left_dominates:
                dominates[left].append(right)
            elif right_dominates:
                dominated_count[left] += 1
        if dominated_count[left] == 0:
            fronts[0].append(left)
    level = 0
    while level < len(fronts) and fronts[level]:
        following = []
        for left in fronts[level]:
            for right in dominates[left]:
                dominated_count[right] -= 1
                if dominated_count[right] == 0:
                    following.append(right)
        level += 1
        if following:
            fronts.append(following)
    return fronts


def crowding_distance(values, front):
    count = len(front)
    distance = np.zeros(count, dtype=float)
    if count <= 2:
        distance[:] = np.inf
        return distance
    front_values = values[np.asarray(front)]
    for objective in range(values.shape[1]):
        order = np.argsort(front_values[:, objective], kind="stable")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = front_values[order[-1], objective] - front_values[order[0], objective]
        if span > 1.0e-15:
            for rank in range(1, count - 1):
                distance[order[rank]] += (
                    front_values[order[rank + 1], objective]
                    - front_values[order[rank - 1], objective]
                ) / span
    return distance


def ranked_selection(values, count):
    selected = []
    for front in non_dominated_sort(values):
        if len(selected) + len(front) <= count:
            selected.extend(front)
        else:
            crowding = crowding_distance(values, front)
            order = np.argsort(-crowding, kind="stable")
            selected.extend([front[index] for index in order[: count - len(selected)]])
            break
    return np.asarray(selected, dtype=int)


def polynomial_mutation(offspring, rng, distribution_index=20.0):
    result = offspring.copy()
    probability = 1.0 / result.shape[1]
    mask = rng.random(result.shape) < probability
    random_values = rng.random(result.shape)
    delta = np.empty_like(result)
    lower_half = random_values < 0.5
    delta[lower_half] = (2.0 * random_values[lower_half]) ** (1.0 / (distribution_index + 1.0)) - 1.0
    delta[~lower_half] = 1.0 - (2.0 * (1.0 - random_values[~lower_half])) ** (1.0 / (distribution_index + 1.0))
    result[mask] += delta[mask]
    return np.clip(result, 0.0, 1.0), int(np.count_nonzero(mask))


def mean_igd(values):
    reference_x = np.linspace(0.0, 1.0, 401)
    reference = np.column_stack((reference_x, 1.0 - np.sqrt(reference_x)))
    distances = np.linalg.norm(reference[:, None, :] - values[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def main():
    rng = np.random.default_rng(SEED)
    swarm_size = 40
    dimensions = 8
    elite_size = 10
    generations = 65
    positions = rng.random((swarm_size, dimensions))
    velocities = rng.uniform(-0.08, 0.08, size=positions.shape)
    initial_values = zdt1(positions)
    initial_igd = mean_igd(initial_values)
    mutation_total = 0
    competition_total = 0
    elite_selection_rounds = 0

    for _ in range(generations):
        values = zdt1(positions)
        elite_indices = ranked_selection(values, elite_size)
        if (
            elite_indices.size == elite_size
            and np.unique(elite_indices).size == elite_size
            and np.all((elite_indices >= 0) & (elite_indices < swarm_size))
        ):
            elite_selection_rounds += 1
        elites = positions[elite_indices]
        elite_values = values[elite_indices]
        objective_min = np.min(values, axis=0)
        objective_span = np.maximum(np.max(values, axis=0) - objective_min, 1.0e-12)
        normalized_elites = (elite_values - objective_min) / objective_span
        normalized_particles = (values - objective_min) / objective_span
        offspring = np.empty_like(positions)
        offspring_velocity = np.empty_like(velocities)
        for index in range(swarm_size):
            pair = rng.choice(elite_size, size=2, replace=False)
            particle_vector = normalized_particles[index]
            particle_norm = max(float(np.linalg.norm(particle_vector)), 1.0e-12)
            cosine = []
            for candidate in pair:
                elite_vector = normalized_elites[candidate]
                cosine.append(
                    float(np.dot(particle_vector, elite_vector))
                    / (particle_norm * max(float(np.linalg.norm(elite_vector)), 1.0e-12))
                )
            winner = elites[pair[int(np.argmax(cosine))]]
            random_1 = rng.random(dimensions)
            random_2 = rng.random(dimensions)
            new_velocity = random_1 * velocities[index] + random_2 * (winner - positions[index])
            offspring_velocity[index] = np.clip(new_velocity, -0.25, 0.25)
            offspring[index] = positions[index] + offspring_velocity[index]
            competition_total += 1
        offspring = np.clip(offspring, 0.0, 1.0)
        offspring, mutation_count = polynomial_mutation(offspring, rng)
        mutation_total += mutation_count

        combined_positions = np.vstack((positions, offspring))
        combined_velocities = np.vstack((velocities, offspring_velocity))
        combined_values = zdt1(combined_positions)
        survivors = ranked_selection(combined_values, swarm_size)
        positions = combined_positions[survivors]
        velocities = combined_velocities[survivors]

    final_values_all = zdt1(positions)
    first_front_indices = non_dominated_sort(final_values_all)[0]
    front_values = final_values_all[np.asarray(first_front_indices)]
    final_igd = mean_igd(front_values)
    ordered = front_values[np.argsort(front_values[:, 0])]
    adjacent = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    spacing = float(np.std(adjacent)) if len(adjacent) else 0.0
    metrics = {
        "initial_igd": initial_igd,
        "final_igd": final_igd,
        "nondominated_count": int(len(front_values)),
        "objective_1_min": float(np.min(front_values[:, 0])),
        "objective_1_max": float(np.max(front_values[:, 0])),
        "objective_2_min": float(np.min(front_values[:, 1])),
        "objective_2_max": float(np.max(front_values[:, 1])),
        "front_spacing_std": spacing,
        "pairwise_competitions": int(competition_total),
        "mutated_variables": int(mutation_total),
        "elite_selection_rounds": int(elite_selection_rounds),
        "generations": generations,
    }
    numeric = np.asarray(list(metrics.values()), dtype=float)
    result = {
        "paper": PAPER,
        "algorithm": "CMOPSO pairwise elite competition on ZDT1",
        "seed": SEED,
        "metrics": metrics,
        "checks": {
            "finite_metrics": bool(np.all(np.isfinite(numeric))),
            "decision_bounds_respected": bool(np.all((positions >= 0.0) & (positions <= 1.0))),
            "nondominated_front_nonempty": bool(len(front_values) > 0),
            "igd_not_worse_than_initial": bool(final_igd <= initial_igd + 1.0e-12),
            "generational_elite_set_used": bool(elite_selection_rounds == generations),
            "one_competition_per_particle_per_generation": bool(
                competition_total == generations * swarm_size
            ),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
