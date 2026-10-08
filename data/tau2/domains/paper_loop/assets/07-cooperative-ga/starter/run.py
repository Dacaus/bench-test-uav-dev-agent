#!/usr/bin/env python3
"""Modified-GA core for cooperative task allocation with resource and timing constraints."""

import json
import math

import numpy as np


PAPER = "Cooperative task allocation with simultaneous arrival and resource constraint for multi-UAV using a genetic algorithm.pdf"
SEED = 20240707
UAV_COUNT = 5
TARGET_COUNT = 3
SLOTS = 3
HEADING_LEVELS = 8
UAV_STARTS = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0], [4.0, -2.0]])
UAV_HEADINGS = np.array([0.2, 2.7, -0.5, -2.5, 1.4])
UAV_RESOURCES = np.array([[4, 2], [2, 4], [4, 4], [3, 3], [3, 3]], dtype=int)
TARGETS = np.array([[3.0, 3.8], [6.7, 4.5], [4.3, 8.5]])
# Every target exceeds every single UAV in at least one resource dimension,
# so the simultaneous-arrival coalition mechanism is exercised rather than vacuous.
TARGET_REQUIREMENTS = np.array([[5, 3], [3, 5], [5, 5]], dtype=int)
TARGET_VALUES = np.array([100.0, 92.0, 88.0])
SPEED = 1.0


def target_order():
    distance_sum = np.sum(np.linalg.norm(UAV_STARTS[:, None, :] - TARGETS[None, :, :], axis=2), axis=0)
    utility = TARGET_VALUES - 0.35 * distance_sum
    return np.argsort(-utility), utility


ORDER, UTILITIES = target_order()


def resource_plan(assignments):
    remaining = UAV_RESOURCES.copy()
    consumed = np.zeros_like(remaining)
    for target in ORDER:
        coalition = [int(value) for value in assignments[target] if value >= 0]
        if not coalition or len(set(coalition)) != len(coalition):
            return False, remaining, consumed
        for resource_type in range(UAV_RESOURCES.shape[1]):
            need = int(TARGET_REQUIREMENTS[target, resource_type])
            ranked = sorted(coalition, key=lambda uav: int(np.sum(remaining[uav])), reverse=True)
            for uav in ranked:
                amount = min(need, int(remaining[uav, resource_type]))
                remaining[uav, resource_type] -= amount
                consumed[uav, resource_type] += amount
                need -= amount
                if need == 0:
                    break
            if need > 0:
                return False, remaining, consumed
    return True, remaining, consumed


def baseline_assignments():
    return np.array([[0, 3, -1], [1, 4, -1], [2, 3, -1]], dtype=int)


def repair(assignments, headings, rng):
    assignments = np.asarray(assignments, dtype=int).copy()
    headings = np.asarray(headings, dtype=int).copy() % HEADING_LEVELS
    for target in range(TARGET_COUNT):
        seen = set()
        for slot in range(SLOTS):
            uav = int(assignments[target, slot])
            if uav < 0 or uav >= UAV_COUNT or uav in seen:
                assignments[target, slot] = -1
            else:
                seen.add(uav)
        if not seen:
            assignments[target, 0] = int(rng.integers(UAV_COUNT))
    feasible, _, _ = resource_plan(assignments)
    if feasible:
        return assignments, headings

    # Attempt resource-aware additions before using the known feasible scaffold.
    for target in ORDER:
        coalition = {int(value) for value in assignments[target] if value >= 0}
        available_slots = [slot for slot in range(SLOTS) if assignments[target, slot] < 0]
        current_capacity = np.sum(UAV_RESOURCES[list(coalition)], axis=0) if coalition else np.zeros(2, dtype=int)
        while np.any(current_capacity < TARGET_REQUIREMENTS[target]) and available_slots:
            deficit = np.maximum(TARGET_REQUIREMENTS[target] - current_capacity, 0)
            candidates = [uav for uav in range(UAV_COUNT) if uav not in coalition]
            chosen = max(candidates, key=lambda uav: int(np.sum(np.minimum(UAV_RESOURCES[uav], deficit))))
            slot = available_slots.pop(0)
            assignments[target, slot] = chosen
            coalition.add(chosen)
            current_capacity += UAV_RESOURCES[chosen]
    feasible, _, _ = resource_plan(assignments)
    if feasible:
        return assignments, headings
    scaffold = baseline_assignments()
    return scaffold, headings


def random_chromosome(rng):
    assignments = np.full((TARGET_COUNT, SLOTS), -1, dtype=int)
    headings = rng.integers(0, HEADING_LEVELS, size=(TARGET_COUNT, SLOTS))
    for target in range(TARGET_COUNT):
        coalition_size = int(rng.integers(1, SLOTS + 1))
        assignments[target, :coalition_size] = rng.choice(UAV_COUNT, size=coalition_size, replace=False)
    return repair(assignments, headings, rng)


def curve_length(start, end, start_heading, end_heading):
    distance = float(np.linalg.norm(end - start))
    scale = max(distance / 3.0, 0.10)
    control_0 = start
    control_1 = start + scale * np.array([math.cos(start_heading), math.sin(start_heading)])
    control_2 = end - scale * np.array([math.cos(end_heading), math.sin(end_heading)])
    control_3 = end
    t_value = np.linspace(0.0, 1.0, 25)[:, None]
    curve = (
        (1.0 - t_value) ** 3 * control_0
        + 3.0 * (1.0 - t_value) ** 2 * t_value * control_1
        + 3.0 * (1.0 - t_value) * t_value**2 * control_2
        + t_value**3 * control_3
    )
    return float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))


def evaluate(chromosome, detailed=False):
    assignments, heading_codes = chromosome
    feasible, remaining, _ = resource_plan(assignments)
    if not feasible:
        return (1.0e9, None) if detailed else 1.0e9
    positions = UAV_STARTS.copy()
    headings = UAV_HEADINGS.copy()
    times = np.zeros(UAV_COUNT)
    distances = np.zeros(UAV_COUNT)
    total_wait = 0.0
    simultaneous_error = 0.0
    simultaneous_relative_error = 0.0
    synchronization_scale = 1.0
    minimum_wait = math.inf
    maximum_raw_arrival_spread = 0.0
    for target in ORDER:
        coalition = [int(value) for value in assignments[target] if value >= 0]
        raw_arrivals = []
        edge_lengths = []
        final_headings = []
        for uav in coalition:
            slot = int(np.flatnonzero(assignments[target] == uav)[0])
            final_heading = 2.0 * math.pi * int(heading_codes[target, slot]) / HEADING_LEVELS
            length = curve_length(positions[uav], TARGETS[target], headings[uav], final_heading)
            raw_arrivals.append(times[uav] + length / SPEED)
            edge_lengths.append(length)
            final_headings.append(final_heading)
        raw_arrivals = np.asarray(raw_arrivals, dtype=float)
        synchronized_time = float(np.max(raw_arrivals))
        waits = synchronized_time - raw_arrivals
        post_wait_arrivals = raw_arrivals + waits
        raw_spread = float(np.ptp(raw_arrivals)) if len(coalition) > 1 else 0.0
        maximum_raw_arrival_spread = max(maximum_raw_arrival_spread, raw_spread)
        minimum_wait = min(minimum_wait, float(np.min(waits)))
        for index, uav in enumerate(coalition):
            total_wait += waits[index]
            # Advance each UAV by its own flight time and applied wait.  This is
            # deliberately not assigned from a constant synchronized-time array:
            # a faulty wait calculation would leave unequal post-wait arrivals.
            times[uav] = post_wait_arrivals[index]
            distances[uav] += edge_lengths[index]
            positions[uav] = TARGETS[target]
            headings[uav] = final_headings[index]
        actual_arrivals = times[np.asarray(coalition, dtype=int)]
        scale = max(1.0, float(np.max(np.abs(actual_arrivals))))
        local_error = float(np.ptp(actual_arrivals)) if len(coalition) > 1 else 0.0
        simultaneous_error = max(simultaneous_error, local_error)
        simultaneous_relative_error = max(simultaneous_relative_error, local_error / scale)
        synchronization_scale = max(synchronization_scale, scale)
    objective = float(np.sum(distances) + 0.65 * np.max(distances) + 0.12 * np.max(times) + 0.03 * total_wait)
    if not detailed:
        return objective
    return objective, {
        "distances": distances,
        "times": times,
        "wait": total_wait,
        "remaining": remaining,
        "simultaneous_error": simultaneous_error,
        "simultaneous_relative_error": simultaneous_relative_error,
        "synchronization_scale": synchronization_scale,
        "minimum_wait": minimum_wait,
        "maximum_raw_arrival_spread": maximum_raw_arrival_spread,
    }


def crossover(parent_a, parent_b, rng):
    assign_a, heading_a = parent_a
    assign_b, heading_b = parent_b
    cut = int(rng.integers(1, TARGET_COUNT * SLOTS))
    flat_assign = np.concatenate((assign_a.reshape(-1)[:cut], assign_b.reshape(-1)[cut:]))
    flat_heading = np.concatenate((heading_a.reshape(-1)[:cut], heading_b.reshape(-1)[cut:]))
    return repair(flat_assign.reshape(TARGET_COUNT, SLOTS), flat_heading.reshape(TARGET_COUNT, SLOTS), rng)


def mutate(chromosome, rng):
    assignments, headings = chromosome
    assignments = assignments.copy()
    headings = headings.copy()
    if rng.random() < 0.5:
        target = int(rng.integers(TARGET_COUNT))
        slot = int(rng.integers(SLOTS))
        used = {int(value) for value in assignments[target] if value >= 0 and int(value) != int(assignments[target, slot])}
        choices = [-1] + [uav for uav in range(UAV_COUNT) if uav not in used]
        assignments[target, slot] = int(rng.choice(choices))
    else:
        target = int(rng.integers(TARGET_COUNT))
        slot = int(rng.integers(SLOTS))
        old = int(headings[target, slot])
        choices = [value for value in range(HEADING_LEVELS) if value != old]
        headings[target, slot] = int(rng.choice(choices))
    return repair(assignments, headings, rng)


def no_duplicates(assignments):
    return all(
        len([value for value in row if value >= 0]) == len(set(int(value) for value in row if value >= 0))
        for row in assignments
    )


def main():
    rng = np.random.default_rng(SEED)
    population_size = 42
    elite_count = 4
    generations = 65
    population = [random_chromosome(rng) for _ in range(population_size)]
    population[0] = (baseline_assignments(), np.zeros((TARGET_COUNT, SLOTS), dtype=int))
    costs = np.asarray([evaluate(chromosome) for chromosome in population])
    initial_best = float(np.min(costs))
    crossover_count = 0
    mutation_count = 0

    for _ in range(generations):
        costs = np.asarray([evaluate(chromosome) for chromosome in population])
        order = np.argsort(costs)
        next_population = [
            (population[int(index)][0].copy(), population[int(index)][1].copy())
            for index in order[:elite_count]
        ]
        inverse = 1.0 / np.maximum(costs, 1.0e-12)
        probabilities = inverse / np.sum(inverse)
        while len(next_population) < population_size:
            parent_indices = rng.choice(population_size, size=2, replace=False, p=probabilities)
            child_a = crossover(population[int(parent_indices[0])], population[int(parent_indices[1])], rng)
            child_b = crossover(population[int(parent_indices[1])], population[int(parent_indices[0])], rng)
            crossover_count += 2
            for child in (child_a, child_b):
                if rng.random() < 0.35:
                    child = mutate(child, rng)
                    mutation_count += 1
                next_population.append(child)
                if len(next_population) == population_size:
                    break
        population = next_population

    costs = np.asarray([evaluate(chromosome) for chromosome in population])
    best = population[int(np.argmin(costs))]
    best_cost, detail = evaluate(best, detailed=True)
    assignments, _ = best
    feasible, remaining, _ = resource_plan(assignments)
    coalition_sizes = [int(np.count_nonzero(assignments[target] >= 0)) for target in range(TARGET_COUNT)]
    metrics = {
        "initial_best_cost": initial_best,
        "best_cost": float(best_cost),
        "relative_improvement": float((initial_best - best_cost) / max(initial_best, 1.0e-12)),
        "total_flight_distance": float(np.sum(detail["distances"])),
        "maximum_uav_distance": float(np.max(detail["distances"])),
        "mission_completion_time": float(np.max(detail["times"])),
        "total_synchronization_wait": float(detail["wait"]),
        "simultaneous_arrival_error": float(detail["simultaneous_error"]),
        "simultaneous_arrival_relative_error": float(detail["simultaneous_relative_error"]),
        "synchronization_time_scale": float(detail["synchronization_scale"]),
        "maximum_pre_wait_arrival_spread": float(detail["maximum_raw_arrival_spread"]),
        "coalition_sizes": coalition_sizes,
        "resource_slack": [int(value) for value in np.sum(remaining, axis=0)],
        "target_execution_order": [int(value) for value in ORDER],
        "crossovers": int(crossover_count),
        "mutations": int(mutation_count),
        "generations": generations,
    }
    scalar_values = [value for value in metrics.values() if not isinstance(value, list)]
    scalar_values += metrics["coalition_sizes"] + metrics["resource_slack"] + metrics["target_execution_order"]
    result = {
        "paper": PAPER,
        "algorithm": "multi-type-gene GA with utility-order unlocking and resource repair",
        "seed": SEED,
        "metrics": metrics,
        "checks": {
            "finite_metrics": bool(np.all(np.isfinite(np.asarray(scalar_values, dtype=float)))),
            "best_not_worse_than_initial": bool(best_cost <= initial_best + 1.0e-10),
            "resource_requirements_satisfied": bool(feasible),
            "simultaneous_arrival_satisfied": bool(
                detail["simultaneous_relative_error"] <= 1.0e-12
                and detail["minimum_wait"] >= -1.0e-12
            ),
            "synchronization_wait_exercised": bool(
                detail["maximum_raw_arrival_spread"] > 1.0e-9
                and detail["wait"] > 1.0e-9
            ),
            "no_duplicate_uav_within_target": bool(no_duplicates(assignments)),
            "deadlock_free_global_target_order": bool(len(set(int(value) for value in ORDER)) == TARGET_COUNT),
            "coalition_size_limit_satisfied": bool(all(1 <= size <= SLOTS for size in coalition_sizes)),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
