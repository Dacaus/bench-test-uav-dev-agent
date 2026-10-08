#!/usr/bin/env python3
"""NumPy reconstruction of HIPSO-MSOS on a small timestamped two-UAV task."""

import json
import math

import numpy as np


PAPER = "A novel hybrid particle swarm optimization for multi-UAV cooperate path planning.pdf"
SEED = 20210305
UAVS = 2
SEGMENTS = 6
VARIABLES_PER_UAV = (SEGMENTS - 1) * 2
LOWER = np.tile(np.array([-1.20, -0.35]), UAVS * (SEGMENTS - 1))
UPPER = np.tile(np.array([1.20, 0.35]), UAVS * (SEGMENTS - 1))
STARTS = np.array([[-3.0, 0.0, 1.0], [3.0, 0.0, 1.0]])
GOALS = np.array([[3.0, 10.0, 1.0], [-3.0, 10.0, 1.0]])
OBSTACLE_CENTER = np.array([0.0, 5.0, 1.2])
OBSTACLE_RADIUS = 1.20


def decode(vector):
    controls = np.asarray(vector, dtype=float).reshape(UAVS, SEGMENTS - 1, 2)
    fractions = np.arange(1, SEGMENTS, dtype=float) / SEGMENTS
    paths = []
    for uav in range(UAVS):
        direction = GOALS[uav, :2] - STARTS[uav, :2]
        axis = direction / np.linalg.norm(direction)
        perpendicular = np.array([-axis[1], axis[0]])
        base_xy = STARTS[uav, :2] + fractions[:, None] * direction
        lateral = np.cumsum(controls[uav, :, 0])
        lateral -= fractions * lateral[-1]
        middle_xy = base_xy + lateral[:, None] * perpendicular
        altitude = STARTS[uav, 2] + np.cumsum(controls[uav, :, 1])
        altitude -= fractions * (altitude[-1] - GOALS[uav, 2])
        middle = np.column_stack((middle_xy, altitude))
        paths.append(np.vstack((STARTS[uav], middle, GOALS[uav])))
    return np.asarray(paths)


def dense_paths(paths, samples_per_segment=5):
    pieces = []
    for index in range(paths.shape[1] - 1):
        fraction = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        pieces.append(paths[:, index, None, :] * (1.0 - fraction[None, :, None]) + paths[:, index + 1, None, :] * fraction[None, :, None])
    pieces.append(paths[:, -1:, :])
    return np.concatenate(pieces, axis=1)


def statistics(vector):
    paths = decode(vector)
    segments = np.diff(paths, axis=1)
    lengths = np.sum(np.linalg.norm(segments, axis=2), axis=1)
    unit = segments / np.maximum(np.linalg.norm(segments, axis=2, keepdims=True), 1.0e-12)
    cosine = np.sum(unit[:, :-1, :] * unit[:, 1:, :], axis=2)
    turning = float(np.sum(np.arccos(np.clip(cosine, -1.0, 1.0))))
    dense = dense_paths(paths)
    obstacle_clearance = float(np.min(np.linalg.norm(dense - OBSTACLE_CENTER, axis=2) - OBSTACLE_RADIUS))
    separation = float(np.min(np.linalg.norm(dense[0] - dense[1], axis=1)))
    altitude_min = float(np.min(dense[:, :, 2]))
    altitude_max = float(np.max(dense[:, :, 2]))
    altitude_cost = float(np.mean(dense[:, :, 2]) + np.std(dense[:, :, 2]))
    return paths, lengths, turning, obstacle_clearance, separation, altitude_min, altitude_max, altitude_cost


def objective(vector):
    _, lengths, turning, clearance, separation, altitude_min, altitude_max, altitude_cost = statistics(vector)
    obstacle_violation = max(0.0, 0.20 - clearance)
    separation_violation = max(0.0, 1.0 - separation)
    altitude_violation = max(0.0, 0.40 - altitude_min) + max(0.0, altitude_max - 4.0)
    return float(
        np.sum(lengths)
        + 0.40 * np.max(lengths)
        + 0.40 * turning
        + 0.08 * altitude_cost
        + 700.0 * obstacle_violation**2
        + 700.0 * separation_violation**2
        + 700.0 * altitude_violation**2
    )


def evaluate(population):
    return np.asarray([objective(individual) for individual in population])


def feasible_seed():
    values = np.zeros((UAVS, SEGMENTS - 1, 2), dtype=float)
    lateral_velocity = np.array([1.05, 1.05, 0.0, -1.05, -1.05])
    height_increment = np.array([0.25, 0.20, 0.0, -0.20, -0.25])
    values[0, :, 0] = lateral_velocity
    values[1, :, 0] = -lateral_velocity
    values[0, :, 1] = height_increment
    values[1, :, 1] = height_increment
    return values.reshape(-1)


def cubic_bspline(control_points, samples=8):
    padded = np.vstack((np.repeat(control_points[:1], 3, axis=0), control_points, np.repeat(control_points[-1:], 3, axis=0)))
    result = []
    for index in range(len(padded) - 3):
        for t_value in np.linspace(0.0, 1.0, samples, endpoint=False):
            basis = np.array(
                [
                    (-t_value**3 + 3.0 * t_value**2 - 3.0 * t_value + 1.0) / 6.0,
                    (3.0 * t_value**3 - 6.0 * t_value**2 + 4.0) / 6.0,
                    (-3.0 * t_value**3 + 3.0 * t_value**2 + 3.0 * t_value + 1.0) / 6.0,
                    t_value**3 / 6.0,
                ]
            )
            result.append(basis @ padded[index : index + 4])
    result.append(control_points[-1])
    return np.asarray(result)


def main():
    rng = np.random.default_rng(SEED)
    population_size = 26
    iterations = 60
    positions = rng.uniform(LOWER, UPPER, size=(population_size, LOWER.size))
    positions[0] = feasible_seed()
    velocities = rng.uniform(-0.10, 0.10, size=positions.shape)
    personal_best = positions.copy()
    personal_cost = evaluate(personal_best)
    initial_best = float(np.min(personal_cost))
    best_index = int(np.argmin(personal_cost))
    global_best = personal_best[best_index].copy()
    global_cost = float(personal_cost[best_index])
    metropolis_accepts = 0
    commensalism_accepts = 0

    for iteration in range(iterations):
        progress = iteration / max(iterations - 1, 1)
        inertia = 0.90 - 0.50 * progress
        cognitive = 2.20 - 1.40 * progress
        social = 0.80 + 1.40 * progress
        temperature = 0.30 * (1.0 - progress) + 0.015
        for index in range(population_size):
            velocities[index] = (
                inertia * velocities[index]
                + cognitive * rng.random(LOWER.size) * (personal_best[index] - positions[index])
                + social * rng.random(LOWER.size) * (global_best - positions[index])
            )
            velocities[index] = np.clip(velocities[index], -0.45, 0.45)
            candidate = np.clip(positions[index] + velocities[index], LOWER, UPPER)
            candidate_cost = objective(candidate)
            positions[index] = candidate
            delta = candidate_cost - personal_cost[index]
            if delta < 0.0 or rng.random() < math.exp(-max(delta, 0.0) / temperature):
                if delta >= 0.0:
                    metropolis_accepts += 1
                personal_best[index] = candidate
                personal_cost[index] = candidate_cost
            if candidate_cost < global_cost:
                global_best = candidate.copy()
                global_cost = float(candidate_cost)

        order = rng.permutation(population_size)
        for pair_start in range(0, population_size - 1, 2):
            left, right = int(order[pair_start]), int(order[pair_start + 1])
            random_scale = rng.uniform(-1.0, 1.0)
            left_trial = np.clip(positions[left] + random_scale * (global_best - positions[right]), LOWER, UPPER)
            right_trial = np.clip(positions[right] + random_scale * (global_best - positions[left]), LOWER, UPPER)
            for index, trial in ((left, left_trial), (right, right_trial)):
                trial_cost = objective(trial)
                current_cost = objective(positions[index])
                if trial_cost < current_cost:
                    positions[index] = trial
                    commensalism_accepts += 1
                    if trial_cost < personal_cost[index]:
                        personal_best[index] = trial
                        personal_cost[index] = trial_cost
                    if trial_cost < global_cost:
                        global_best = trial.copy()
                        global_cost = float(trial_cost)

    paths, lengths, turning, clearance, separation, altitude_min, altitude_max, _ = statistics(global_best)
    smoothed = [cubic_bspline(path) for path in paths]
    smoothed_lengths = [float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) for path in smoothed]
    metrics = {
        "initial_best_cost": initial_best,
        "best_cost": global_cost,
        "relative_improvement": float((initial_best - global_cost) / max(initial_best, 1.0e-12)),
        "common_arrival_time": float(SEGMENTS),
        "total_raw_path_length": float(np.sum(lengths)),
        "maximum_raw_path_length": float(np.max(lengths)),
        "total_bspline_path_length": float(np.sum(smoothed_lengths)),
        "turning_angle_sum": turning,
        "minimum_obstacle_clearance": clearance,
        "minimum_uav_separation": separation,
        "minimum_altitude": altitude_min,
        "maximum_altitude": altitude_max,
        "metropolis_acceptances": int(metropolis_accepts),
        "commensalism_acceptances": int(commensalism_accepts),
        "iterations": iterations,
    }
    numeric = np.asarray(list(metrics.values()), dtype=float)
    result = {
        "paper": PAPER,
        "algorithm": "HIPSO-MSOS with timestamp segmentation and cubic B-spline smoothing",
        "seed": SEED,
        "metrics": metrics,
        "checks": {
            "finite_metrics": bool(np.all(np.isfinite(numeric))),
            "best_not_worse_than_initial": bool(global_cost <= initial_best + 1.0e-10),
            "common_timestamp_count": bool(all(len(path) == SEGMENTS + 1 for path in paths)),
            "obstacle_clearance_positive": bool(clearance > 0.0),
            "uav_separation_satisfied": bool(separation >= 1.0),
            "altitude_constraint_satisfied": bool(altitude_min >= 0.40 and altitude_max <= 4.0),
            "endpoints_preserved": bool(np.allclose(paths[:, 0], STARTS) and np.allclose(paths[:, -1], GOALS)),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
