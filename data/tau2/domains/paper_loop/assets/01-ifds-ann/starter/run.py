#!/usr/bin/env python3
"""Small deterministic NumPy reconstruction of ANN-guided IFDS planning."""

import json
import math

import numpy as np


PAPER = "3D real-time dynamic path planning for UAV based on improved interfered fluid dynamical system and artificial neural network.pdf"
SEED = 20240101


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1.0e-12)


def _ann_parameters(position, goal, centers, velocities, radii):
    """Deterministic MLP surrogate for the unavailable trained network."""
    offsets = centers - position
    distances = np.linalg.norm(offsets, axis=1)
    clearances = distances - radii
    nearest = int(np.argmin(clearances))
    goal_direction = _unit(goal - position)
    obstacle_direction = _unit(offsets[nearest])
    features = np.array(
        [
            min(np.linalg.norm(goal - position) / 15.0, 1.5),
            np.clip(clearances[nearest] / 4.0, -1.0, 1.5),
            float(np.dot(goal_direction, obstacle_direction)),
            min(np.linalg.norm(velocities[nearest]), 1.0),
            1.0,
        ],
        dtype=float,
    )
    weights_1 = np.array(
        [
            [0.5, -1.4, 0.8, 0.2, 0.1],
            [-0.2, 1.0, -0.8, 0.4, 0.5],
            [0.3, -0.9, 1.1, 0.7, -0.2],
            [0.6, 0.2, -0.4, 0.8, 0.3],
            [-0.1, -1.2, 0.5, -0.2, 0.8],
            [0.2, 0.5, 0.3, -0.5, 0.4],
        ],
        dtype=float,
    )
    bias_1 = np.array([0.2, -0.1, 0.0, 0.1, 0.3, -0.2])
    weights_2 = np.array(
        [
            [0.9, -0.2, 0.4, 0.1, 0.6, -0.3],
            [-0.1, 0.7, 0.5, -0.2, 0.2, 0.8],
            [0.5, -0.7, 0.9, 0.3, -0.5, 0.4],
        ],
        dtype=float,
    )
    hidden = np.maximum(weights_1 @ features + bias_1, 0.0)
    raw = np.tanh(weights_2 @ hidden + np.array([0.1, 0.0, -0.2]))
    repulsion = 0.75 + 1.25 * (raw[0] + 1.0) / 2.0
    tangential = 0.65 + 1.10 * (raw[1] + 1.0) / 2.0
    alpha = 0.85 * math.pi * raw[2]
    return float(repulsion), float(tangential), float(alpha), nearest


def _sphere_terms(position, center, radius, alpha):
    relative = position - center
    radius_sq = radius * radius
    field = max(float(np.dot(relative, relative) / radius_sq), 1.0e-6)
    normal = 2.0 * relative / radius_sq
    normal_norm = max(float(np.linalg.norm(normal)), 1.0e-12)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(_unit(normal), reference))) > 0.92:
        reference = np.array([0.0, 1.0, 0.0])
    tangent_1 = _unit(np.cross(normal, reference))
    tangent_2 = _unit(np.cross(normal, tangent_1))
    tangent = math.cos(alpha) * tangent_1 + math.sin(alpha) * tangent_2
    return field, normal, tangent, normal_norm


def _ifds_velocity(position, goal, centers, radii, obstacle_velocities, speed, params):
    repulsion, tangential, alpha = params
    sigma = speed * _unit(goal - position)
    matrices = []
    raw_weights = []
    for center, radius in zip(centers, radii):
        field, normal, tangent, normal_norm = _sphere_terms(position, center, radius, alpha)
        repulsive_matrix = np.outer(normal, normal) / (
            (field ** (1.0 / repulsion)) * normal_norm * normal_norm
        )
        tangential_matrix = np.outer(tangent, normal) / (
            (field ** (1.0 / tangential)) * max(np.linalg.norm(tangent), 1.0e-12) * normal_norm
        )
        matrices.append(np.eye(3) - repulsive_matrix + tangential_matrix)
        raw_weights.append(1.0 / max(field - 1.0, 0.05))
    weights = np.asarray(raw_weights, dtype=float)
    weights /= np.sum(weights)
    perturbation = np.sum(weights[:, None, None] * np.stack(matrices), axis=0)
    obstacle_velocity = np.sum(weights[:, None] * obstacle_velocities, axis=0)
    velocity = perturbation @ (sigma - obstacle_velocity) + obstacle_velocity
    velocity_norm = float(np.linalg.norm(velocity))
    if velocity_norm > 1.35 * speed:
        velocity *= (1.35 * speed) / velocity_norm
    return velocity


def main():
    np.random.default_rng(SEED)  # Establishes the documented deterministic seed.
    start = np.array([0.0, 0.0, 0.5])
    goal = np.array([12.0, 8.0, 4.0])
    centers = np.array([[4.6, 3.0, 1.7], [8.2, 5.4, 2.9]], dtype=float)
    radii = np.array([1.25, 1.15], dtype=float)
    obstacle_velocities = np.array([[0.0, 0.035, 0.0], [-0.025, 0.0, 0.0]], dtype=float)
    position = start.copy()
    path = [position.copy()]
    dt = 0.18
    speed = 1.0
    parameter_trace = []
    nearest_counts = np.zeros(len(centers), dtype=int)

    for _ in range(180):
        if np.linalg.norm(goal - position) <= speed * dt * 1.3:
            position = goal.copy()
            path.append(position.copy())
            break
        repulsion, tangential, alpha, nearest = _ann_parameters(
            position, goal, centers, obstacle_velocities, radii
        )
        nearest_counts[nearest] += 1
        parameter_trace.append([repulsion, tangential, alpha])
        velocity = _ifds_velocity(
            position,
            goal,
            centers,
            radii,
            obstacle_velocities,
            speed,
            (repulsion, tangential, alpha),
        )
        candidate = position + velocity * dt
        centers = centers + obstacle_velocities * dt
        for center, radius in zip(centers, radii):
            offset = candidate - center
            distance = float(np.linalg.norm(offset))
            safe_radius = radius + 0.08
            if distance < safe_radius:
                candidate = center + _unit(offset) * safe_radius
        position = candidate
        path.append(position.copy())

    path_array = np.asarray(path)
    segment_lengths = np.linalg.norm(np.diff(path_array, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    final_distance = float(np.linalg.norm(path_array[-1] - goal))
    # Reconstruct obstacle centers at every path time to measure dynamic clearance.
    time_index = np.arange(len(path_array), dtype=float)[:, None, None]
    initial_centers = np.array([[4.6, 3.0, 1.7], [8.2, 5.4, 2.9]], dtype=float)
    center_history = initial_centers[None, :, :] + time_index * dt * obstacle_velocities[None, :, :]
    clearances = np.linalg.norm(path_array[:, None, :] - center_history, axis=2) - radii[None, :]
    min_clearance = float(np.min(clearances))
    params_array = np.asarray(parameter_trace)
    metrics = {
        "steps": int(len(path_array) - 1),
        "path_length": path_length,
        "straight_line_length": float(np.linalg.norm(goal - start)),
        "final_goal_distance": final_distance,
        "minimum_dynamic_clearance": min_clearance,
        "mean_repulsion_coefficient": float(np.mean(params_array[:, 0])),
        "mean_tangential_coefficient": float(np.mean(params_array[:, 1])),
        "nearest_obstacle_counts": [int(value) for value in nearest_counts],
    }
    finite = bool(np.all(np.isfinite(np.asarray([value for value in metrics.values() if not isinstance(value, list)]))))
    result = {
        "paper": PAPER,
        "algorithm": "ANN-guided interfered fluid dynamical system (IFDS) core",
        "seed": SEED,
        "metrics": metrics,
        "checks": {
            "finite_metrics": finite,
            "collision_free": bool(min_clearance > 0.0),
            "goal_reached": bool(final_distance <= 0.30),
            "ann_parameters_bounded": bool(
                np.all((params_array[:, 0] >= 0.75) & (params_array[:, 0] <= 2.0))
                and np.all((params_array[:, 1] >= 0.65) & (params_array[:, 1] <= 1.75))
            ),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
