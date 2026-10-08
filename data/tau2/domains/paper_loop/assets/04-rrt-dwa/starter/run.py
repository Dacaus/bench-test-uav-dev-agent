#!/usr/bin/env python3
"""Deterministic NumPy-only core of the paper's hybrid RRT-DWA planner."""

import json
import math

import numpy as np


PAPER = "A hybrid RRT-DWA path planning .pdf"
SEED = 20260404
START = np.array([1.0, 1.0])
GOAL = np.array([19.0, 18.0])
STATIC_CENTERS = np.array([[5.5, 6.4], [10.3, 9.4], [14.3, 13.8]])
STATIC_RADII = np.array([1.45, 1.65, 1.35])
VEHICLE_RADIUS = 0.22


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def segment_clear(start, end, centers=STATIC_CENTERS, radii=STATIC_RADII):
    samples = start[None, :] + np.linspace(0.0, 1.0, 25)[:, None] * (end - start)[None, :]
    clearance = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2) - radii[None, :] - VEHICLE_RADIUS
    return bool(np.min(clearance) > 0.0)


def rrt_plan(rng):
    nodes = [START.copy()]
    parents = [-1]
    step_size = 1.15
    goal_index = None
    for _ in range(1800):
        sample = GOAL if rng.random() < 0.22 else rng.uniform([0.3, 0.3], [19.7, 19.7])
        node_array = np.asarray(nodes)
        nearest = int(np.argmin(np.linalg.norm(node_array - sample, axis=1)))
        direction = sample - node_array[nearest]
        distance = float(np.linalg.norm(direction))
        if distance < 1.0e-12:
            continue
        candidate = node_array[nearest] + direction / distance * min(step_size, distance)
        if not segment_clear(node_array[nearest], candidate):
            continue
        nodes.append(candidate)
        parents.append(nearest)
        new_index = len(nodes) - 1
        if np.linalg.norm(candidate - GOAL) <= step_size and segment_clear(candidate, GOAL):
            nodes.append(GOAL.copy())
            parents.append(new_index)
            goal_index = len(nodes) - 1
            break
    if goal_index is None:
        raise RuntimeError("deterministic RRT failed to connect the goal")
    indices = []
    cursor = goal_index
    while cursor >= 0:
        indices.append(cursor)
        cursor = parents[cursor]
    raw_path = np.asarray([nodes[index] for index in reversed(indices)])
    simplified = [raw_path[0]]
    anchor = 0
    while anchor < len(raw_path) - 1:
        candidate = len(raw_path) - 1
        while candidate > anchor + 1 and not segment_clear(raw_path[anchor], raw_path[candidate]):
            candidate -= 1
        simplified.append(raw_path[candidate])
        anchor = candidate
    return raw_path, np.asarray(simplified), len(nodes)


def select_subgoals(path):
    if len(path) <= 2:
        return path[1:].copy()
    chosen = []
    last = path[0]
    for index in range(1, len(path) - 1):
        incoming = path[index] - path[index - 1]
        outgoing = path[index + 1] - path[index]
        angle = abs(wrap_angle(math.atan2(outgoing[1], outgoing[0]) - math.atan2(incoming[1], incoming[0])))
        if np.linalg.norm(path[index] - last) > 1.3 and angle > math.radians(12.0):
            chosen.append(path[index])
            last = path[index]
    chosen.append(path[-1])
    return np.asarray(chosen)


def predict_unicycle(state, linear_velocity, angular_velocity, dt=0.25, horizon=1.5):
    steps = int(round(horizon / dt))
    trajectory = np.empty((steps, 3), dtype=float)
    x, y, yaw = state[:3]
    for index in range(steps):
        yaw = wrap_angle(yaw + angular_velocity * dt)
        x += linear_velocity * math.cos(yaw) * dt
        y += linear_velocity * math.sin(yaw) * dt
        trajectory[index] = [x, y, yaw]
    return trajectory


def dwa_command(state, subgoal, final_goal, dynamic_position, dynamic_velocity):
    dt = 0.25
    max_linear, max_angular = 2.0, 1.15
    linear_acceleration, angular_acceleration = 0.85, 1.25
    linear_low = max(0.0, state[3] - linear_acceleration * dt)
    linear_high = min(max_linear, state[3] + linear_acceleration * dt)
    angular_low = max(-max_angular, state[4] - angular_acceleration * dt)
    angular_high = min(max_angular, state[4] + angular_acceleration * dt)
    linear_samples = np.linspace(linear_low, linear_high, 7)
    angular_samples = np.linspace(angular_low, angular_high, 11)
    best = None
    best_score = -np.inf
    closest_dynamic_prediction = np.inf
    for linear_velocity in linear_samples:
        for angular_velocity in angular_samples:
            trajectory = predict_unicycle(state, linear_velocity, angular_velocity)
            points = trajectory[:, :2]
            static_clearance = np.min(
                np.linalg.norm(points[:, None, :] - STATIC_CENTERS[None, :, :], axis=2)
                - STATIC_RADII[None, :]
                - VEHICLE_RADIUS
            )
            times = 0.25 * np.arange(1, len(points) + 1)
            predicted_dynamic = dynamic_position[None, :] + times[:, None] * dynamic_velocity[None, :]
            dynamic_clearance = np.min(np.linalg.norm(points - predicted_dynamic, axis=1) - 0.68 - VEHICLE_RADIUS)
            clearance = float(min(static_clearance, dynamic_clearance))
            closest_dynamic_prediction = min(closest_dynamic_prediction, float(dynamic_clearance))
            if clearance <= 0.0:
                continue
            # Paper's optimized safety window, Eq. (6)-(7).
            safe_speed = math.sqrt(max(0.0, 2.0 * (clearance + 0.20) * linear_acceleration))
            if linear_velocity > safe_speed + 1.0e-12:
                continue
            subgoal_distance = float(np.linalg.norm(points[-1] - subgoal))
            final_distance = float(np.linalg.norm(points[-1] - final_goal))
            delta_velocity = abs(linear_velocity - state[3])
            delta_angular = abs(angular_velocity - state[4])
            goal_weight = 2.6 * math.exp(-0.025 * final_distance)
            obstacle_weight = 1.8 * clearance / (clearance + 0.70)
            smooth_weight = 0.55 * 0.40 / (0.40 + delta_velocity + delta_angular)
            score = (
                -goal_weight * subgoal_distance
                + obstacle_weight * math.tanh(clearance)
                + smooth_weight / (1.0 + delta_velocity / max_linear + delta_angular / max_angular)
                + 0.08 * linear_velocity
            )
            if score > best_score:
                best_score = score
                best = (float(linear_velocity), float(angular_velocity), clearance)
    if best is None:
        desired = math.atan2(subgoal[1] - state[1], subgoal[0] - state[0])
        return 0.0, float(np.clip(wrap_angle(desired - state[2]) / dt, -max_angular, max_angular)), 0.0, closest_dynamic_prediction
    return best[0], best[1], best[2], closest_dynamic_prediction


def main():
    rng = np.random.default_rng(SEED)
    raw_global_path, global_path, tree_nodes = rrt_plan(rng)
    subgoals = select_subgoals(global_path)
    initial_yaw = math.atan2(subgoals[0, 1] - START[1], subgoals[0, 0] - START[0])
    state = np.array([START[0], START[1], initial_yaw, 0.0, 0.0], dtype=float)
    dynamic_position = np.array([9.6, 15.5], dtype=float)
    dynamic_velocity = np.array([0.10, -0.23], dtype=float)
    path = [state[:2].copy()]
    dynamic_history = [dynamic_position.copy()]
    subgoal_index = 0
    avoidance_triggers = 0
    command_changes = []
    sampled_clearances = []
    dt = 0.25

    for _ in range(260):
        if np.linalg.norm(state[:2] - GOAL) <= 0.65:
            break
        while subgoal_index < len(subgoals) - 1 and np.linalg.norm(state[:2] - subgoals[subgoal_index]) <= 1.0:
            subgoal_index += 1
        linear, angular, clearance, dynamic_prediction = dwa_command(
            state, subgoals[subgoal_index], GOAL, dynamic_position, dynamic_velocity
        )
        if dynamic_prediction < 1.2:
            avoidance_triggers += 1
        command_changes.append(abs(linear - state[3]) + abs(angular - state[4]))
        sampled_clearances.append(clearance)
        state[2] = wrap_angle(state[2] + angular * dt)
        state[0] += linear * math.cos(state[2]) * dt
        state[1] += linear * math.sin(state[2]) * dt
        state[3] = linear
        state[4] = angular
        dynamic_position = dynamic_position + dynamic_velocity * dt
        path.append(state[:2].copy())
        dynamic_history.append(dynamic_position.copy())

    local_path = np.asarray(path)
    dynamic_array = np.asarray(dynamic_history)
    static_clearance = np.min(
        np.linalg.norm(local_path[:, None, :] - STATIC_CENTERS[None, :, :], axis=2)
        - STATIC_RADII[None, :]
        - VEHICLE_RADIUS
    )
    dynamic_clearance = np.min(np.linalg.norm(local_path - dynamic_array, axis=1) - 0.68 - VEHICLE_RADIUS)
    minimum_clearance = float(min(static_clearance, dynamic_clearance))
    local_length = float(np.sum(np.linalg.norm(np.diff(local_path, axis=0), axis=1)))
    global_length = float(np.sum(np.linalg.norm(np.diff(global_path, axis=0), axis=1)))
    final_distance = float(np.linalg.norm(local_path[-1] - GOAL))
    metrics = {
        "rrt_tree_nodes": int(tree_nodes),
        "raw_rrt_waypoints": int(len(raw_global_path)),
        "simplified_global_waypoints": int(len(global_path)),
        "selected_subgoals": int(len(subgoals)),
        "global_path_length": global_length,
        "executed_path_length": local_length,
        "executed_steps": int(len(local_path) - 1),
        "final_goal_distance": final_distance,
        "minimum_clearance": minimum_clearance,
        "dynamic_avoidance_triggers": int(avoidance_triggers),
        "mean_command_change": float(np.mean(command_changes)) if command_changes else 0.0,
    }
    numeric = np.asarray(list(metrics.values()), dtype=float)
    result = {
        "paper": PAPER,
        "algorithm": "goal-biased RRT plus dynamically weighted DWA with subgoals",
        "seed": SEED,
        "metrics": metrics,
        "checks": {
            "finite_metrics": bool(np.all(np.isfinite(numeric))),
            "global_path_collision_free": bool(all(segment_clear(global_path[i], global_path[i + 1]) for i in range(len(global_path) - 1))),
            "executed_path_collision_free": bool(minimum_clearance > 0.0),
            "goal_reached": bool(final_distance <= 0.80),
            "world_bounds_respected": bool(np.all((local_path >= 0.0) & (local_path <= 20.0))),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
