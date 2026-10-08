#!/usr/bin/env python3
"""Deterministic NumPy reconstruction of the QL-MOPSO core pipeline."""

from __future__ import annotations

import json
import math

import numpy as np


SEED = 20251019
RNG = np.random.default_rng(SEED)
ACTIONS = np.array(
    [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    dtype=int,
)


def make_grid() -> np.ndarray:
    grid = np.zeros((12, 12), dtype=np.int8)
    grid[2:10, 5] = 1
    grid[6, 5] = 0
    grid[3, 1:5] = 1
    grid[9, 7:11] = 1
    return grid


def diagonal_corner_is_clear(
    grid: np.ndarray, state: tuple[int, int], candidate: tuple[int, int]
) -> bool:
    delta = np.asarray(candidate) - np.asarray(state)
    if not np.all(np.abs(delta) == 1):
        return True
    side_a = (state[0] + int(delta[0]), state[1])
    side_b = (state[0], state[1] + int(delta[1]))
    rows, columns = grid.shape
    if not all(0 <= row < rows and 0 <= column < columns for row, column in (side_a, side_b)):
        return False
    return bool(grid[side_a] == 0 and grid[side_b] == 0)


def transition_is_valid(
    grid: np.ndarray, state: tuple[int, int], candidate: tuple[int, int]
) -> bool:
    rows, columns = grid.shape
    delta = np.asarray(candidate) - np.asarray(state)
    neighboring = bool(np.max(np.abs(delta)) == 1 and np.any(delta != 0))
    inside = bool(0 <= candidate[0] < rows and 0 <= candidate[1] < columns)
    if not neighboring or not inside or grid[candidate] != 0:
        return False
    return diagonal_corner_is_clear(grid, state, candidate)


def validate_q_path(path: np.ndarray, grid: np.ndarray) -> tuple[bool, bool]:
    if len(path) < 2 or not np.allclose(path, np.rint(path)):
        return False, False
    states = [tuple(int(value) for value in point) for point in path]
    every_step_valid = all(
        transition_is_valid(grid, state, candidate)
        for state, candidate in zip(states[:-1], states[1:])
    )
    no_corner_cutting = all(
        diagonal_corner_is_clear(grid, state, candidate)
        for state, candidate in zip(states[:-1], states[1:])
    )
    return every_step_valid, no_corner_cutting


def train_q_path(grid: np.ndarray) -> tuple[np.ndarray, float]:
    start, goal = (0, 0), (11, 11)
    q = np.zeros((*grid.shape, len(ACTIONS)), dtype=float)
    alpha, gamma = 0.15, 0.95
    epsilon = 0.35

    for episode in range(2_500):
        state = start
        for _ in range(240):
            if RNG.random() < epsilon:
                action = int(RNG.integers(len(ACTIONS)))
            else:
                maxima = np.flatnonzero(q[state] == q[state].max())
                action = int(RNG.choice(maxima))
            candidate = tuple((np.asarray(state) + ACTIONS[action]).tolist())
            if not transition_is_valid(grid, state, candidate):
                nxt, reward = state, -15.0
            else:
                nxt = candidate
                reward = 100.0 if nxt == goal else -1.0
            q[state + (action,)] += alpha * (
                reward + gamma * float(q[nxt].max()) - q[state + (action,)]
            )
            state = nxt
            if state == goal:
                break
        epsilon = max(0.03, epsilon * 0.998)

    path = [start]
    state = start
    visited = {start}
    for _ in range(100):
        ranked = np.argsort(q[state])[::-1]
        nxt = None
        for action in ranked:
            candidate = tuple((np.asarray(state) + ACTIONS[action]).tolist())
            if transition_is_valid(grid, state, candidate) and candidate not in visited:
                nxt = candidate
                break
        if nxt is None:
            raise RuntimeError("Q policy did not yield a loop-free path")
        path.append(nxt)
        visited.add(nxt)
        state = nxt
        if state == goal:
            break
    if state != goal:
        raise RuntimeError("Q policy did not reach the goal")
    return np.asarray(path, dtype=float), float(q[start].max())


def resample_guidance(path_2d: np.ndarray, interior: int = 7) -> np.ndarray:
    segment = np.linalg.norm(np.diff(path_2d, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    samples = np.linspace(0.0, cumulative[-1], interior + 2)
    coords = np.column_stack(
        [np.interp(samples, cumulative, path_2d[:, d]) for d in range(2)]
    )
    return np.column_stack((coords, np.full(len(coords), 4.0)))


def objectives(flat: np.ndarray, guidance: np.ndarray) -> np.ndarray:
    start, end = guidance[0], guidance[-1]
    points = np.vstack((start, flat.reshape(-1, 3), end))
    delta = np.diff(points, axis=0)
    length = float(np.linalg.norm(delta, axis=1).sum())
    direct = float(np.linalg.norm(end - start))
    tortuosity = 1.0 - direct / max(length, 1e-12)

    threats = ((np.array([5.0, 6.0, 4.0]), 1.3), (np.array([8.5, 8.0, 4.0]), 1.0))
    danger = 2.0
    threat_penalty = 0.0
    for point in points[1:-1]:
        for center, radius in threats:
            distance = float(np.linalg.norm(point - center))
            intrusion = max(0.0, radius + danger - distance) / danger
            threat_penalty += min(3.0, intrusion)
    threat_penalty /= max(1, (len(points) - 2) * len(threats))

    altitude = float(np.mean(np.abs(points[:, 2] - 4.0)) / 4.0)
    if len(delta) > 1:
        unit = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        smooth = float(np.mean(1.0 - np.sum(unit[:-1] * unit[1:], axis=1)))
    else:
        smooth = 0.0
    deviation = float(np.mean(np.linalg.norm(points - guidance, axis=1)) / 12.0)
    return np.array([tortuosity, threat_penalty, altitude, smooth, deviation])


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b + 1e-12) and np.any(a < b - 1e-12))


def nondominated(costs: np.ndarray) -> np.ndarray:
    keep = np.ones(len(costs), dtype=bool)
    for i in range(len(costs)):
        if keep[i] and any(dominates(costs[j], costs[i]) for j in range(len(costs)) if j != i):
            keep[i] = False
    return np.flatnonzero(keep)


def run_mopso(
    guidance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    interior = guidance[1:-1]
    particles = 34
    lower = np.tile(np.array([0.0, 0.0, 1.5]), len(interior))
    upper = np.tile(np.array([11.0, 11.0, 7.0]), len(interior))
    guide_flat = interior.ravel()
    position = np.clip(
        guide_flat + RNG.normal(0.0, 1.25, size=(particles, guide_flat.size)), lower, upper
    )
    position[0] = guide_flat
    velocity = RNG.normal(0.0, 0.15, size=position.shape)
    costs = np.array([objectives(p, guidance) for p in position])
    weights = np.array([0.24, 0.28, 0.12, 0.16, 0.20])
    initial_score = float(np.min(costs @ weights))
    pbest, pbest_cost = position.copy(), costs.copy()
    archive_pos, archive_cost = position[nondominated(costs)], costs[nondominated(costs)]
    accepted_updates = 0
    strict_weighted_improvements = 0

    for iteration in range(65):
        inertia = 0.9 - 0.5 * iteration / 64.0
        leader = archive_pos[RNG.integers(len(archive_pos), size=particles)]
        r1, r2, r3 = (RNG.random(position.shape) for _ in range(3))
        velocity = (
            inertia * velocity
            + 1.4 * r1 * (pbest - position)
            + 1.5 * r2 * (leader - position)
            + 0.65 * r3 * (guide_flat - position)
        )
        position = np.clip(position + velocity, lower, upper)
        costs = np.array([objectives(p, guidance) for p in position])
        for i in range(particles):
            old_weighted_cost = float(pbest_cost[i] @ weights)
            if dominates(costs[i], pbest_cost[i]) or (
                not dominates(pbest_cost[i], costs[i]) and costs[i] @ weights < pbest_cost[i] @ weights
            ):
                accepted_updates += 1
                if float(costs[i] @ weights) < old_weighted_cost - 1.0e-12:
                    strict_weighted_improvements += 1
                pbest[i], pbest_cost[i] = position[i].copy(), costs[i].copy()
        merged_pos = np.vstack((archive_pos, position))
        merged_cost = np.vstack((archive_cost, costs))
        keep = nondominated(merged_cost)
        if len(keep) > 70:
            order = np.argsort(merged_cost[keep] @ weights)
            keep = keep[order[:70]]
        archive_pos, archive_cost = merged_pos[keep], merged_cost[keep]

    scores = archive_cost @ weights
    best = int(np.argmin(scores))
    return (
        archive_pos[best],
        archive_cost,
        initial_score,
        float(scores[best]),
        accepted_updates,
        strict_weighted_improvements,
    )


def main() -> None:
    grid = make_grid()
    path_2d, q_start = train_q_path(grid)
    valid_steps, no_corner_cutting = validate_q_path(path_2d, grid)
    diagonal_steps = int(np.count_nonzero(np.all(np.abs(np.diff(path_2d, axis=0)) == 1, axis=1)))
    guidance = resample_guidance(path_2d)
    best, archive_cost, initial, final, accepted_updates, strict_improvements = run_mopso(
        guidance
    )
    result = {
        "paper": "Reinforcement Learning-Guided Particle Swarm Optimization",
        "algorithm": "QL-MOPSO core reconstruction",
        "seed": SEED,
        "metrics": {
            "q_path_nodes": int(len(path_2d)),
            "q_path_diagonal_steps": diagonal_steps,
            "q_start_value": round(q_start, 6),
            "archive_size": int(len(archive_cost)),
            "initial_weighted_cost": round(initial, 6),
            "final_weighted_cost": round(final, 6),
            "accepted_personal_best_updates": accepted_updates,
            "strict_weighted_personal_best_improvements": strict_improvements,
            "best_objectives": [round(float(x), 6) for x in objectives(best, guidance)],
        },
        "checks": {
            "q_path_reaches_goal": bool(np.array_equal(path_2d[-1], [11.0, 11.0])),
            "q_path_every_step_collision_free": valid_steps,
            "q_path_no_corner_cutting": no_corner_cutting,
            "guided_seed_preserved": bool(final <= initial + 1e-12),
            "mopso_updates_exercised": bool(
                accepted_updates > 0 and strict_improvements > 0
            ),
            "finite": bool(np.isfinite(archive_cost).all()),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
