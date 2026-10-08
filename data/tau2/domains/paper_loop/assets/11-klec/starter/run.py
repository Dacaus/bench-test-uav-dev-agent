#!/usr/bin/env python3
"""NumPy-only reconstruction of KLEC using the paper's KLDE example."""

import json

import numpy as np


PAPER = "Knowledge_Learning_for_Evolutionary_Computation.pdf"
ALGORITHM = "Knowledge Learning Differential Evolution (KLDE)"
SEED = 202511


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


class KnowledgeLibrary:
    """D-16-16-D FNN matching the KLM architecture in Figure 3."""

    def __init__(self, rng: np.random.Generator, dimensions: int) -> None:
        hidden = 16
        self.w1 = rng.normal(0.0, 0.16, size=(dimensions, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, 0.16, size=(hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.w3 = rng.normal(0.0, 0.12, size=(hidden, dimensions))
        self.b3 = np.zeros(dimensions)
        self.trained = False

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        h1 = sigmoid(x @ self.w1 + self.b1)
        h2 = sigmoid(h1 @ self.w2 + self.b2)
        output = h2 @ self.w3 + self.b3
        return h1, h2, output

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x.reshape(1, -1))[2][0]

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 8) -> float:
        learning_rate = 0.025
        for _ in range(epochs):
            h1, h2, output = self.forward(x)
            grad_out = 2.0 * (output - y) / max(1, len(x))
            grad_w3 = h2.T @ grad_out
            grad_b3 = np.sum(grad_out, axis=0)
            grad_h2 = (grad_out @ self.w3.T) * h2 * (1.0 - h2)
            grad_w2 = h1.T @ grad_h2
            grad_b2 = np.sum(grad_h2, axis=0)
            grad_h1 = (grad_h2 @ self.w2.T) * h1 * (1.0 - h1)
            grad_w1 = x.T @ grad_h1
            grad_b1 = np.sum(grad_h1, axis=0)

            for gradient in (grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3):
                np.clip(gradient, -2.0, 2.0, out=gradient)
            self.w1 -= learning_rate * grad_w1
            self.b1 -= learning_rate * grad_b1
            self.w2 -= learning_rate * grad_w2
            self.b2 -= learning_rate * grad_b2
            self.w3 -= learning_rate * grad_w3
            self.b3 -= learning_rate * grad_b3
        self.trained = True
        prediction = self.forward(x)[2]
        return float(np.mean((prediction - y) ** 2))


def objective(x: np.ndarray) -> np.ndarray:
    shift = np.array([0.9, -1.1, 0.4, -0.6, 1.2, -0.25])
    z = x - shift
    return np.sum(z * z + 0.2 * (1.0 - np.cos(4.0 * z)), axis=-1)


def normalized_position(x: np.ndarray, lower: float, span: float) -> np.ndarray:
    return 2.0 * (x - lower) / span - 1.0


def main() -> None:
    rng = np.random.default_rng(SEED)
    population_size, dimensions, generations = 32, 6, 55
    lower, upper = -5.0, 5.0
    span = upper - lower
    population = rng.uniform(lower, upper, size=(population_size, dimensions))
    values = objective(population)
    initial_best = float(np.min(values))
    knowledge = KnowledgeLibrary(rng, dimensions)
    best_trace = [initial_best]
    experience_count = 0
    knowledge_trials = 0
    training_rounds = 0
    last_mse = 0.0
    differential_weight, crossover_rate, learning_probability = 0.65, 0.9, 0.5

    for _ in range(generations):
        experience_positions = []
        experience_directions = []
        for i in range(population_size):
            choices = np.delete(np.arange(population_size), i)
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)

            if knowledge.trained and rng.random() < learning_probability:
                query = normalized_position(population[r1], lower, span)
                learned_direction = knowledge.predict(query) * span
                mutant = population[r1] + 2.0 * rng.random() * learned_direction
                knowledge_trials += 1
            else:
                mutant = population[r1] + differential_weight * (
                    population[r2] - population[r3]
                )

            mutant = np.clip(mutant, lower, upper)
            mask = rng.random(dimensions) <= crossover_rate
            mask[int(rng.integers(dimensions))] = True
            trial = np.where(mask, mutant, population[i])
            trial_value = float(objective(trial.reshape(1, -1))[0])
            if trial_value < float(values[i]):
                experience_positions.append(
                    normalized_position(population[i], lower, span)
                )
                experience_directions.append((trial - population[i]) / span)
                population[i] = trial
                values[i] = trial_value

        if experience_positions:
            x_train = np.asarray(experience_positions)
            y_train = np.asarray(experience_directions)
            last_mse = knowledge.train(x_train, y_train)
            experience_count += len(x_train)
            training_rounds += 1
        best_trace.append(float(np.min(values)))

    best_index = int(np.argmin(values))
    known_optimum = np.array([0.9, -1.1, 0.4, -0.6, 1.2, -0.25])
    result = {
        "paper": PAPER,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "metrics": {
            "dimensions": dimensions,
            "population": population_size,
            "generations": generations,
            "initial_best": initial_best,
            "final_best": float(values[best_index]),
            "improvement": initial_best - float(values[best_index]),
            "distance_to_known_optimum": float(
                np.linalg.norm(population[best_index] - known_optimum)
            ),
            "successful_experiences": experience_count,
            "knowledge_training_rounds": training_rounds,
            "knowledge_guided_trials": knowledge_trials,
            "final_training_mse": last_mse,
        },
        "checks": {
            "finite": bool(
                np.isfinite(values).all()
                and np.isfinite(last_mse)
                and np.isfinite(population).all()
            ),
            "knowledge_was_learned": bool(knowledge.trained and training_rounds > 0),
            "knowledge_was_used": bool(knowledge_trials > 0),
            "best_so_far_nonincreasing": bool(np.all(np.diff(best_trace) <= 1e-12)),
            "bounds_respected": bool(
                np.all((population >= lower) & (population <= upper))
            ),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
