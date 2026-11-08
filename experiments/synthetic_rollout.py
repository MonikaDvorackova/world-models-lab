"""Reproducible linear system-identification benchmark."""
import json
import numpy as np
from src.latent_dynamics import LinearWorldModel, rollout_rmse

def run(seed=42, n_train=1000, horizon=25):
    rng = np.random.default_rng(seed)
    A = np.array([[0.8, 0.1], [-0.1, 0.85]])
    B = np.array([[0.4, -0.2]])
    states = rng.normal(size=(n_train, 2))
    actions = rng.normal(size=(n_train, 1))
    targets = states @ A + actions @ B
    model = LinearWorldModel().fit(states, actions, targets)
    action_seq = rng.normal(size=(horizon, 1))
    truth = [np.array([0.3, -0.2])]
    for action in action_seq:
        truth.append(truth[-1] @ A + action @ B)
    prediction = model.rollout(truth[0], action_seq)
    return {"seed": seed, "train_samples": n_train, "horizon": horizon,
            "rollout_rmse": rollout_rmse(prediction, np.stack(truth))}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
