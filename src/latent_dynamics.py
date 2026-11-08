"""Deterministic linear latent world model with ridge-regularized system identification.

This baseline intentionally uses observable states as latent representations.
It is suitable for testing rollout evaluation before adding neural encoders.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class LinearWorldModel:
    ridge: float = 1e-6

    def fit(self, states, actions, next_states):
        s = np.asarray(states, dtype=float)
        a = np.asarray(actions, dtype=float)
        y = np.asarray(next_states, dtype=float)
        if s.ndim != 2 or a.ndim != 2 or y.shape != s.shape or len(a) != len(s):
            raise ValueError("Expected aligned 2D state/action/next-state arrays")
        if not len(s) or not np.isfinite(s).all() or not np.isfinite(a).all() or not np.isfinite(y).all():
            raise ValueError("Data must be nonempty and finite")
        if self.ridge < 0:
            raise ValueError("ridge must be nonnegative")
        x = np.column_stack([s, a, np.ones(len(s))])
        penalty = np.eye(x.shape[1]) * self.ridge
        penalty[-1, -1] = 0
        self.weights = np.linalg.solve(x.T @ x + penalty, x.T @ y) if self.ridge > 0 else np.linalg.lstsq(x, y, rcond=None)[0]
        self.state_dim, self.action_dim = s.shape[1], a.shape[1]
        return self

    def predict(self, state, action):
        if not hasattr(self, "weights"):
            raise RuntimeError("Fit the model before predicting")
        s, a = np.asarray(state, dtype=float), np.asarray(action, dtype=float)
        if s.shape != (self.state_dim,) or a.shape != (self.action_dim,):
            raise ValueError("Incorrect state or action dimensions")
        return np.concatenate([s, a, [1.0]]) @ self.weights

    def rollout(self, initial_state, actions):
        states = [np.asarray(initial_state, dtype=float)]
        for action in actions:
            states.append(self.predict(states[-1], action))
        return np.stack(states)

def rollout_rmse(predicted, actual):
    p, a = np.asarray(predicted), np.asarray(actual)
    if p.shape != a.shape or not p.size:
        raise ValueError("Predictions and targets must have matching nonempty shapes")
    return float(np.sqrt(np.mean((p - a) ** 2)))
