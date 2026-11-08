"""Seeded multi-step rollout evaluation with persistence baseline."""
import numpy as np
from src.latent_dynamics import rollout_rmse

def evaluate_trajectories(model, trajectories):
    errors, baselines = [], []
    for initial, actions, truth in trajectories:
        actual = np.asarray(truth, dtype=float)
        if actual.ndim != 2 or actual.shape[0] != len(actions)+1 or not np.isfinite(actual).all():
            raise ValueError("truth must be finite states including initial state")
        if not np.allclose(actual[0], initial):
            raise ValueError("truth must start at initial state")
        predicted = model.rollout(initial, actions)
        baseline = np.repeat(actual[:1], len(actual), axis=0)
        errors.append(np.mean((predicted[1:] - actual[1:])**2))
        baselines.append(np.mean((baseline[1:] - actual[1:])**2))
    if not errors:
        raise ValueError("At least one trajectory is required")
    return {"model_rmse": float(np.sqrt(np.mean(errors))),
            "persistence_rmse": float(np.sqrt(np.mean(baselines)))}
