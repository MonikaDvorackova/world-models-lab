import numpy as np
import pytest
from src.latent_dynamics import LinearWorldModel, rollout_rmse

def test_identifies_controlled_dynamics():
    rng = np.random.default_rng(7)
    states = rng.normal(size=(300, 2))
    actions = rng.normal(size=(300, 1))
    next_states = states @ np.array([[0.9, 0.1], [-0.2, 0.8]]) + actions @ np.array([[0.5, -0.3]]) + 0.1
    model = LinearWorldModel().fit(states, actions, next_states)
    prediction = np.stack([model.predict(s, a) for s, a in zip(states, actions)])
    assert rollout_rmse(prediction, next_states) < 1e-5

def test_rollout_has_initial_state_and_steps():
    model = LinearWorldModel().fit([[0], [1], [2]], [[1], [1], [1]], [[1], [2], [3]])
    assert np.allclose(model.rollout([0], [[1], [1], [1]]).ravel(), [0, 1, 2, 3], atol=1e-5)

def test_rejects_invalid_data():
    with pytest.raises(ValueError):
        LinearWorldModel().fit([[0]], [[float("nan")]], [[1]])
    with pytest.raises(RuntimeError):
        LinearWorldModel().predict([0], [0])
