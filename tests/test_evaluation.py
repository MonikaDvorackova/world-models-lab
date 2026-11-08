import numpy as np
import pytest
from src.evaluation import evaluate_trajectories
from src.latent_dynamics import LinearWorldModel

def test_model_beats_persistence():
    states=np.arange(12,dtype=float).reshape(-1,1)
    actions=np.ones((11,1))
    model=LinearWorldModel().fit(states[:-1],actions,states[1:])
    result=evaluate_trajectories(model,[(np.array([0.]),np.ones((4,1)),np.arange(5.).reshape(-1,1))])
    assert result["model_rmse"] < result["persistence_rmse"]
    with pytest.raises(ValueError):
        evaluate_trajectories(model,[])
