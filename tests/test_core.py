from src.core import rollout

def test_rollout():
    assert rollout(0, [1, 0], 0.5) == [0, 1, 0.5]
