def rollout(state: float, actions: list[float], damping: float = 0.9) -> list[float]:
    """Simple deterministic latent-state baseline, not a learned world model."""
    states = [state]
    for action in actions:
        state = damping * state + action
        states.append(state)
    return states
