from __future__ import annotations

from rescue_env.core.environment import RescueEnvironment


def test_env_step_returns_episode_stats() -> None:
    env = RescueEnvironment(difficulty="easy")
    _ = env.reset(seed=42)

    action = {
        "action_type": "scan_thermal",
        "parameters": {"duration": 1.0},
    }
    observation, reward, done, truncated, info = env.step(action)

    assert hasattr(observation, "model_dump")
    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)

    assert "episode_stats" in info
    assert isinstance(info["episode_stats"], dict)
    assert "score_breakdown" in info
    assert isinstance(info["score_breakdown"], dict)


def test_env_state_available_after_reset() -> None:
    env = RescueEnvironment(difficulty="easy")
    env.reset(seed=7)
    state = env.state()

    assert state.difficulty == "easy"
    assert state.map_data.width > 0
    assert state.map_data.height > 0
