"""Smoke-test Phase A: one-round Saboteur episodes with team ±1 reward."""
import statistics

import gymnasium as gym
import pytag.gym_wrapper  # noqa: F401 — register envs


def main():
    env = gym.make("TAG/Saboteur-v0")
    lengths = []
    terminal_rewards = []
    role_checks = []

    for ep in range(20):
        obs, info = env.reset(seed=ep)
        assert info.get("team_reward", 0.0) == 0.0
        assert "role" in info
        done = False
        steps = 0
        reward = 0.0
        while not done:
            action = env.unwrapped.sample_rnd_action()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if not done:
                assert reward == 0.0, f"mid-episode reward should be 0, got {reward}"
            if steps > 500:
                raise RuntimeError("episode exceeded 500 steps — singleRoundEpisode may be off")

        lengths.append(steps)
        terminal_rewards.append(reward)
        assert reward in (-1.0, 1.0), f"terminal reward must be ±1, got {reward}"
        assert info["team_reward"] == reward
        assert info["has_won"] == int(reward)
        assert info["role"] in ("miner", "saboteur")

        # Consistency: miner win => miner +1 / saboteur -1
        miners_won = info["miners_won"]
        is_saboteur = info["is_saboteur"]
        expected = (-1.0 if miners_won else 1.0) if is_saboteur else (1.0 if miners_won else -1.0)
        assert reward == expected, (
            f"reward mismatch: role={info['role']} miners_won={miners_won} reward={reward}"
        )
        role_checks.append((info["role"], miners_won, reward))

    print(f"episodes={len(lengths)}")
    print(f"length mean={statistics.mean(lengths):.1f} min={min(lengths)} max={max(lengths)}")
    print(f"rewards={terminal_rewards}")
    print(f"wins(+1)={sum(1 for r in terminal_rewards if r > 0)} "
          f"losses(-1)={sum(1 for r in terminal_rewards if r < 0)}")
    print("Phase A smoke test OK")


if __name__ == "__main__":
    main()
