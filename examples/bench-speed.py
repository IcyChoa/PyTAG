"""PyTAG step-latency microbenchmark.

Measures Gym env.step() throughput after JVM warmup for one or more TAG games.
For single-player envs like TAG/Saboteur-v0 (python + random opponents), each
step already includes opponent turns until control returns to the Python agent.

Rough interpretation (single env, masked random):
  > ~1000 SPS  — fine for single-env PPO prototyping
  ~100–1000 SPS — usable; consider vectorizing later for large runs
  < ~100 SPS   — bridge/game is a real training bottleneck

Optional Java-only baseline (no Jpype), from TabletopGames/:
  java -cp target/TAG.jar evaluation.SpeedBenchmark
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _percentile(sorted_arr: np.ndarray, q: float) -> float:
    """Linear percentile on a pre-sorted array (q in [0, 100])."""
    if sorted_arr.size == 0:
        return float("nan")
    return float(np.percentile(sorted_arr, q))


def benchmark_env(game_id: str, n_steps: int, warmup: int) -> dict:
    import gymnasium as gym

    env = gym.make(game_id)
    obs, info = env.reset()

    # Warmup (not timed)
    for _ in range(warmup):
        action = env.unwrapped.sample_rnd_action()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    times: list[float] = []
    episodes = 0
    episode_lengths: list[int] = []
    ep_len = 0
    action_n = int(getattr(env.action_space, "n", 0) or 0)
    obs_shape = tuple(getattr(obs, "shape", ()))

    for _ in range(n_steps):
        t0 = time.perf_counter()
        action = env.unwrapped.sample_rnd_action()
        obs, reward, terminated, truncated, info = env.step(action)
        times.append(time.perf_counter() - t0)
        ep_len += 1
        if terminated or truncated:
            episodes += 1
            episode_lengths.append(ep_len)
            ep_len = 0
            obs, info = env.reset()

    env.close()

    arr = np.asarray(times, dtype=np.float64)
    total_s = float(arr.sum())
    ms = arr * 1000.0
    ms_sorted = np.sort(ms)

    return {
        "game_id": game_id,
        "steps": n_steps,
        "warmup": warmup,
        "total_s": total_s,
        "sps": n_steps / total_s if total_s > 0 else float("nan"),
        "ms_mean": float(ms.mean()),
        "ms_p50": _percentile(ms_sorted, 50),
        "ms_p95": _percentile(ms_sorted, 95),
        "episodes": episodes,
        "mean_ep_len": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        "obs_shape": obs_shape,
        "action_n": action_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTAG Gym step latency / SPS")
    parser.add_argument(
        "--games",
        type=str,
        default="TAG/Saboteur-v0,TAG/TicTacToe-v0",
        help="Comma-separated Gym env ids",
    )
    parser.add_argument("--steps", type=int, default=5_000, help="Timed steps per game")
    parser.add_argument("--warmup", type=int, default=200, help="Untimed warmup steps per game")
    args = parser.parse_args()

    game_ids = [g.strip() for g in args.games.split(",") if g.strip()]

    # Cold start: first import + make + reset (JVM + jar)
    t_cold0 = time.perf_counter()
    import pytag.gym_wrapper  # noqa: F401 — registers TAG/* envs
    import gymnasium as gym

    cold_env = gym.make(game_ids[0])
    cold_env.reset()
    cold_env.close()
    cold_s = time.perf_counter() - t_cold0
    print(f"Cold start (import + make + reset {game_ids[0]}): {cold_s * 1000:.1f} ms")
    print()

    results = []
    for game_id in game_ids:
        print(f"Benchmarking {game_id} (warmup={args.warmup}, steps={args.steps})...")
        r = benchmark_env(game_id, args.steps, args.warmup)
        results.append(r)
        print(
            f"  SPS={r['sps']:.1f}  "
            f"ms/step mean={r['ms_mean']:.3f}  p50={r['ms_p50']:.3f}  p95={r['ms_p95']:.3f}  "
            f"episodes={r['episodes']}  mean_ep_len={r['mean_ep_len']:.1f}  "
            f"obs={r['obs_shape']}  action_n={r['action_n']}"
        )
        print()

    print("=" * 60)
    print(f"{'game':<22} {'SPS':>10} {'ms_mean':>10} {'ms_p50':>10} {'ms_p95':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['game_id']:<22} {r['sps']:10.1f} {r['ms_mean']:10.3f} "
            f"{r['ms_p50']:10.3f} {r['ms_p95']:10.3f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
