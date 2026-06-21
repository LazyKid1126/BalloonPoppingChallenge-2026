"""Train optional RL selector or residual policies for Scenario1GncAgent.

This is an exploration scaffold. The official submission path remains the
model-based agent; trained models are loaded only when their paths are supplied
to Scenario1GncAgent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from BalloonPoppingGymEnv.agents.gnc_agent import Scenario1GncAgent
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class GncRlEnv(gym.Env):
    """Gym wrapper around the model-based GNC agent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        mode: str,
        scenario: int = 1,
        seed_base: int = 0,
        max_episode_steps: int | None = None,
        agent_kwargs: dict | None = None,
    ):
        super().__init__()
        if mode not in ("selector", "residual"):
            raise ValueError("mode must be 'selector' or 'residual'")

        self.mode = mode
        self.scenario = int(scenario)
        self.seed_base = int(seed_base)
        self.max_episode_steps = max_episode_steps
        self.agent_kwargs = dict(agent_kwargs or {})
        self.top_k = int(self.agent_kwargs.get("top_k", 16))
        self.max_residual_deg = float(self.agent_kwargs.get("max_residual_deg", 3.0))

        if self.mode == "selector":
            self.action_space = spaces.Discrete(self.top_k)
            self.observation_space = spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(self.top_k * 8,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=-self.max_residual_deg,
                high=self.max_residual_deg,
                shape=(2,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(16,),
                dtype=np.float32,
            )

        self.env = None
        self.agent = None
        self.observation = None
        self.info = None
        self.steps = 0
        self.episode_index = 0
        self.last_popped = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = self.seed_base + self.episode_index if seed is None else int(seed)
        self.episode_index += 1

        scenario_parameters, given_parameters = load_scenario_parameters(self.scenario)
        scenario_parameters["scenario"]["random_seed"] = episode_seed
        self.env = BalloonPoppingEnv(render_mode=None, parameters=scenario_parameters)
        self.agent = Scenario1GncAgent(given_parameters, **self.agent_kwargs)
        self.observation, self.info = self.env.reset(seed=episode_seed)
        self.steps = 0
        self.last_popped = 0
        return self._policy_observation(), {}

    def step(self, action):
        if self.env is None or self.agent is None or self.observation is None:
            raise RuntimeError("Call reset before step")

        if self.mode == "selector":
            self.agent.external_selector_action = int(action)
        else:
            self.agent.external_residual_action = np.asarray(action, dtype=float)

        env_action = self.agent.get_action(self.observation)
        self.observation, reward, terminated, truncated, self.info = self.env.step(
            env_action
        )
        self.steps += 1
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True

        popped = int(self.info.get("popped_count", 0))
        self.last_popped = popped
        shaped_reward = float(reward)

        info = dict(self.info)
        info.update(self.agent.get_debug_stats())
        return self._policy_observation(), shaped_reward, terminated, truncated, info

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = None

    def _policy_observation(self):
        if self.agent is None or self.observation is None:
            if self.mode == "selector":
                return np.zeros(self.top_k * 8, dtype=np.float32)
            return np.zeros(16, dtype=np.float32)

        if self.mode == "selector":
            obs = self.agent.selector_observation(self.observation).reshape(-1)
        else:
            obs = self.agent.residual_observation(self.observation)
        return np.asarray(obs, dtype=np.float32)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("selector", "residual"), required=True)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument("--planner-mode", default="cluster_predictive")
    parser.add_argument("--pn-gain", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-residual-deg", type=float, default=3.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed. Install optional RL dependencies "
            "with: uv pip install -r requirements-rl.txt"
        ) from exc

    agent_kwargs = {
        "planner_mode": args.planner_mode,
        "pn_gain": args.pn_gain,
        "top_k": args.top_k,
        "max_residual_deg": args.max_residual_deg,
    }
    env = GncRlEnv(
        mode=args.mode,
        scenario=args.scenario,
        seed_base=args.seed_base,
        max_episode_steps=args.max_episode_steps,
        agent_kwargs=agent_kwargs,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        tensorboard_log=args.tensorboard_log,
    )
    model.learn(total_timesteps=args.timesteps)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Saved {args.mode} model to {output}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
