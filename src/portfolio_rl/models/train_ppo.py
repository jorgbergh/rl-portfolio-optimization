"""
Train a PPO agent on the portfolio environment.

Run from project root:
    python -m portfolio_rl.models.train_ppo
"""

from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from portfolio_rl.envs.portfolio_env import PortfolioEnv


def make_env_factory(
    returns: np.ndarray,
    features: np.ndarray,
    transaction_cost: float,
    window_length: int,
) -> Callable[[], PortfolioEnv]:
    """
    Create a factory function that returns a new PortfolioEnv instance.
    """

    def _make_env() -> PortfolioEnv:
        return PortfolioEnv(
            returns=returns,
            features=features,
            transaction_cost=transaction_cost,
            window_length=window_length,
            random_start=True,
        )

    return _make_env


def main() -> None:
    root = Path(".")
    processed_dir = root / "data" / "processed"
    experiments_dir = root / "experiments"
    models_dir = experiments_dir / "models"
    logs_dir = experiments_dir / "logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load processed training data
    returns_train = np.load(processed_dir / "returns_train.npy")
    features_train = np.load(processed_dir / "features_train.npy")

    # Basic env settings
    transaction_cost = 0.001
    window_length = 252  # ~1 trading year per episode

    env_factory = make_env_factory(
        returns=returns_train,
        features=features_train,
        transaction_cost=transaction_cost,
        window_length=window_length,
    )

    # Single-vectorized environment (can scale up if you want)
    vec_env = DummyVecEnv([env_factory])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=str(logs_dir),
    )

    total_timesteps = 200_000
    model_name = f"ppo_portfolio_{total_timesteps//1000}k"

    model.learn(total_timesteps=total_timesteps)
    model_path = models_dir / f"{model_name}.zip"
    model.save(model_path)

    print(f"Saved trained PPO model to {model_path}")


if __name__ == "__main__":
    main()
