"""
Hyperparameter tuning for A2C portfolio agent using classical k-fold CV.

Run from project root:
    python -m portfolio_rl.models.tune_hyperparams_a2c

Logic:
  - Keep the test set completely untouched.
  - Concatenate TRAIN and VAL into one "training pool".
  - Split this pool into k contiguous folds (no shuffling).
  - For each hyperparameter config:
        For each fold:
            Train on all folds except this one.
            Validate on this held-out fold.
        Score = mean validation Sharpe across folds.
  - Print and save ranked results.

This script does NOT load or use the test set at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from portfolio_rl.envs.portfolio_env import PortfolioEnv
from portfolio_rl.utils.metrics import sharpe_ratio


# ----------------- helpers ----------------- #


@dataclass
class HyperParams:
    learning_rate: float
    gamma: float
    gae_lambda: float
    transaction_cost: float
    total_timesteps: int
    window_length: int

    def short_name(self) -> str:
        return (
            f"lr={self.learning_rate}_g={self.gamma}_"
            f"lam={self.gae_lambda}_tc={self.transaction_cost}_"
            f"steps={self.total_timesteps}_win={self.window_length}"
        )


def generate_hyperparam_grid() -> List[HyperParams]:
    """
    Define a grid of hyperparameters to search over for A2C.
    Adjust or extend as you like.
    """
    learning_rates = [1e-3, 3e-4, 1e-4]
    gammas = [0.90, 0.95, 0.99]
    gae_lambdas = [0.85, 0.90, 0.95]
    transaction_costs = [0.0005, 0.001]
    total_timesteps_list = [200_000, 400_000]
    window_lengths = [126, 252]

    configs: List[HyperParams] = []
    for (lr, g, lam, tc, steps, win) in product(
        learning_rates,
        gammas,
        gae_lambdas,
        transaction_costs,
        total_timesteps_list,
        window_lengths,
    ):
        configs.append(
            HyperParams(
                learning_rate=lr,
                gamma=g,
                gae_lambda=lam,
                transaction_cost=tc,
                total_timesteps=steps,
                window_length=win,
            )
        )
    return configs


def make_kfold_splits(
    returns_all: np.ndarray,
    features_all: np.ndarray,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Classical k-fold for time series (no shuffling).

    Splits the training pool into k contiguous chunks:
        Fold 1 | Fold 2 | ... | Fold k

    Returns a list of tuples per fold:
        (returns_train_fold, features_train_fold,
         returns_val_fold,   features_val_fold)
    """
    T = returns_all.shape[0]
    fold_size = T // n_folds
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end = T if fold_idx == n_folds - 1 else (fold_idx + 1) * fold_size

        # validation segment
        r_val = returns_all[val_start:val_end]
        f_val = features_all[val_start:val_end]

        # training = everything except this segment
        r_train = np.concatenate(
            [returns_all[:val_start], returns_all[val_end:]], axis=0
        )
        f_train = np.concatenate(
            [features_all[:val_start], features_all[val_end:]], axis=0
        )

        splits.append((r_train, f_train, r_val, f_val))

    return splits


def make_train_env_factory(
    returns: np.ndarray,
    features: np.ndarray,
    transaction_cost: float,
    window_length: int,
    seed: int,
) -> Callable[[], PortfolioEnv]:
    """
    Factory that creates a new PortfolioEnv instance for training.
    """

    def _make_env() -> PortfolioEnv:
        env = PortfolioEnv(
            returns=returns,
            features=features,
            transaction_cost=transaction_cost,
            window_length=window_length,
            random_start=True,
        )
        env.reset(seed=seed)
        return env

    return _make_env


def run_policy_once(
    env: PortfolioEnv,
    model: A2C,
    deterministic: bool = True,
) -> np.ndarray:
    """
    Roll out a trained policy on an environment and return daily net returns.
    """
    obs, _ = env.reset()
    done = False
    truncated = False
    rets: List[float] = []

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        rets.append(info["net_return"])

    return np.array(rets, dtype=np.float32)


def evaluate_config_kfold(
    hp: HyperParams,
    kfold_splits: List[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    base_seed: int = 0,
) -> float:
    """
    For one hyperparameter config:
      - run k-fold CV on the training pool using A2C
      - return mean validation Sharpe across folds
    """
    print(f"\n=== Evaluating A2C config: {hp.short_name()} ===")

    fold_sharpes: List[float] = []

    for fold_idx, (r_train, f_train, r_val, f_val) in enumerate(
        kfold_splits
    ):
        seed = base_seed + fold_idx
        set_random_seed(seed)

        # Training env for this fold
        env_factory = make_train_env_factory(
            returns=r_train,
            features=f_train,
            transaction_cost=hp.transaction_cost,
            window_length=hp.window_length,
            seed=seed,
        )
        vec_env = DummyVecEnv([env_factory])

        # A2C model
        model = A2C(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=hp.learning_rate,
            gamma=hp.gamma,
            gae_lambda=hp.gae_lambda,
            n_steps=5,              # can tune separately if you want
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            use_rms_prop=True,
            normalize_advantage=True,
            verbose=0,
        )

        # Train
        model.learn(total_timesteps=hp.total_timesteps)

        # Validation on this fold
        env_val = PortfolioEnv(
            returns=r_val,
            features=f_val,
            transaction_cost=hp.transaction_cost,
            window_length=len(r_val) - 2,
            random_start=False,
        )
        net_rets = run_policy_once(env_val, model)
        s = sharpe_ratio(net_rets)
        fold_sharpes.append(s)
        print(f"  Fold {fold_idx + 1}: Sharpe = {s:.4f}")

    mean_sharpe = float(np.mean(fold_sharpes))
    print(f"  Mean CV Sharpe (A2C): {mean_sharpe:.4f}")
    return mean_sharpe


# ----------------- main ----------------- #


def main() -> None:
    root = Path(".")
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load original train and val splits
    returns_train = np.load(processed_dir / "returns_train.npy")
    features_train = np.load(processed_dir / "features_train.npy")
    returns_val = np.load(processed_dir / "returns_val.npy")
    features_val = np.load(processed_dir / "features_val.npy")

    # Create the full "training pool" by concatenating train + val
    returns_all = np.concatenate([returns_train, returns_val], axis=0)
    features_all = np.concatenate([features_train, features_val], axis=0)

    # Build k-fold splits (on training pool only)
    k = 3  # change to 5 if you want more folds
    kfold_splits = make_kfold_splits(returns_all, features_all, n_folds=k)

    # Hyperparameter grid
    grid = generate_hyperparam_grid()

    # Evaluate each config with k-fold CV
    results: List[Tuple[HyperParams, float]] = []
    for hp in grid:
        mean_sharpe = evaluate_config_kfold(
            hp=hp,
            kfold_splits=kfold_splits,
            base_seed=0,
        )
        results.append((hp, mean_sharpe))

    # Rank configs by mean CV Sharpe (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n=========== A2C K-fold Hyperparameter Results ===========")
    for rank, (hp, score) in enumerate(results, start=1):
        print(f"{rank:>2}. Sharpe={score: .4f} | {hp.short_name()}")

    # Save to CSV
    out_csv = reports_dir / "hyperparam_kfold_results_a2c.csv"
    with out_csv.open("w") as f:
        f.write(
            "rank,mean_sharpe,learning_rate,gamma,gae_lambda,"
            "transaction_cost,total_timesteps,window_length\n"
        )
        for rank, (hp, score) in enumerate(results, start=1):
            f.write(
                f"{rank},{score},"
                f"{hp.learning_rate},{hp.gamma},{hp.gae_lambda},"
                f"{hp.transaction_cost},{hp.total_timesteps},{hp.window_length}\n"
            )

    print(f"\nSaved A2C k-fold tuning results to {out_csv}")


if __name__ == "__main__":
    main()
