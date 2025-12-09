"""
Evaluate a trained PPO portfolio agent against baselines using ONLY the validation set.

Run from project root:
    python -m portfolio_rl.models.evaluate
"""

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import A2C

from portfolio_rl.envs.portfolio_env import PortfolioEnv
from portfolio_rl.models.baselines import (
    buy_and_hold_single,
    equal_weight,
    summarize_performance,
)
from portfolio_rl.utils.metrics import equity_curve
from portfolio_rl.utils.plotting import plot_equity_curves


def run_policy_once(
    env: PortfolioEnv,
    model: PPO,
    deterministic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out a trained policy on a given environment.

    Returns:
        daily_net_returns: shape (T_eval,)
        portfolio_values: shape (T_eval,)
        turnovers: shape (T_eval,)
    """
    obs, _ = env.reset()
    done = False
    truncated = False

    net_returns: list[float] = []
    values: list[float] = []
    turnovers: list[float] = []

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)

        net_returns.append(info["net_return"])
        values.append(info["portfolio_value"])
        turnovers.append(info["turnover"])

    return (
        np.array(net_returns, dtype=np.float32),
        np.array(values, dtype=np.float32),
        np.array(turnovers, dtype=np.float32),
    )


def safe_print_summary(title: str, summary: dict) -> None:
    """
    Nicely print a summary dict with mixed float/string values.
    """
    print(f"\n=== {title} ===")
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            print(f"{k:>15}: {v:.4f}")
        else:
            print(f"{k:>15}: {v}")


def main() -> None:
    root = Path(".")
    processed_dir = root / "data" / "processed"
    models_dir = root / "experiments" / "models"
    reports_dir = root / "reports"
    figs_dir = reports_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ONLY validation data ---- #
    returns_val = np.load(processed_dir / "returns_val.npy")
    features_val = np.load(processed_dir / "features_val.npy")

    # ---- Load trained model ---- #
    model_path = models_dir / "a2c_portfolio_200k.zip"  # "a2c_portfolio_200k.zip"
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    # ---- RL on validation set ---- #
    env_val = PortfolioEnv(
        returns=returns_val,
        features=features_val,
        transaction_cost=0.001,
        window_length=len(returns_val) - 2,
        random_start=False,
    )
    rl_ret_val, rl_val_curve, rl_turn_val = run_policy_once(env_val, model)
    rl_summary_val: Dict[str, float] = summarize_performance(
        rl_ret_val, turnovers=rl_turn_val, name="RL (val)"
    )

    # ---- Baselines on validation set ---- #
    bh_ret_val = buy_and_hold_single(returns_val, asset_index=0)
    bh_curve_val = equity_curve(bh_ret_val)
    bh_summary_val = summarize_performance(
        bh_ret_val, name="Buy&Hold SPY (val)")

    ew_ret_val = equal_weight(returns_val)
    ew_curve_val = equity_curve(ew_ret_val)
    ew_summary_val = summarize_performance(
        ew_ret_val, name="EqualWeight (val)")

    # ---- Print summaries ---- #
    safe_print_summary("Validation (RL)", rl_summary_val)
    safe_print_summary("Validation (Buy&Hold SPY)", bh_summary_val)
    safe_print_summary("Validation (EqualWeight)", ew_summary_val)

    # ---- Save equity curve plot for validation set ---- #
    curves_val = {
        "RL (val)": rl_val_curve,
        "Buy&Hold SPY (val)": bh_curve_val,
        "EqualWeight (val)": ew_curve_val,
    }
    out_path = figs_dir / "equity_curves_val.png"
    plot_equity_curves(
        curves=curves_val,
        out_path=out_path,
        title="Validation Equity Curves: RL vs Baselines",
        x_label="Time step",
        y_label="Portfolio value",
    )
    print(f"\nSaved validation equity curve plot to {out_path}")


if __name__ == "__main__":
    main()
