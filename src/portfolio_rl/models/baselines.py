"""
Baseline portfolio strategies for comparison with RL.
"""

from typing import Dict

import numpy as np

from portfolio_rl.utils.metrics import (
    equity_curve,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
)


def buy_and_hold_single(
    returns: np.ndarray,
    asset_index: int = 0,
) -> np.ndarray:
    """
    Buy-and-hold a single asset.

    Args:
        returns: array of shape (T, N_assets)
        asset_index: which column to hold.

    Returns:
        daily_returns: array of shape (T,)
    """
    returns = np.asarray(returns, dtype=np.float32)
    return returns[:, asset_index]


def equal_weight(
    returns: np.ndarray,
) -> np.ndarray:
    """
    Equal-weighted portfolio rebalanced daily.

    Args:
        returns: array of shape (T, N_assets)

    Returns:
        daily_returns: array of shape (T,)
    """
    returns = np.asarray(returns, dtype=np.float32)
    n_assets = returns.shape[1]
    weights = np.ones(n_assets, dtype=np.float32) / n_assets
    daily_returns = (returns @ weights).astype(np.float32)
    return daily_returns


def summarize_performance(
    daily_returns: np.ndarray,
    turnovers: np.ndarray | None = None,
    name: str | None = None,
) -> Dict[str, float]:
    """
    Compute a dictionary of performance metrics for convenience.
    """
    eq = equity_curve(daily_returns)
    r_annual = annualized_return(daily_returns)
    vol_annual = annualized_volatility(daily_returns)
    sharpe = sharpe_ratio(daily_returns)
    max_dd, _, _ = max_drawdown(eq)

    summary: Dict[str, float] = {
        "annual_return": r_annual,
        "annual_vol": vol_annual,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }

    if turnovers is not None and len(turnovers) > 0:
        summary["avg_turnover"] = float(np.mean(turnovers))

    if name is not None:
        summary["name"] = name

    return summary
