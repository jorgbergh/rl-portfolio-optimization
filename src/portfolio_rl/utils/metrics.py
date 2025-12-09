"""
Utility functions for portfolio performance metrics.
"""

from typing import Tuple

import numpy as np


def equity_curve(daily_returns: np.ndarray, initial_value: float = 1.0) -> np.ndarray:
    """
    Compute equity curve (portfolio value over time) from daily returns.

    Args:
        daily_returns: shape (T,) array of simple net returns.
        initial_value: starting portfolio value.

    Returns:
        equity: shape (T,) array of portfolio values.
    """
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    equity = np.empty_like(daily_returns)
    value = initial_value
    for i, r in enumerate(daily_returns):
        value *= (1.0 + r)
        equity[i] = value
    return equity


def annualized_return(daily_returns: np.ndarray, freq: int = 252) -> float:
    """
    Compute annualized return from daily returns.
    """
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    avg_daily = daily_returns.mean()
    return float((1.0 + avg_daily) ** freq - 1.0)


def annualized_volatility(daily_returns: np.ndarray, freq: int = 252) -> float:
    """
    Compute annualized volatility from daily returns.
    """
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    return float(daily_returns.std(ddof=1) * np.sqrt(freq))


def sharpe_ratio(
    daily_returns: np.ndarray,
    freq: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute Sharpe ratio using daily returns.

    risk_free_rate is annual; converted to daily internally.
    """
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    r_annual = annualized_return(daily_returns, freq=freq)
    vol_annual = annualized_volatility(daily_returns, freq=freq)
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / freq) - 1.0
    excess_annual = (1.0 + r_annual) / (1.0 + rf_daily * freq) - 1.0

    if vol_annual == 0.0:
        return float("nan")

    return float(excess_annual / vol_annual)


def max_drawdown(equity: np.ndarray) -> Tuple[float, int, int]:
    """
    Compute maximum drawdown and its start/end indices.

    Returns:
        max_dd: maximum drawdown (negative number)
        peak_idx: index of peak
        trough_idx: index of trough
    """
    equity = np.asarray(equity, dtype=np.float64)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0

    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(equity[: trough_idx + 1]))
    max_dd = float(drawdowns[trough_idx])
    return max_dd, peak_idx, trough_idx


def average_turnover(turnovers: np.ndarray) -> float:
    """
    Compute average turnover per day.

    turnovers: array of daily turnover values.
    """
    turnovers = np.asarray(turnovers, dtype=np.float64)
    if turnovers.size == 0:
        return 0.0
    return float(turnovers.mean())
