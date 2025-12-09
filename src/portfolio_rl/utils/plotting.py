"""
Plotting utilities for portfolio RL experiments.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


def plot_equity_curves(
    curves: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "Equity Curves",
    x_label: str = "Time step",
    y_label: str = "Portfolio value",
) -> None:
    """
    Plot multiple equity curves on the same figure and save to disk.

    Args:
        curves: mapping from legend name to equity curve array (shape (T,))
        out_path: file path to save the figure (e.g. Path("reports/figs/equity_test.png"))
        title: plot title
        x_label: x-axis label
        y_label: y-axis label
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for name, eq in curves.items():
        eq = np.asarray(eq, dtype=float)
        plt.plot(eq, label=name)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
