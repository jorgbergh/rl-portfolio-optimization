"""
Gymnasium environment for continuous-weight portfolio optimization.
"""

from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PortfolioEnv(gym.Env):
    """
    Portfolio environment with continuous actions mapped to long-only weights.

    - Observation: feature vector + current weights.
    - Action: raw vector in R^N, mapped to weights via softmax.
    - Reward: log(1 + portfolio_return - transaction_cost * turnover).

    Args:
        returns: np.ndarray of shape (T, N_assets)
        features: np.ndarray of shape (T, obs_dim_raw)
        transaction_cost: per-unit turnover cost (e.g. 0.001)
        window_length: episode length in steps
        random_start: if True, start at random time; else start at t=0
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        returns: np.ndarray,
        features: np.ndarray,
        transaction_cost: float = 0.001,
        window_length: int = 252,
        random_start: bool = True,
    ):
        super().__init__()

        assert returns.shape[0] == features.shape[0], \
            "returns and features must have same length"

        self.returns = returns.astype(np.float32)
        self.features = features.astype(np.float32)

        self.T, self.n_assets = self.returns.shape
        self.obs_dim_raw = self.features.shape[1]
        self.transaction_cost = float(transaction_cost)
        self.window_length = int(window_length)
        self.random_start = bool(random_start)

        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )


        # Observation: [features, current_weights]
        obs_dim = self.obs_dim_raw + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Internal state
        self._t: int | None = None
        self._start_idx: int | None = None
        self._w_prev: np.ndarray | None = None
        self._portfolio_value: float | None = None

    # --------- helper methods --------- #

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Map raw action in R^N to a probability simplex via softmax.
        """
        a = np.asarray(action, dtype=np.float64)
        a = a - np.max(a)  # numerical stability
        e = np.exp(a)
        w = e / e.sum()
        return w.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Build current observation: [features_t, current_weights].
        """
        assert self._t is not None and self._w_prev is not None
        feat_t = self.features[self._t]
        obs = np.concatenate([feat_t, self._w_prev], axis=0)
        return obs.astype(np.float32)

    # --------- Gymnasium API --------- #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict | None = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.

        Returns:
            obs, info
        """
        super().reset(seed=seed)

        if self.random_start:
            max_start = max(self.T - self.window_length - 1, 0)
            start_idx = self.np_random.integers(0, max_start + 1)
        else:
            start_idx = 0

        self._start_idx = int(start_idx)
        self._t = self._start_idx

        # start equal-weighted
        self._w_prev = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._portfolio_value = 1.0

        obs = self._get_obs()
        info: Dict = {
            "portfolio_value": self._portfolio_value,
            "t": self._t,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in time.

        Returns:
            obs, reward, terminated, truncated, info
        """
        assert self._t is not None
        assert self._w_prev is not None
        assert self._portfolio_value is not None

        # map action -> weights
        w_t = self._action_to_weights(action)

        # portfolio return at time t
        R_t = self.returns[self._t]  # shape (N,)
        r_t = float(np.dot(w_t, R_t))

        # transaction cost
        turnover = float(np.sum(np.abs(w_t - self._w_prev)))
        cost_t = self.transaction_cost * turnover

        net_ret = r_t - cost_t
        net_ret_clipped = max(net_ret, -0.99)

        # update portfolio value
        self._portfolio_value *= (1.0 + net_ret_clipped)

        reward = float(np.log1p(net_ret_clipped))

        # advance time
        self._t += 1
        self._w_prev = w_t

        # termination logic
        terminated = False
        truncated = False

        if self._t >= self._start_idx + self.window_length:
            truncated = True
        if self._t >= self.T:
            truncated = True

        if not truncated and not terminated:
            obs = self._get_obs()
        else:
            # if episode is over, we still need to return a valid obs
            obs = self._get_obs()

        info = {
            "portfolio_value": self._portfolio_value,
            "raw_return": r_t,
            "net_return": net_ret,
            "turnover": turnover,
            "t": self._t,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Simple text rendering.
        """
        if self._t is None or self._portfolio_value is None or self._w_prev is None:
            print("Environment not initialized. Call reset() first.")
            return
        print(
            f"t={self._t}, "
            f"V={self._portfolio_value:.4f}, "
            f"w={np.round(self._w_prev, 3)}"
        )
