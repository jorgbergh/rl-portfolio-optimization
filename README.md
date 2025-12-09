# RL Portfolio Optimization

Portfolio allocation using reinforcement learning algorithms. This project trains RL agents (PPO and A2C) to optimize investment portfolios by learning to allocate capital across ETFs based on market features.

## Project Overview

This project tackles the portfolio optimization problem using deep reinforcement learning. Instead of relying on traditional financial models, we train agents to learn optimal trading strategies from historical data. The agents allocate capital across multiple ETFs while managing transaction costs.

## Features

- **Custom Gym Environment**: A portfolio environment that simulates trading with continuous actions mapped to asset weights.
- **RL Algorithms**: Implementations of PPO (Proximal Policy Optimization) and A2C (Advantage Actor-Critic).
- **Data Pipeline**: Automatic ETF data download and feature engineering.
- **Hyperparameter Tuning**: Tools to optimize agent hyperparameters.
- **Evaluation & Metrics**: Standard financial metrics and performance visualization.

## Quick Start

### Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### Download Data

Fetch historical price data for ETFs:

```bash
python -m portfolio_rl.data.download_data
```

### Prepare Data

Process raw data into train/validation/test sets with engineered features:

```bash
python -m portfolio_rl.data.make_dataset
```

### Train a Model

Train a PPO agent:

```bash
python -m portfolio_rl.models.train_ppo
```

Or train an A2C agent:

```bash
python -m portfolio_rl.models.train_a2c
```

### Evaluate

Evaluate the trained model on test data:

```bash
python -m portfolio_rl.models.evaluate
```

### Hyperparameter Tuning

Find better hyperparameters:

```bash
python -m portfolio_rl.models.tune_hyperparams
```

## Project Structure

```
src/portfolio_rl/
├── data/              # Data loading and preprocessing
├── envs/              # Gym environment for portfolio trading
├── models/            # Training and evaluation scripts
└── utils/             # Metrics and plotting utilities
```

## Key Files

- `envs/portfolio_env.py`: Custom Gym environment implementing portfolio trading logic.
- `models/train_ppo.py`: PPO training script.
- `models/train_a2c.py`: A2C training script.
- `data/make_dataset.py`: Feature engineering and data splitting.

## Requirements

See `requirements.txt` for all dependencies. Main packages:

- `gymnasium`: RL environment framework
- `stable-baselines3`: RL algorithms (PPO, A2C)
- `numpy`, `pandas`: Data processing
- `torch`: Deep learning
- `yfinance`: ETF data download
- `tensorboard`: Training visualization

## Notes

- All actions are from the root directory using Python module syntax.
- Models are saved in `experiments/models/`.
- Training logs are stored in `experiments/logs/` for TensorBoard visualization.
- Transaction costs are included in the reward function.