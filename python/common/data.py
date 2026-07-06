"""Data generation + normalization (verbatim from the frozen ICLR scripts).

Two loaders exist because the ICLR paper used two training paradigms:
  - generate_training_data_tf:  (state, next_state) pairs, one-step teacher forcing
    (from kan_hrpinn_paper.py / *_parameter_ablation.py / *_config_ablation.py)
  - generate_training_data_bptt: sliding-window sequences, seq_len=20
    (from kan_hrpinn_paper_bptt.py)
"""

import numpy as np
import torch
from scipy.integrate import odeint
from torch.utils.data import TensorDataset, DataLoader


class DataNormalizer:
    """Normalize dataset to [-1, 1] range."""

    def __init__(self, data):
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)
        self.range = self.max_vals - self.min_vals
        self.range[self.range < 1e-8] = 1.0

    def normalize(self, data):
        return 2 * (data - self.min_vals) / self.range - 1

    def denormalize(self, data):
        return (data + 1) * self.range / 2 + self.min_vals


def _generate_trajectories(system, t_span, dt, num_trajectories, seed):
    np.random.seed(seed)
    all_trajectories = []
    for _ in range(num_trajectories):
        x0 = np.random.uniform(-2, 2)
        v0 = np.random.uniform(-2, 2)
        t = np.arange(t_span[0], t_span[1] + dt, dt)
        trajectory = odeint(system.full_dynamics, [x0, v0], t)
        all_trajectories.append(trajectory)
    return np.vstack(all_trajectories)


def generate_training_data_tf(system, t_span=(0, 20), dt=0.05, num_trajectories=8,
                              seed=42, device='cpu'):
    """One-step teacher-forcing pairs. Verbatim from kan_hrpinn_paper.py."""
    full_trajectory = _generate_trajectories(system, t_span, dt, num_trajectories, seed)
    normalizer = DataNormalizer(full_trajectory)
    normalized_trajectory = normalizer.normalize(full_trajectory)
    X = torch.from_numpy(normalized_trajectory).float().to(device)

    n_total = X.shape[0]
    n_train = int(n_total * 0.8)

    X_train = X[:n_train]
    X_train_curr = X_train[:-1]
    X_train_next = X_train[1:]

    X_test = X[n_train:]
    X_test_curr = X_test[:-1]
    X_test_next = X_test[1:]

    train_loader = DataLoader(
        TensorDataset(X_train_curr, X_train_next),
        batch_size=32,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_curr, X_test_next),
        batch_size=32,
        shuffle=False
    )

    return train_loader, test_loader, normalizer, dt


def create_sequences(data, seq_len):
    """Sliding-window sequences. Verbatim from kan_hrpinn_paper_bptt.py."""
    n_timesteps, n_dims = data.shape
    sequences = []
    for i in range(n_timesteps - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return torch.stack(sequences)


def generate_training_data_bptt(system, t_span=(0, 20), dt=0.05, num_trajectories=8,
                                seed=42, device='cpu', seq_len=20):
    """Sliding-window sequences for BPTT. Verbatim from kan_hrpinn_paper_bptt.py."""
    full_trajectory = _generate_trajectories(system, t_span, dt, num_trajectories, seed)
    normalizer = DataNormalizer(full_trajectory)
    normalized_trajectory = normalizer.normalize(full_trajectory)
    X = torch.from_numpy(normalized_trajectory).float().to(device)

    n_total = X.shape[0]
    n_train = int(n_total * 0.8)

    X_train = X[:n_train]
    X_test = X[n_train:]

    X_train_sequences = create_sequences(X_train, seq_len)
    X_test_sequences = create_sequences(X_test, seq_len)

    train_loader = DataLoader(
        TensorDataset(X_train_sequences),
        batch_size=32,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_sequences),
        batch_size=32,
        shuffle=False
    )

    return train_loader, test_loader, normalizer, dt


def training_states(system, t_span=(0, 20), dt=0.05, num_trajectories=8, seed=42):
    """All normalized training-split states (for PTQ calibration).

    Reproduces the exact train split used by both loaders above.
    """
    full_trajectory = _generate_trajectories(system, t_span, dt, num_trajectories, seed)
    normalizer = DataNormalizer(full_trajectory)
    normalized = normalizer.normalize(full_trajectory).astype(np.float32)
    n_train = int(normalized.shape[0] * 0.8)
    return normalized[:n_train], normalizer
