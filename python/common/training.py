"""Training loops (verbatim from the frozen ICLR scripts).

train_model_tf   — one-step teacher forcing (kan_hrpinn_paper.py / ablation scripts)
train_model_bptt — autoregressive BPTT over seq_len windows (kan_hrpinn_paper_bptt.py)
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .hrpinn import HRPINNCell


def train_model_tf(system, residual_network, train_loader, test_loader, normalizer, dt,
                   epochs=100, lr=1e-2, device='cpu', grid_update_freq=20,
                   reg_activation=0.0, reg_entropy=0.0, early_stop_patience=30,
                   early_stop_min_delta=1e-7, verbose=False):
    """Train the HRPINN model (one-step teacher forcing)."""

    hrpinn_cell = HRPINNCell(system, residual_network, normalizer, dt=dt)
    hrpinn_cell.to(device)

    optimizer = optim.Adam(residual_network.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0

    if verbose:
        print(f"    Training {system.name()}...")

    for epoch in range(epochs):
        # Training
        residual_network.train()
        train_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            x_curr, x_next_true = batch[0].to(device), batch[1].to(device)
            x_next_pred = hrpinn_cell(x_curr)

            data_loss = criterion(x_next_pred, x_next_true)

            # Regularization (only for KAN - MLP doesn't have regularization_loss method)
            if hasattr(residual_network, 'regularization_loss') and (reg_activation > 0 or reg_entropy > 0):
                reg_loss = residual_network.regularization_loss(reg_activation, reg_entropy)
                loss = data_loss + reg_loss
            else:
                loss = data_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_network.parameters(), 1.0)
            optimizer.step()

            train_loss += data_loss.item()
            batch_count += 1

        train_loss /= batch_count
        train_losses.append(train_loss)

        # Grid update (only for KAN)
        if grid_update_freq > 0 and (epoch + 1) % grid_update_freq == 0:
            if hasattr(residual_network, 'kan'):
                residual_network.eval()
                for batch in train_loader:
                    x_curr = batch[0].to(device)
                    residual_network.forward(x_curr, update_grid=True)
                    break
                residual_network.train()

        # Testing
        residual_network.eval()
        test_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in test_loader:
                x_curr, x_next_true = batch[0].to(device), batch[1].to(device)
                x_next_pred = hrpinn_cell(x_curr)
                loss = criterion(x_next_pred, x_next_true)
                test_loss += loss.item()
                batch_count += 1

        test_loss /= batch_count
        test_losses.append(test_loss)
        scheduler.step()

        # Early stopping
        if test_loss < best_test_loss - early_stop_min_delta:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break

    return residual_network, train_losses, test_losses


def train_model_bptt(system, residual_network, train_loader, test_loader, normalizer, dt,
                     epochs=100, lr=1e-2, device='cpu', grid_update_freq=20,
                     reg_activation=0.0, reg_entropy=0.0, early_stop_patience=30,
                     early_stop_min_delta=1e-7, verbose=False):
    """Train the HRPINN model with BPTT (Backpropagation Through Time)."""

    hrpinn_cell = HRPINNCell(system, residual_network, normalizer, dt=dt)
    hrpinn_cell.to(device)

    optimizer = optim.Adam(residual_network.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        residual_network.train()
        train_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            sequences = batch[0].to(device)  # [Batch, Seq_Len, Dim]
            batch_size, seq_len, state_dim = sequences.shape

            current_state = sequences[:, 0, :]
            total_loss = 0.0

            for t in range(1, seq_len):
                pred_state = hrpinn_cell(current_state)
                target_state = sequences[:, t, :]
                step_loss = criterion(pred_state, target_state)
                total_loss += step_loss
                current_state = pred_state

            data_loss = total_loss / (seq_len - 1)

            if hasattr(residual_network, 'regularization_loss') and (reg_activation > 0 or reg_entropy > 0):
                reg_loss = residual_network.regularization_loss(reg_activation, reg_entropy)
                loss = data_loss + reg_loss
            else:
                loss = data_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_network.parameters(), 1.0)
            optimizer.step()

            train_loss += data_loss.item()
            batch_count += 1

        train_loss /= batch_count
        train_losses.append(train_loss)

        # Grid update (only for KAN)
        if grid_update_freq > 0 and (epoch + 1) % grid_update_freq == 0:
            if hasattr(residual_network, 'kan'):
                residual_network.eval()
                for batch in train_loader:
                    x_sequences = batch[0].to(device)
                    x_flat = x_sequences.view(-1, x_sequences.size(-1))
                    residual_network.forward(x_flat, update_grid=True)
                    break
                residual_network.train()

        # Testing (also with BPTT to ensure consistency)
        residual_network.eval()
        test_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in test_loader:
                sequences = batch[0].to(device)
                batch_size, seq_len, state_dim = sequences.shape

                current_state = sequences[:, 0, :]
                total_loss = 0.0

                for t in range(1, seq_len):
                    pred_state = hrpinn_cell(current_state)
                    target_state = sequences[:, t, :]
                    step_loss = criterion(pred_state, target_state)
                    total_loss += step_loss
                    current_state = pred_state

                loss = total_loss / (seq_len - 1)
                test_loss += loss.item()
                batch_count += 1

        test_loss /= batch_count
        test_losses.append(test_loss)
        scheduler.step()

        # Early stopping
        if test_loss < best_test_loss - early_stop_min_delta:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            break

    return residual_network, train_losses, test_losses
