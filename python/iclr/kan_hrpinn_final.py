"""
Rigorous KAN-HRPINN Implementation with B-Spline KAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import warnings
warnings.filterwarnings('ignore')

from kan import KAN
# ============================================================================
# 2. SYSTEM PHYSICS DEFINITIONS
# ============================================================================

class DuffingOscillator:
    """
    Duffing: ẋ = v, v̇ = -x - 0.3x³ - 0.2v
    Known: v̇ = -x - 0.2v
    Residual: v̇ += -0.3x³
    """
    
    @staticmethod
    def full_dynamics(state, t):
        x, v = state
        dx = v
        dv = -x - 0.3*x**3 - 0.2*v
        return [dx, dv]
    
    @staticmethod
    def known_acceleration(state):
        """Known physics: -x - 0.2v"""
        if isinstance(state, np.ndarray):
            x, v = state[0], state[1]
            return -x - 0.2*v
        else:
            x, v = state[:, 0:1], state[:, 1:2]
            return -x - 0.2*v
    
    @staticmethod
    def residual_true(state):
        """True residual: -0.3x³"""
        if isinstance(state, np.ndarray):
            x = state[0]
            return -0.3 * x**3
        else:
            x = state[:, 0:1]
            return -0.3 * x**3
    
    @staticmethod
    def name():
        return "Duffing"


class VanDerPolOscillator:
    """
    Van der Pol: ẋ = v, v̇ = -x + (1-x²)v
    Known: v̇ = -x
    Residual: v̇ += (1-x²)v
    """
    
    @staticmethod
    def full_dynamics(state, t):
        x, v = state
        dx = v
        dv = -x + (1 - x**2)*v
        return [dx, dv]
    
    @staticmethod
    def known_acceleration(state):
        """Known physics: -x"""
        if isinstance(state, np.ndarray):
            x = state[0]
            return -x
        else:
            x = state[:, 0:1]
            return -x
    
    @staticmethod
    def residual_true(state):
        """True residual: (1-x²)v"""
        if isinstance(state, np.ndarray):
            x, v = state[0], state[1]
            return (1 - x**2) * v
        else:
            x, v = state[:, 0:1], state[:, 1:2]
            return (1 - x**2) * v
    
    @staticmethod
    def name():
        return "VdP"


# ============================================================================
# 3. DATA GENERATION WITH NORMALIZATION
# ============================================================================

class DataNormalizer:
    """Normalize dataset to [-1, 1] range."""
    
    def __init__(self, data):
        """
        Args:
            data: (N, 2) array of [x, v] states
        """
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)
        self.range = self.max_vals - self.min_vals
        self.range[self.range < 1e-8] = 1.0  # Avoid division by zero
    
    def normalize(self, data):
        """Normalize to [-1, 1]"""
        return 2 * (data - self.min_vals) / self.range - 1
    
    def denormalize(self, data):
        """Denormalize from [-1, 1]"""
        return (data + 1) * self.range / 2 + self.min_vals


def generate_training_data(system, t_span=(0, 20), dt=0.05, num_trajectories=8, device='cpu'):
    """
    Generate training data with consistent timestep.
    
    Returns:
        train_loader, test_loader, normalizer, dt
    """
    print(f"  Generating {system.name()} data...")
    
    all_trajectories = []
    np.random.seed(42)
    
    for i in range(num_trajectories):
        x0 = np.random.uniform(-2, 2)
        v0 = np.random.uniform(-2, 2)
        
        t = np.arange(t_span[0], t_span[1] + dt, dt)
        trajectory = odeint(system.full_dynamics, [x0, v0], t)
        all_trajectories.append(trajectory)
    
    full_trajectory = np.vstack(all_trajectories)
    
    # Create normalizer
    normalizer = DataNormalizer(full_trajectory)
    
    # Normalize data
    normalized_trajectory = normalizer.normalize(full_trajectory)
    X = torch.from_numpy(normalized_trajectory).float().to(device)
    
    n_total = X.shape[0]
    n_train = int(n_total * 0.8)
    
    # Create (current_state, next_state) pairs for training
    X_train = X[:n_train]
    X_train_curr = X_train[:-1]  # All states except last
    X_train_next = X_train[1:]    # All states except first
    
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
    
    print(f"    Training samples: {X_train_curr.shape[0]}, Test samples: {X_test_curr.shape[0]}")
    print(f"    Timestep: {dt:.4f} s")
    print(f"    Data range (normalized): [{X.min():.4f}, {X.max():.4f}]")
    
    return train_loader, test_loader, normalizer, dt


# ============================================================================
# 4. HRPINN CELL WITH HARD CONSTRAINTS
# ============================================================================

class HRPINNCell(nn.Module):
    """
    HRPINN cell with hard-constrained physics.
    
    Implements: x_next = x + dt * v
                v_next = v + dt * (known_acceleration + residual_network(state))
    """
    
    def __init__(self, system, residual_kan, normalizer, dt=0.05):
        super().__init__()
        self.system = system
        self.residual_kan = residual_kan
        self.normalizer = normalizer
        self.dt = dt
        
        # Register normalizer constants as buffers for device compatibility
        self.register_buffer('norm_min', torch.tensor(normalizer.min_vals, dtype=torch.float32))
        self.register_buffer('norm_range', torch.tensor(normalizer.range, dtype=torch.float32))
    
    def forward(self, state_normalized):
        """
        Forward pass with hard constraints.
        
        Args:
            state_normalized: (batch_size, 2) - normalized [x, v]
        
        Returns:
            next_state_normalized: (batch_size, 2)
        """
        # Denormalize for physics computation (keep gradients)
        min_vals = self.norm_min.to(state_normalized.device)
        range_vals = self.norm_range.to(state_normalized.device)
        state = (state_normalized + 1) * range_vals / 2 + min_vals
        
        # Known physics (hard-constrained) - use system method for consistency
        dv_known = self.system.known_acceleration(state)
        if not isinstance(dv_known, torch.Tensor):
            dv_known = torch.tensor(dv_known, dtype=state.dtype, device=state.device)
        if dv_known.dim() == 0:
            dv_known = dv_known.unsqueeze(0)
        if dv_known.dim() == 1:
            dv_known = dv_known.unsqueeze(1)
        
        # Residual (learned from normalized state)
        dv_residual = self.residual_kan(state_normalized)
        
        # Total acceleration
        dv_total = dv_known + dv_residual
        
        # Integration step
        x, v = state[:, 0:1], state[:, 1:2]
        x_next = x + self.dt * v
        v_next = v + self.dt * dv_total
        
        next_state = torch.cat([x_next, v_next], dim=1)
        
        # Normalize for next step
        next_state_normalized = 2 * (next_state - min_vals) / range_vals - 1
        
        return next_state_normalized


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

def train_model(system, residual_kan, train_loader, test_loader, normalizer, dt, 
                epochs=100, lr=1e-2, device='cpu', grid_update_freq=20,
                reg_activation=0.01, reg_entropy=0.01, early_stop_patience=20,
                early_stop_min_delta=1e-6):
    """
    Train the KAN-HRPINN model with grid updates, regularization, and early stopping.
    
    Args:
        system: System physics class
        residual_kan: KAN model to train
        train_loader: Training data loader
        test_loader: Test data loader
        normalizer: Data normalizer
        dt: Timestep
        epochs: Maximum number of epochs
        lr: Learning rate
        device: Device to train on
        grid_update_freq: Frequency of grid updates (every N epochs)
        reg_activation: Regularization weight for activation
        reg_entropy: Regularization weight for entropy
        early_stop_patience: Early stopping patience (epochs without improvement)
        early_stop_min_delta: Minimum change to qualify as improvement
    """
    
    hrpinn_cell = HRPINNCell(system, residual_kan, normalizer, dt=dt)
    hrpinn_cell.to(device)
    
    optimizer = optim.Adam(residual_kan.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    
    print(f"\n  Training KAN-HRPINN for {system.name()}...")
    print(f"    Grid update frequency: every {grid_update_freq} epochs")
    print(f"    Regularization: activation={reg_activation}, entropy={reg_entropy}")
    print(f"    Early stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    
    for epoch in range(epochs):
        # Training
        residual_kan.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            x_curr, x_next_true = batch[0].to(device), batch[1].to(device)
            
            # Model prediction
            x_next_pred = hrpinn_cell(x_curr)
            
            # Data loss
            data_loss = criterion(x_next_pred, x_next_true)
            
            # Regularization loss (optional - helps with sparsity for symbolic discovery)
            # Note: Too much regularization can hurt discovery accuracy
            if reg_activation > 0 or reg_entropy > 0:
                reg_loss = residual_kan.regularization_loss(
                    regularize_activation=reg_activation,
                    regularize_entropy=reg_entropy
                )
                loss = data_loss + reg_loss
            else:
                loss = data_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_kan.parameters(), 1.0)
            optimizer.step()
            
            train_loss += data_loss.item()  # Track data loss only for reporting
            batch_count += 1
        
        train_loss /= batch_count
        train_losses.append(train_loss)
        
        # Grid update
        if grid_update_freq > 0 and (epoch + 1) % grid_update_freq == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Updating grids...")
            for batch in train_loader:
                x_curr = batch[0].to(device)
                residual_kan.forward(x_curr, update_grid=True)
                break  # Update on first batch
        
        # Testing
        residual_kan.eval()
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
        
        # Early stopping check
        if test_loss < best_test_loss - early_stop_min_delta:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % grid_update_freq == 0 or (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Test: {test_loss:.6f}", end='')
            if patience_counter > 0:
                print(f" (patience: {patience_counter}/{early_stop_patience})")
            else:
                print()
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n    Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
            break
    
    return residual_kan, train_losses, test_losses


# ============================================================================
# 6. SYMBOLIC DISCOVERY VALIDATION
# ============================================================================

def extract_learned_residual_2d(system, residual_kan, normalizer, x_range, v_range, device='cpu'):
    """
    Extract learned residual function over 2D state space.
    
    Returns:
        X_grid, V_grid, learned_residual
    """
    residual_kan.eval()
    
    X_grid, V_grid = np.meshgrid(x_range, v_range)
    learned_residual = np.zeros_like(X_grid)
    
    min_vals = torch.tensor(normalizer.min_vals, dtype=torch.float32, device=device)
    range_vals = torch.tensor(normalizer.range, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                state = torch.tensor([[x, v]], dtype=torch.float32, device=device)
                state_norm = 2 * (state - min_vals) / range_vals - 1
                residual = residual_kan(state_norm).cpu().numpy()[0, 0]
                learned_residual[j, i] = residual
    
    return X_grid, V_grid, learned_residual


def compute_discovery_metrics(learned_residual, true_residual):
    """Compute MSE and R² for symbolic discovery."""
    mse = np.mean((learned_residual - true_residual) ** 2)
    ss_res = np.sum((true_residual - learned_residual) ** 2)
    ss_tot = np.sum((true_residual - np.mean(true_residual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return mse, r2


# ============================================================================
# 7. MAIN VALIDATION
# ============================================================================

def validate_system(system, device='cpu'):
    """Validate KAN-HRPINN on a system."""
    
    print(f"\n{'='*70}")
    print(f"VALIDATING: {system.name()}")
    print(f"{'='*70}")
    
    # Data generation
    train_loader, test_loader, normalizer, dt = generate_training_data(
        system, t_span=(0, 20), dt=0.05, num_trajectories=8, device=device
    )
    
    # Create KAN with multi-layer support
    print(f"\n  Creating KAN with layers_hidden=[2, 8, 1]...")
    residual_kan = KAN(
        layers_hidden=[2, 8, 1],
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1]
    ).to(device)
    
    kan_params = sum(p.numel() for p in residual_kan.parameters())
    print(f"  KAN Parameters: {kan_params}")
    
    # Training
    residual_kan, train_losses, test_losses = train_model(
        system, residual_kan, train_loader, test_loader, normalizer, dt,
        epochs=100, lr=1e-2, device=device,
        grid_update_freq=20,  # Configurable grid update frequency
        reg_activation=0.000, reg_entropy=0.000,  # Lighter regularization for better discovery
        early_stop_patience=30, early_stop_min_delta=1e-7  # More lenient early stopping
    )
    
    # Symbolic discovery validation
    print(f"\n  Validating symbolic discovery...")
    
    x_range = np.linspace(-2, 2, 40)
    v_range = np.linspace(-2, 2, 40)
    
    # True residual
    true_residual = np.zeros((len(v_range), len(x_range)))
    for i, x in enumerate(x_range):
        for j, v in enumerate(v_range):
            true_residual[j, i] = system.residual_true(np.array([x, v]))
    
    # Learned residual
    X_grid, V_grid, learned_residual = extract_learned_residual_2d(
        system, residual_kan, normalizer, x_range, v_range, device
    )
    
    # Compute metrics
    mse, r2 = compute_discovery_metrics(learned_residual, true_residual)
    
    print(f"\n  Results for {system.name()}:")
    print(f"    Test MSE: {test_losses[-1]:.6f}")
    print(f"    Symbolic Discovery MSE: {mse:.6f}")
    print(f"    Symbolic Discovery R²: {r2:.6f}")
    print(f"    KAN Parameters: {kan_params}")
    
    results = {
        'system_name': system.name(),
        'test_mse': test_losses[-1],
        'discovery_mse': mse,
        'discovery_r2': r2,
        'kan_params': kan_params,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'x_range': x_range,
        'v_range': v_range,
        'X_grid': X_grid,
        'V_grid': V_grid,
        'true_residual': true_residual,
        'learned_residual': learned_residual
    }
    
    return results


def main():
    """Main validation routine."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Validate systems
    systems = [DuffingOscillator(), VanDerPolOscillator()]
    
    all_results = {}
    for system in systems:
        results = validate_system(system, device=device)
        all_results[system.name()] = results
    
    # ========== VISUALIZATION ==========
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (system_name, results) in enumerate(all_results.items()):
        # Training curves
        axes[idx, 0].semilogy(results['train_losses'], label='Train', linewidth=2)
        axes[idx, 0].semilogy(results['test_losses'], label='Test', linewidth=2)
        axes[idx, 0].set_xlabel('Epoch')
        axes[idx, 0].set_ylabel('MSE Loss')
        axes[idx, 0].set_title(f'{system_name} - Training Curves')
        axes[idx, 0].legend(fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        
        # True residual heatmap
        im1 = axes[idx, 1].contourf(
            results['X_grid'], results['V_grid'], results['true_residual'],
            levels=20, cmap='RdBu_r'
        )
        axes[idx, 1].set_xlabel('x')
        axes[idx, 1].set_ylabel('v')
        axes[idx, 1].set_title(f'{system_name} - True Residual')
        plt.colorbar(im1, ax=axes[idx, 1])
        
        # Learned residual heatmap
        im2 = axes[idx, 2].contourf(
            results['X_grid'], results['V_grid'], results['learned_residual'],
            levels=20, cmap='RdBu_r'
        )
        axes[idx, 2].set_xlabel('x')
        axes[idx, 2].set_ylabel('v')
        axes[idx, 2].set_title(f'{system_name} - Learned Residual (MSE={results["discovery_mse"]:.4f})')
        plt.colorbar(im2, ax=axes[idx, 2])
    
    plt.tight_layout()
    output_path = 'kan_hrpinn_final_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # ========== SUMMARY REPORT ==========
    print(f"\n{'='*70}")
    print("SYMBOLIC DISCOVERY REPORT")
    print(f"{'='*70}\n")
    
    print(f"{'System':<20} {'Test MSE':<15} {'Discovery MSE':<15} {'Discovery R²':<15}")
    print("-" * 65)
    
    for system_name, results in all_results.items():
        print(f"{system_name:<20} {results['test_mse']:<15.6f} {results['discovery_mse']:<15.6f} {results['discovery_r2']:<15.6f}")
    
    print(f"\n{'System':<20} {'KAN Parameters':<20}")
    print("-" * 40)
    
    for system_name, results in all_results.items():
        print(f"{system_name:<20} {results['kan_params']:<20}")
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == '__main__':
    results = main()
