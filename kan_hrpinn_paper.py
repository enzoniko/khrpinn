"""

KAN-HRPINN vs MLP-HRPINN Comparison with Statistical Rigor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import warnings
import os
from collections import defaultdict
warnings.filterwarnings('ignore')

from kan import KAN

# ============================================================================
# 1. SYSTEM PHYSICS DEFINITIONS
# ============================================================================

class SystemPhysics:
    """Base class for physics systems."""
    
    @staticmethod
    def full_dynamics(state, t):
        raise NotImplementedError
    
    @staticmethod
    def known_acceleration(state):
        raise NotImplementedError
    
    @staticmethod
    def residual_true(state):
        raise NotImplementedError
    
    @staticmethod
    def name():
        raise NotImplementedError


class DuffingOscillator(SystemPhysics):
    """
    Duffing: ẋ = v, v̇ = -x - 0.2v - 0.3x³
    Known: v̇ = -x - 0.2v
    Residual: v̇ += -0.3x³
    """
    
    @staticmethod
    def full_dynamics(state, t):
        x, v = state
        dx = v
        dv = -x - 0.2*v - 0.3*x**3
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


class VanDerPolOscillator(SystemPhysics):
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
# 2. RESIDUAL LEARNERS
# ============================================================================

class MLPResidual(nn.Module):
    """Standard MLP baseline: Linear -> ReLU -> Linear -> ReLU -> Linear"""
    
    def __init__(self, input_dim=2, hidden_dims=[32, 32], output_dim=1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, **kwargs):
        """Forward pass. Accepts kwargs for compatibility (e.g., update_grid is ignored)."""
        return self.net(x)


class KANResidual(nn.Module):
    """Multi-Layer B-Spline KAN: 2 -> 8 -> 1"""
    
    def __init__(self, layers_hidden=[2, 8, 1], scale_base=1.0, scale_spline=1.0, **kan_kwargs):
        super().__init__()
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=scale_base,  # Now configurable
            scale_spline=scale_spline,  # Now configurable
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            **kan_kwargs
        )
    
    def forward(self, x, update_grid=False):
        return self.kan(x, update_grid=update_grid)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.kan.regularization_loss(regularize_activation, regularize_entropy)


# ============================================================================
# 3. DATA GENERATION & NORMALIZATION
# ============================================================================

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


def generate_training_data(system, t_span=(0, 20), dt=0.05, num_trajectories=8, 
                          seed=42, device='cpu'):
    """Generate training data with consistent timestep."""
    np.random.seed(seed)
    all_trajectories = []
    
    for i in range(num_trajectories):
        x0 = np.random.uniform(-2, 2)
        v0 = np.random.uniform(-2, 2)
        
        t = np.arange(t_span[0], t_span[1] + dt, dt)
        trajectory = odeint(system.full_dynamics, [x0, v0], t)
        all_trajectories.append(trajectory)
    
    full_trajectory = np.vstack(all_trajectories)
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


# ============================================================================
# 4. HRPINN CELL WITH HARD CONSTRAINTS
# ============================================================================

class HRPINNCell(nn.Module):
    """HRPINN cell with hard-constrained physics."""
    
    def __init__(self, system, residual_network, normalizer, dt=0.05):
        super().__init__()
        self.system = system
        self.residual_network = residual_network
        self.normalizer = normalizer
        self.dt = dt
        
        self.register_buffer('norm_min', torch.tensor(normalizer.min_vals, dtype=torch.float32))
        self.register_buffer('norm_range', torch.tensor(normalizer.range, dtype=torch.float32))
    
    def forward(self, state_normalized):
        """Forward pass with hard constraints."""
        min_vals = self.norm_min.to(state_normalized.device)
        range_vals = self.norm_range.to(state_normalized.device)
        state = (state_normalized + 1) * range_vals / 2 + min_vals
        
        # Known physics (hard-constrained)
        dv_known = self.system.known_acceleration(state)
        if not isinstance(dv_known, torch.Tensor):
            dv_known = torch.tensor(dv_known, dtype=state.dtype, device=state.device)
        if dv_known.dim() == 0:
            dv_known = dv_known.unsqueeze(0)
        if dv_known.dim() == 1:
            dv_known = dv_known.unsqueeze(1)
        
        # Residual (learned)
        dv_residual = self.residual_network(state_normalized)
        
        # Total acceleration
        dv_total = dv_known + dv_residual
        
        # Integration step
        x, v = state[:, 0:1], state[:, 1:2]
        x_next = x + self.dt * v
        v_next = v + self.dt * dv_total
        
        next_state = torch.cat([x_next, v_next], dim=1)
        next_state_normalized = 2 * (next_state - min_vals) / range_vals - 1
        
        return next_state_normalized


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

def train_model(system, residual_network, train_loader, test_loader, normalizer, dt,
                epochs=100, lr=1e-2, device='cpu', grid_update_freq=20,
                reg_activation=0.0, reg_entropy=0.0, early_stop_patience=30,
                early_stop_min_delta=1e-7, verbose=True):
    """Train the HRPINN model."""
    
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
            # Aligned with kan_hrpinn_final.py: check if method exists before calling
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
            # Only update grids for KAN models (check for 'kan' attribute)
            if hasattr(residual_network, 'kan'):
                residual_network.eval()  # Set to eval mode for grid update
                for batch in train_loader:
                    x_curr = batch[0].to(device)
                    # KANResidual.forward accepts update_grid parameter
                    residual_network.forward(x_curr, update_grid=True)
                    break
                residual_network.train()  # Return to train mode
        
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


# ============================================================================
# 6. SYMBOLIC EXTRACTION
# ============================================================================

def extract_symbolic_term(model, system, normalizer, x_range=np.linspace(-2, 2, 50),
                          v_range=np.linspace(-2, 2, 50), device='cpu'):
    """
    Extract symbolic term by linear regression on candidate features.
    
    Returns:
        best_candidate: str, coefficient: float, r2: float
    """
    model.eval()
    
    # Candidate features dictionary
    candidates = {
        'x': lambda x, v: x,
        'v': lambda x, v: v,
        'x^3': lambda x, v: x**3,
        'xv': lambda x, v: x * v,
        'x^2v': lambda x, v: x**2 * v,
    }
    
    # Evaluate model on grid
    X_grid, V_grid = np.meshgrid(x_range, v_range)
    learned_residual = np.zeros_like(X_grid)
    
    min_vals = torch.tensor(normalizer.min_vals, dtype=torch.float32, device=device)
    range_vals = torch.tensor(normalizer.range, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                state = torch.tensor([[x, v]], dtype=torch.float32, device=device)
                state_norm = 2 * (state - min_vals) / range_vals - 1
                residual = model(state_norm).cpu().numpy()[0, 0]
                learned_residual[j, i] = residual
    
    # Flatten for regression
    x_flat = X_grid.flatten()
    v_flat = V_grid.flatten()
    residual_flat = learned_residual.flatten()
    
    # Build feature matrix
    features = []
    feature_names = []
    for name, func in candidates.items():
        feat = func(x_flat, v_flat)
        features.append(feat)
        feature_names.append(name)
    
    X_features = np.column_stack(features)
    
    # Linear regression with all features
    reg = LinearRegression()
    reg.fit(X_features, residual_flat)
    
    # Find best single feature (highest absolute coefficient)
    coefs = np.abs(reg.coef_)
    best_idx = np.argmax(coefs)
    best_candidate = feature_names[best_idx]
    best_coef = reg.coef_[best_idx]
    
    # Compute R² for best single feature
    y_pred_single = reg.coef_[best_idx] * X_features[:, best_idx]
    r2 = r2_score(residual_flat, y_pred_single)
    
    # For VdP: if best is 'x^2v', check if we should report as composite form
    # (This is just for reporting - the coefficient is what matters)
    
    return best_candidate, best_coef, r2


# ============================================================================
# 7. METRICS COMPUTATION
# ============================================================================

def compute_discovery_metrics(model, system, normalizer, x_range=np.linspace(-2, 2, 50),
                              v_range=np.linspace(-2, 2, 50), device='cpu'):
    """Compute discovery MSE and R²."""
    model.eval()
    
    X_grid, V_grid = np.meshgrid(x_range, v_range)
    learned_residual = np.zeros_like(X_grid)
    true_residual = np.zeros_like(X_grid)
    
    min_vals = torch.tensor(normalizer.min_vals, dtype=torch.float32, device=device)
    range_vals = torch.tensor(normalizer.range, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                state = torch.tensor([[x, v]], dtype=torch.float32, device=device)
                state_norm = 2 * (state - min_vals) / range_vals - 1
                residual = model(state_norm).cpu().numpy()[0, 0]
                learned_residual[j, i] = residual
                true_residual[j, i] = system.residual_true(np.array([x, v]))
    
    mse = np.mean((learned_residual - true_residual) ** 2)
    ss_res = np.sum((true_residual - learned_residual) ** 2)
    ss_tot = np.sum((true_residual - np.mean(true_residual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return mse, r2, learned_residual, true_residual


# ============================================================================
# 8. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence intervals."""
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(data)
    
    return mean, lower, upper


def format_ci(mean, lower, upper, decimals=6):
    """Format confidence interval as string."""
    error = max(mean - lower, upper - mean)
    return f"{mean:.{decimals}f} ± {error:.{decimals}f}"


# ============================================================================
# 9. EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run experiments with multiple seeds and compute statistics."""
    
    def __init__(self, n_seeds=5, device='cpu'):
        self.n_seeds = n_seeds
        self.device = device
        self.results = defaultdict(list)
    
    def run_single_experiment(self, system, model_class, model_kwargs, seed, 
                             train_kwargs=None):
        """Run a single experiment with given seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate data
        train_loader, test_loader, normalizer, dt = generate_training_data(
            system, seed=seed, device=self.device
        )
        
        # Create model
        if model_class == MLPResidual:
            residual_network = MLPResidual(**model_kwargs).to(self.device)
        elif model_class == KANResidual:
            residual_network = KANResidual(**model_kwargs).to(self.device)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # Train with hyperparameters
        train_params = train_kwargs if train_kwargs else {}
        residual_network, train_losses, test_losses = train_model(
            system, residual_network, train_loader, test_loader, normalizer, dt,
            epochs=100, lr=1e-2, device=self.device, verbose=False, **train_params
        )
        
        # Compute metrics
        test_mse = test_losses[-1]
        discovery_mse, discovery_r2, learned_residual, true_residual = compute_discovery_metrics(
            residual_network, system, normalizer, device=self.device
        )
        
        # Parameter count
        param_count = sum(p.numel() for p in residual_network.parameters())
        
        # Symbolic extraction
        best_candidate, best_coef, symbolic_r2 = extract_symbolic_term(
            residual_network, system, normalizer, device=self.device
        )
        
        return {
            'test_mse': test_mse,
            'discovery_mse': discovery_mse,
            'discovery_r2': discovery_r2,
            'param_count': param_count,
            'symbolic_candidate': best_candidate,
            'symbolic_coef': best_coef,
            'symbolic_r2': symbolic_r2,
            'learned_residual': learned_residual,
            'true_residual': true_residual,
        }
    
    def run_experiments(self, system, model_class, model_kwargs, model_name, train_kwargs=None):
        """Run experiments for all seeds."""
        print(f"\n{'='*70}")
        print(f"Running {model_name} on {system.name()} ({self.n_seeds} seeds)")
        print(f"{'='*70}")
        
        all_results = []
        for seed in range(1, self.n_seeds + 1):
            print(f"  Seed {seed}/{self.n_seeds}...", end=' ', flush=True)
            result = self.run_single_experiment(system, model_class, model_kwargs, seed, train_kwargs)
            all_results.append(result)
            print(f"Test MSE: {result['test_mse']:.6f}, Discovery R²: {result['discovery_r2']:.6f}")
        
        # Aggregate results
        metrics = ['test_mse', 'discovery_mse', 'discovery_r2', 'param_count', 'symbolic_r2']
        aggregated = {}
        
        for metric in metrics:
            values = [r[metric] for r in all_results]
            mean, lower, upper = bootstrap_ci(values)
            aggregated[metric] = {
                'mean': mean,
                'lower': lower,
                'upper': upper,
                'values': values
            }
        
        # Symbolic extraction results
        symbolic_results = []
        for i, r in enumerate(all_results):
            symbolic_results.append({
                'seed': i + 1,
                'candidate': r['symbolic_candidate'],
                'coef': r['symbolic_coef'],
                'r2': r['symbolic_r2']
            })
        
        # Store best seed's residuals for visualization (highest discovery R²)
        best_idx = np.argmax([r['discovery_r2'] for r in all_results])
        aggregated['learned_residual'] = all_results[best_idx]['learned_residual']
        aggregated['true_residual'] = all_results[best_idx]['true_residual']
        aggregated['best_seed'] = best_idx + 1
        aggregated['symbolic_results'] = symbolic_results
        
        return aggregated


# ============================================================================
# 10. OUTPUT GENERATION
# ============================================================================

def generate_comparison_table(all_results, output_file='comparison_table.md'):
    """Generate markdown comparison table."""
    with open(output_file, 'w') as f:
        f.write("# KAN-HRPINN vs MLP-HRPINN Comparison\n\n")
        f.write("Results with 95% Bootstrap Confidence Intervals (N=5 seeds)\n\n")
        
        f.write("| System | Model | Test MSE | Discovery MSE | Discovery R² | Parameters |\n")
        f.write("|--------|-------|----------|---------------|--------------|------------|\n")
        
        for system_name in ['Duffing', 'VdP']:
            for model_name in ['MLP', 'KAN']:
                key = f"{system_name}_{model_name}"
                if key in all_results:
                    r = all_results[key]
                    test_mse_str = format_ci(r['test_mse']['mean'], r['test_mse']['lower'], r['test_mse']['upper'])
                    disc_mse_str = format_ci(r['discovery_mse']['mean'], r['discovery_mse']['lower'], r['discovery_mse']['upper'])
                    disc_r2_str = format_ci(r['discovery_r2']['mean'], r['discovery_r2']['lower'], r['discovery_r2']['upper'])
                    params = int(r['param_count']['mean'])
                    
                    f.write(f"| {system_name} | {model_name} | {test_mse_str} | {disc_mse_str} | {disc_r2_str} | {params} |\n")
    
    print(f"\nSaved: {output_file}")


def generate_heatmaps(all_results, output_file='discovery_heatmaps.png'):
    """Generate heatmap comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    systems = ['Duffing', 'VdP']
    x_range = np.linspace(-2, 2, 50)
    v_range = np.linspace(-2, 2, 50)
    X_grid, V_grid = np.meshgrid(x_range, v_range)
    
    for idx, system_name in enumerate(systems):
        # True residual
        if f"{system_name}_KAN" in all_results:
            true_residual = all_results[f"{system_name}_KAN"]['true_residual']
            im1 = axes[idx, 0].contourf(X_grid, V_grid, true_residual, levels=20, cmap='RdBu_r')
            axes[idx, 0].set_title(f'{system_name} - True Residual')
            axes[idx, 0].set_xlabel('x')
            axes[idx, 0].set_ylabel('v')
            plt.colorbar(im1, ax=axes[idx, 0])
        
        # MLP learned
        if f"{system_name}_MLP" in all_results:
            mlp_residual = all_results[f"{system_name}_MLP"]['learned_residual']
            im2 = axes[idx, 1].contourf(X_grid, V_grid, mlp_residual, levels=20, cmap='RdBu_r')
            axes[idx, 1].set_title(f'{system_name} - MLP Learned')
            axes[idx, 1].set_xlabel('x')
            axes[idx, 1].set_ylabel('v')
            plt.colorbar(im2, ax=axes[idx, 1])
        
        # KAN learned
        if f"{system_name}_KAN" in all_results:
            kan_residual = all_results[f"{system_name}_KAN"]['learned_residual']
            im3 = axes[idx, 2].contourf(X_grid, V_grid, kan_residual, levels=20, cmap='RdBu_r')
            axes[idx, 2].set_title(f'{system_name} - KAN Learned')
            axes[idx, 2].set_xlabel('x')
            axes[idx, 2].set_ylabel('v')
            plt.colorbar(im3, ax=axes[idx, 2])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")


def generate_symbolic_report(all_results, output_file='symbolic_extraction_report.txt'):
    """Generate symbolic extraction report."""
    with open(output_file, 'w') as f:
        f.write("Symbolic Extraction Report\n")
        f.write("=" * 70 + "\n\n")
        
        for system_name in ['Duffing', 'VdP']:
            key = f"{system_name}_KAN"
            if key in all_results:
                f.write(f"{system_name} System - KAN Results:\n")
                f.write("-" * 70 + "\n")
                
                symbolic_results = all_results[key]['symbolic_results']
                for result in symbolic_results:
                    f.write(f"Seed {result['seed']}: Discovered {result['coef']:.6f} * {result['candidate']} ")
                    f.write(f"(R² = {result['r2']:.6f})\n")
                
                # Expected values
                if system_name == 'Duffing':
                    f.write(f"\nExpected: -0.3 * x^3\n")
                    f.write(f"Note: The true residual is -0.3x³. Best candidate should be 'x^3' with coefficient ≈ -0.3\n")
                elif system_name == 'VdP':
                    f.write(f"\nExpected: (1-x²)v = v - x²v\n")
                    f.write(f"Note: The true residual is (1-x²)v. Best candidate should be 'x^2v' with coefficient ≈ -1.0\n")
                    f.write(f"      (or 'v' with coefficient ≈ 1.0, but 'x^2v' captures the nonlinearity better)\n")
                
                f.write("\n")
    
    print(f"Saved: {output_file}")


# ============================================================================
# 11. MAIN EXPERIMENT
# ============================================================================

def main():
    """Run complete experiment suite."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Systems
    systems = [DuffingOscillator(), VanDerPolOscillator()]
    
    # Model configurations
    model_configs = {
        'MLP': (MLPResidual, {'input_dim': 2, 'hidden_dims': [32, 32], 'output_dim': 1}),
        'KAN': (KANResidual, {'layers_hidden': [2, 8, 1]}),
    }
    
    # Run experiments
    runner = ExperimentRunner(n_seeds=5, device=device)
    all_results = {}
    
    for system in systems:
        for model_name, (model_class, model_kwargs) in model_configs.items():
            key = f"{system.name()}_{model_name}"
            results = runner.run_experiments(system, model_class, model_kwargs, model_name)
            all_results[key] = results
    
    # Generate outputs
    print(f"\n{'='*70}")
    print("GENERATING OUTPUTS")
    print(f"{'='*70}")
    
    generate_comparison_table(all_results)
    generate_heatmaps(all_results)
    generate_symbolic_report(all_results)
    
    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*70}\n")
    
    return all_results


# ============================================================================
# 12. ABLATION SUITE
# ============================================================================

def run_ablation_suite(n_seeds=5, device='cpu'):
    """
    Run ablation study to prove Linear Bias and Grid Starvation issues.
    
    Configurations:
    - Config A (Baseline): scale_base=1.0, grid_update_freq=20, reg_entropy=0.0
    - Config B (Spline-Forced): scale_base=0.1, grid_update_freq=5, reg_entropy=0.0
    - Config C (Sparse-Symbolic): scale_base=0.1, grid_update_freq=5, reg_entropy=0.01
    """
    device = torch.device(device if isinstance(device, str) else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n{'='*70}")
    print("ABLATION STUDY: Linear Bias & Grid Starvation")
    print(f"{'='*70}\n")
    print(f"Using device: {device}\n")
    
    # Systems
    systems = [DuffingOscillator(), VanDerPolOscillator()]
    
    # Ablation configurations
    ablation_configs = {
        'Config A (Baseline)': {
            'scale_base': 1.0,
            'scale_spline': 1.0,
            'grid_update_freq': 20,
            'early_stop_patience': 20,
            'reg_entropy': 0.0,
            'reg_activation': 0.0,
        },
        'Config B (Spline-Forced)': {
            'scale_base': 0.1,  # Critical: Low linear weight
            'scale_spline': 1.0,
            'grid_update_freq': 5,  # Critical: Frequent updates
            'early_stop_patience': 50,
            'reg_entropy': 0.0,
            'reg_activation': 0.0,
        },
        'Config C (Sparse-Symbolic)': {
            'scale_base': 0.1,
            'scale_spline': 1.0,
            'grid_update_freq': 5,
            'early_stop_patience': 50,
            'reg_entropy': 0.01,  # Critical: Regularization
            'reg_activation': 0.0,
        },
    }
    
    # Run ablation experiments
    runner = ExperimentRunner(n_seeds=n_seeds, device=device)
    all_results = {}
    
    for system in systems:
        for config_name, config_params in ablation_configs.items():
            key = f"{system.name()}_{config_name}"
            print(f"\n{'='*70}")
            print(f"{system.name()} - {config_name}")
            print(f"{'='*70}")
            print(f"  scale_base={config_params['scale_base']}, "
                  f"grid_update_freq={config_params['grid_update_freq']}, "
                  f"reg_entropy={config_params['reg_entropy']}")
            
            # Model kwargs
            model_kwargs = {
                'layers_hidden': [2, 8, 1],
                'scale_base': config_params['scale_base'],
                'scale_spline': config_params['scale_spline'],
            }
            
            # Training kwargs
            train_kwargs = {
                'grid_update_freq': config_params['grid_update_freq'],
                'early_stop_patience': config_params['early_stop_patience'],
                'reg_entropy': config_params['reg_entropy'],
                'reg_activation': config_params['reg_activation'],
            }
            
            # Run experiments
            results = runner.run_experiments(
                system, KANResidual, model_kwargs, config_name, train_kwargs
            )
            all_results[key] = results
    
    # Generate outputs
    print(f"\n{'='*70}")
    print("GENERATING ABLATION OUTPUTS")
    print(f"{'='*70}")
    
    generate_ablation_report(all_results)
    generate_ablation_heatmaps(all_results)
    print_ablation_conclusion(all_results)
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}\n")
    
    return all_results


def generate_ablation_report(all_results, output_file='ablation_report.md'):
    """Generate markdown ablation report."""
    with open(output_file, 'w') as f:
        f.write("# Ablation Study: Linear Bias & Grid Starvation\n\n")
        f.write("Results with 95% Bootstrap Confidence Intervals (N=5 seeds)\n\n")
        
        # Table for each system
        for system_name in ['Duffing', 'VdP']:
            f.write(f"## {system_name} System\n\n")
            f.write("| Configuration | Test MSE | Discovery R² |\n")
            f.write("|---------------|----------|--------------|\n")
            
            configs = ['Config A (Baseline)', 'Config B (Spline-Forced)', 'Config C (Sparse-Symbolic)']
            for config_name in configs:
                key = f"{system_name}_{config_name}"
                if key in all_results:
                    r = all_results[key]
                    test_mse_str = format_ci(r['test_mse']['mean'], r['test_mse']['lower'], r['test_mse']['upper'])
                    disc_r2_str = format_ci(r['discovery_r2']['mean'], r['discovery_r2']['lower'], r['discovery_r2']['upper'])
                    f.write(f"| {config_name} | {test_mse_str} | {disc_r2_str} |\n")
            
            f.write("\n")
    
    print(f"Saved: {output_file}")


def generate_ablation_heatmaps(all_results, output_file='ablation_heatmaps.png'):
    """Generate ablation heatmap comparison."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    systems = ['Duffing', 'VdP']
    configs = ['Config A (Baseline)', 'Config B (Spline-Forced)', 'Config C (Sparse-Symbolic)']
    x_range = np.linspace(-2, 2, 50)
    v_range = np.linspace(-2, 2, 50)
    X_grid, V_grid = np.meshgrid(x_range, v_range)
    
    for idx, system_name in enumerate(systems):
        # True residual (first column)
        key_true = f"{system_name}_Config A (Baseline)"
        if key_true in all_results:
            true_residual = all_results[key_true]['true_residual']
            im = axes[idx, 0].contourf(X_grid, V_grid, true_residual, levels=20, cmap='RdBu_r')
            axes[idx, 0].set_title(f'{system_name} - True Residual')
            axes[idx, 0].set_xlabel('x')
            axes[idx, 0].set_ylabel('v')
            plt.colorbar(im, ax=axes[idx, 0])
        
        # Learned residuals for each config
        for col_idx, config_name in enumerate(configs, start=1):
            key = f"{system_name}_{config_name}"
            if key in all_results:
                learned_residual = all_results[key]['learned_residual']
                disc_r2 = all_results[key]['discovery_r2']['mean']
                im = axes[idx, col_idx].contourf(X_grid, V_grid, learned_residual, levels=20, cmap='RdBu_r')
                axes[idx, col_idx].set_title(f'{config_name}\n(R² = {disc_r2:.4f})')
                axes[idx, col_idx].set_xlabel('x')
                axes[idx, col_idx].set_ylabel('v')
                plt.colorbar(im, ax=axes[idx, col_idx])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")


def print_ablation_conclusion(all_results):
    """Print conclusion about which configuration performed best."""
    print(f"\n{'='*70}")
    print("ABLATION CONCLUSION")
    print(f"{'='*70}\n")
    
    for system_name in ['Duffing', 'VdP']:
        print(f"{system_name} System:")
        print("-" * 70)
        
        best_config = None
        best_r2 = -float('inf')
        config_results = {}
        
        configs = ['Config A (Baseline)', 'Config B (Spline-Forced)', 'Config C (Sparse-Symbolic)']
        for config_name in configs:
            key = f"{system_name}_{config_name}"
            if key in all_results:
                r2 = all_results[key]['discovery_r2']['mean']
                test_mse = all_results[key]['test_mse']['mean']
                config_results[config_name] = {'r2': r2, 'test_mse': test_mse}
                
                print(f"  {config_name}:")
                print(f"    Discovery R²: {r2:.6f}")
                print(f"    Test MSE: {test_mse:.6f}")
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_config = config_name
        
        print(f"\n  → Best Configuration: {best_config} (R² = {best_r2:.6f})")
        
        # Expected results
        if system_name == 'Duffing':
            print(f"  Expected: Config B or C should recover -0.3x³ term")
        elif system_name == 'VdP':
            print(f"  Expected: Config B or C should recover (1-x²)v term")
        
        print()


if __name__ == '__main__':
    # Run ablation suite instead of main comparison
    results = run_ablation_suite(n_seeds=5)
