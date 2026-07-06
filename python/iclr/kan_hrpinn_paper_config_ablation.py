"""
KAN-HRPINN Configuration Ablation Study

Rigorous testing of KAN hyperparameters to find the most stable configuration
for symbolic discovery.
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
import csv
from collections import defaultdict
from tqdm import tqdm
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

class KANResidual(nn.Module):
    """Multi-Layer B-Spline KAN: 2 -> 8 -> 1"""
    
    def __init__(self, layers_hidden=[2, 8, 1], scale_base=1.0, scale_spline=1.0, grid_size=5, **kan_kwargs):
        super().__init__()
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,  # Now configurable
            spline_order=3,
            scale_noise=0.1,
            scale_base=scale_base,
            scale_spline=scale_spline,
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
                early_stop_min_delta=1e-7, verbose=False):
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
    
    for epoch in range(epochs):
        # Training
        residual_network.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            x_curr, x_next_true = batch[0].to(device), batch[1].to(device)
            x_next_pred = hrpinn_cell(x_curr)
            
            data_loss = criterion(x_next_pred, x_next_true)
            
            # Regularization (only for KAN)
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
    """Compute bootstrap confidence intervals, filtering out NaN values."""
    # Filter out NaN and Inf values
    data_clean = [x for x in data if not (np.isnan(x) or np.isinf(x))]
    
    if len(data_clean) == 0:
        return float('nan'), float('nan'), float('nan')
    
    n = len(data_clean)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_clean, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(data_clean)
    
    return mean, lower, upper


def format_ci(mean, lower, upper, decimals=6):
    """Format confidence interval as string, handling NaN values."""
    if np.isnan(mean) or np.isnan(lower) or np.isnan(upper):
        return "N/A"
    error = max(mean - lower, upper - mean)
    return f"{mean:.{decimals}f} ± {error:.{decimals}f}"


# ============================================================================
# 9. EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(system, config_params, seed, device='cpu'):
    """Run a single experiment with given seed and configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:
        # Generate data
        train_loader, test_loader, normalizer, dt = generate_training_data(
            system, seed=seed, device=device
        )
        
        # Create model with config parameters
        model_kwargs = {
            'layers_hidden': [2, 8, 1],
            'scale_base': config_params['scale_base'],
            'scale_spline': 1.0,
            'grid_size': config_params['grid_size'],
        }
        residual_network = KANResidual(**model_kwargs).to(device)
        
        # Train with hyperparameters
        train_params = {
            'grid_update_freq': config_params['grid_update_freq'],
            'early_stop_patience': config_params.get('early_stop_patience', 30),
            'reg_entropy': config_params['reg_entropy'],
            'reg_activation': config_params.get('reg_activation', 0.0),
            'verbose': False,
        }
        residual_network, train_losses, test_losses = train_model(
            system, residual_network, train_loader, test_loader, normalizer, dt,
            epochs=100, lr=1e-2, device=device, **train_params
        )
        
        # Compute metrics
        test_mse = test_losses[-1]
        discovery_mse, discovery_r2, learned_residual, true_residual = compute_discovery_metrics(
            residual_network, system, normalizer, device=device
        )
        
        # Parameter count
        param_count = sum(p.numel() for p in residual_network.parameters())
        
        # Symbolic extraction
        best_candidate, best_coef, symbolic_r2 = extract_symbolic_term(
            residual_network, system, normalizer, device=device
        )
        
        # Check for NaNs/Infs
        if np.isnan(discovery_r2) or np.isinf(discovery_r2):
            discovery_r2 = 0.0
        if np.isnan(test_mse) or np.isinf(test_mse):
            test_mse = float('nan')
        if np.isnan(discovery_mse) or np.isinf(discovery_mse):
            discovery_mse = float('nan')
        
        # Format symbolic formula
        symbolic_formula = f"{best_coef:.6f} * {best_candidate}"
        
        return {
            'test_mse': test_mse,
            'discovery_mse': discovery_mse,
            'discovery_r2': discovery_r2,
            'param_count': param_count,
            'symbolic_candidate': best_candidate,
            'symbolic_coef': best_coef,
            'symbolic_r2': symbolic_r2,
            'symbolic_formula': symbolic_formula,
            'learned_residual': learned_residual,
            'true_residual': true_residual,
            'success': True
        }
    except Exception as e:
        # Catch NaNs and other errors
        return {
            'test_mse': float('nan'),
            'discovery_mse': float('nan'),
            'discovery_r2': 0.0,
            'param_count': 0,
            'symbolic_candidate': 'N/A',
            'symbolic_coef': 0.0,
            'symbolic_r2': 0.0,
            'symbolic_formula': 'N/A',
            'learned_residual': None,
            'true_residual': None,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# 10. CONFIGURATION ABLATION SUITE
# ============================================================================

def run_config_ablation(n_seeds=100, device='cpu', output_dir='config_ablation_results'):
    """
    Run configuration ablation study with 7 configurations.
    
    Configurations:
    - Config A (Baseline): scale_base=1.0, grid_update_freq=20, reg_entropy=0.0, grid_size=5
    - Config B (Spline-Forced): scale_base=0.1, grid_update_freq=5, reg_entropy=0.0, grid_size=5
    - Config C (Sparse-Low): scale_base=0.1, grid_update_freq=5, reg_entropy=0.01, grid_size=5
    - Config D (Sparse-High): scale_base=0.1, grid_update_freq=5, reg_entropy=0.1, grid_size=5
    - Config E (Aggressive-Grid): scale_base=0.1, grid_update_freq=1, reg_entropy=0.0, grid_size=5
    - Config F (Coarse-Grid): scale_base=0.1, grid_update_freq=5, reg_entropy=0.0, grid_size=3
    - Config G (Fine-Grid): scale_base=0.1, grid_update_freq=5, reg_entropy=0.0, grid_size=10
    """
    device = torch.device(device if isinstance(device, str) else ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CONFIGURATION ABLATION STUDY")
    print(f"{'='*70}\n")
    print(f"Using device: {device}")
    print(f"Number of seeds per configuration: {n_seeds}")
    print(f"Output directory: {output_dir}\n")
    
    # Systems
    systems = [DuffingOscillator(), VanDerPolOscillator()]
    
    # Ablation configurations
    ablation_configs = {
        'Config A (Baseline)': {
            'scale_base': 1.0,
            'grid_update_freq': 20,
            'reg_entropy': 0.0,
            'grid_size': 5,
            'early_stop_patience': 20,
        },
        'Config B (Spline-Forced)': {
            'scale_base': 0.1,
            'grid_update_freq': 5,
            'reg_entropy': 0.0,
            'grid_size': 5,
            'early_stop_patience': 50,
        },
        'Config C (Sparse-Low)': {
            'scale_base': 0.1,
            'grid_update_freq': 5,
            'reg_entropy': 0.01,
            'grid_size': 5,
            'early_stop_patience': 50,
        },
        'Config D (Sparse-High)': {
            'scale_base': 0.1,
            'grid_update_freq': 5,
            'reg_entropy': 0.1,
            'grid_size': 5,
            'early_stop_patience': 50,
        },
        'Config E (Aggressive-Grid)': {
            'scale_base': 0.1,
            'grid_update_freq': 1,
            'reg_entropy': 0.0,
            'grid_size': 5,
            'early_stop_patience': 50,
        },
        'Config F (Coarse-Grid)': {
            'scale_base': 0.1,
            'grid_update_freq': 5,
            'reg_entropy': 0.0,
            'grid_size': 3,
            'early_stop_patience': 50,
        },
        'Config G (Fine-Grid)': {
            'scale_base': 0.1,
            'grid_update_freq': 5,
            'reg_entropy': 0.0,
            'grid_size': 10,
            'early_stop_patience': 50,
        },
    }
    
    # Store all raw results for CSV
    raw_results = []
    
    # Store aggregated results for statistics
    all_results = {}
    
    # Run experiments
    for system in systems:
        system_name = system.name()
        all_results[system_name] = {}
        
        for config_name, config_params in ablation_configs.items():
            print(f"\n{'='*70}")
            print(f"{system_name} - {config_name}")
            print(f"{'='*70}")
            print(f"  scale_base={config_params['scale_base']}, "
                  f"grid_update_freq={config_params['grid_update_freq']}, "
                  f"reg_entropy={config_params['reg_entropy']}, "
                  f"grid_size={config_params['grid_size']}")
            
            config_results = []
            
            # Run with progress bar
            success_count = 0
            for seed in tqdm(range(1, n_seeds + 1), desc=f"  Seeds"):
                result = run_single_experiment(system, config_params, seed, device)
                config_results.append(result)
                
                if result['success']:
                    success_count += 1
                
                # Store raw result for CSV
                raw_results.append({
                    'System': system_name,
                    'ConfigName': config_name,
                    'Seed': seed,
                    'TestMSE': result['test_mse'],
                    'DiscoveryR2': result['discovery_r2'],
                    'ParamCount': result['param_count'],
                    'SymbolicFormula': result['symbolic_formula'],
                    'Success': result['success'],
                })
            
            # Print success rate
            print(f"  Success rate: {success_count}/{n_seeds} ({100*success_count/n_seeds:.1f}%)")
            
            # Aggregate results (filter out failed experiments for statistics)
            metrics = ['test_mse', 'discovery_mse', 'discovery_r2', 'param_count', 'symbolic_r2']
            aggregated = {}
            
            for metric in metrics:
                values = [r[metric] for r in config_results]
                mean, lower, upper = bootstrap_ci(values)
                aggregated[metric] = {
                    'mean': mean,
                    'lower': lower,
                    'upper': upper,
                    'values': values
                }
            
            # Find median seed (by discovery R²) - only from successful experiments
            successful_results = [r for r in config_results if r['success'] and r['learned_residual'] is not None]
            if len(successful_results) > 0:
                discovery_r2_values = [r['discovery_r2'] for r in successful_results]
                median_idx = np.argsort(discovery_r2_values)[len(discovery_r2_values) // 2]
                median_result = successful_results[median_idx]
                
                aggregated['learned_residual'] = median_result['learned_residual']
                aggregated['true_residual'] = median_result['true_residual']
                aggregated['median_seed'] = successful_results.index(median_result) + 1
            else:
                # Fallback: use first config's true residual if available
                if system_name in all_results and len(all_results[system_name]) > 0:
                    first_config = list(all_results[system_name].keys())[0]
                    aggregated['true_residual'] = all_results[system_name][first_config]['true_residual']
                else:
                    # Generate true residual as fallback
                    x_range = np.linspace(-2, 2, 50)
                    v_range = np.linspace(-2, 2, 50)
                    X_grid, V_grid = np.meshgrid(x_range, v_range)
                    true_residual = np.zeros_like(X_grid)
                    for i, x in enumerate(x_range):
                        for j, v in enumerate(v_range):
                            true_residual[j, i] = system.residual_true(np.array([x, v]))
                    aggregated['true_residual'] = true_residual
                aggregated['learned_residual'] = None
                aggregated['median_seed'] = None
            
            all_results[system_name][config_name] = aggregated
    
    # Save raw CSV
    csv_file = os.path.join(output_dir, 'config_ablation_results.csv')
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['System', 'ConfigName', 'Seed', 'TestMSE', 'DiscoveryR2', 'ParamCount', 'SymbolicFormula', 'Success']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_results)
    print(f"\nSaved: {csv_file}")
    
    # Generate outputs
    print(f"\n{'='*70}")
    print("GENERATING OUTPUTS")
    print(f"{'='*70}")
    
    generate_config_ablation_table(all_results, output_dir=output_dir)
    generate_config_heatmaps(all_results, output_dir=output_dir)
    
    print(f"\n{'='*70}")
    print("CONFIGURATION ABLATION STUDY COMPLETE")
    print(f"{'='*70}\n")
    
    return all_results, raw_results


# ============================================================================
# 11. OUTPUT GENERATION
# ============================================================================

def generate_config_ablation_table(all_results, output_dir='config_ablation_results'):
    """Generate markdown table with statistics."""
    output_file = os.path.join(output_dir, 'config_ablation_table.md')
    with open(output_file, 'w') as f:
        f.write("# Configuration Ablation Study\n\n")
        f.write("Results with 95% Bootstrap Confidence Intervals (N=100 seeds)\n\n")
        
        # Table for each system
        for system_name in ['Duffing', 'VdP']:
            f.write(f"## {system_name} System\n\n")
            f.write("| Configuration | Test MSE | Discovery R² |\n")
            f.write("|---------------|----------|--------------|\n")
            
            configs = [
                'Config A (Baseline)',
                'Config B (Spline-Forced)',
                'Config C (Sparse-Low)',
                'Config D (Sparse-High)',
                'Config E (Aggressive-Grid)',
                'Config F (Coarse-Grid)',
                'Config G (Fine-Grid)',
            ]
            
            for config_name in configs:
                if system_name in all_results and config_name in all_results[system_name]:
                    r = all_results[system_name][config_name]
                    test_mse_str = format_ci(r['test_mse']['mean'], r['test_mse']['lower'], r['test_mse']['upper'])
                    disc_r2_str = format_ci(r['discovery_r2']['mean'], r['discovery_r2']['lower'], r['discovery_r2']['upper'])
                    f.write(f"| {config_name} | {test_mse_str} | {disc_r2_str} |\n")
            
            f.write("\n")
    
    print(f"Saved: {output_file}")


def generate_config_heatmaps(all_results, output_dir='config_ablation_results'):
    """Generate heatmap comparison for all 7 configs on VdP and Duffing."""
    output_file = os.path.join(output_dir, 'config_heatmaps.png')
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    
    systems = ['Duffing', 'VdP']
    configs = [
        'Config A (Baseline)',
        'Config B (Spline-Forced)',
        'Config C (Sparse-Low)',
        'Config D (Sparse-High)',
        'Config E (Aggressive-Grid)',
        'Config F (Coarse-Grid)',
        'Config G (Fine-Grid)',
    ]
    x_range = np.linspace(-2, 2, 50)
    v_range = np.linspace(-2, 2, 50)
    X_grid, V_grid = np.meshgrid(x_range, v_range)
    
    for idx, system_name in enumerate(systems):
        # True residual (first column)
        if system_name in all_results and configs[0] in all_results[system_name]:
            true_residual = all_results[system_name][configs[0]]['true_residual']
            im = axes[idx, 0].contourf(X_grid, V_grid, true_residual, levels=20, cmap='RdBu_r')
            axes[idx, 0].set_title(f'{system_name}\nTrue Residual')
            axes[idx, 0].set_xlabel('x')
            axes[idx, 0].set_ylabel('v')
            plt.colorbar(im, ax=axes[idx, 0])
        
        # Learned residuals for each config
        for col_idx, config_name in enumerate(configs, start=1):
            if system_name in all_results and config_name in all_results[system_name]:
                learned_residual = all_results[system_name][config_name]['learned_residual']
                disc_r2 = all_results[system_name][config_name]['discovery_r2']['mean']
                median_seed = all_results[system_name][config_name]['median_seed']
                
                if learned_residual is not None:
                    im = axes[idx, col_idx].contourf(X_grid, V_grid, learned_residual, levels=20, cmap='RdBu_r')
                    seed_str = f"Seed {median_seed}" if median_seed is not None else "No valid seed"
                    axes[idx, col_idx].set_title(f'{config_name}\n(R² = {disc_r2:.4f}, {seed_str})')
                else:
                    # Show empty plot with error message
                    axes[idx, col_idx].text(0.5, 0.5, 'All experiments failed', 
                                           ha='center', va='center', transform=axes[idx, col_idx].transAxes)
                    axes[idx, col_idx].set_title(f'{config_name}\n(R² = {disc_r2:.4f}, Failed)')
                
                axes[idx, col_idx].set_xlabel('x')
                axes[idx, col_idx].set_ylabel('v')
                if learned_residual is not None:
                    plt.colorbar(im, ax=axes[idx, col_idx])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")


# ============================================================================
# 12. MAIN
# ============================================================================

if __name__ == '__main__':
    results = run_config_ablation(n_seeds=100)
