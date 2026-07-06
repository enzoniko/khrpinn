"""
KAN-HRPINN Parameter Efficiency Ablation Study
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
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import scipy.stats as stats
import pickle
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
    """Multi-Layer B-Spline KAN with flexible layers"""
    
    def __init__(self, layers_hidden=[2, 8, 1], scale_base=1.0, scale_spline=1.0, **kan_kwargs):
        super().__init__()
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=5,
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
# 8. EXPERIMENT CONFIGURATIONS
# ============================================================================

def get_experiment_configs():
    """Define all experiment configurations."""
    configs = []
    
    # KAN Variations (B-Spline, Config B settings)
    kan_configs = [
        {
            'Arch': 'KAN',
            'ConfigName': 'KAN Very Small',
            'model_class': KANResidual,
            'model_kwargs': {'layers_hidden': [2, 4, 1], 'scale_base': 0.1, 'scale_spline': 1.0},
            'train_kwargs': {'grid_update_freq': 5, 'early_stop_patience': 50, 'reg_entropy': 0.0, 'reg_activation': 0.0}
        },
        {
            'Arch': 'KAN',
            'ConfigName': 'KAN Small',
            'model_class': KANResidual,
            'model_kwargs': {'layers_hidden': [2, 8, 1], 'scale_base': 0.1, 'scale_spline': 1.0},
            'train_kwargs': {'grid_update_freq': 5, 'early_stop_patience': 50, 'reg_entropy': 0.0, 'reg_activation': 0.0}
        },
        {
            'Arch': 'KAN',
            'ConfigName': 'KAN Wide',
            'model_class': KANResidual,
            'model_kwargs': {'layers_hidden': [2, 16, 1], 'scale_base': 0.1, 'scale_spline': 1.0},
            'train_kwargs': {'grid_update_freq': 5, 'early_stop_patience': 50, 'reg_entropy': 0.0, 'reg_activation': 0.0}
        },
        {
            'Arch': 'KAN',
            'ConfigName': 'KAN Deep',
            'model_class': KANResidual,
            'model_kwargs': {'layers_hidden': [2, 8, 8, 1], 'scale_base': 0.1, 'scale_spline': 1.0},
            'train_kwargs': {'grid_update_freq': 5, 'early_stop_patience': 50, 'reg_entropy': 0.0, 'reg_activation': 0.0}
        },
    ]
    
    # MLP Variations (ReLU)
    mlp_configs = [
        {
            'Arch': 'MLP',
            'ConfigName': 'MLP Tiny',
            'model_class': MLPResidual,
            'model_kwargs': {'input_dim': 2, 'hidden_dims': [8, 8], 'output_dim': 1},
            'train_kwargs': {'grid_update_freq': 0, 'early_stop_patience': 50, 'reg_entropy': 0.0, 'reg_activation': 0.0}
        },
        {
            'Arch': 'MLP',
            'ConfigName': 'MLP Small',
            'model_class': MLPResidual,
            'model_kwargs': {'input_dim': 2, 'hidden_dims': [16, 16], 'output_dim': 1},
            'train_kwargs': {'grid_update_freq': 0, 'early_stop_patience': 50, 'reg_entropy': 0.0, 'reg_activation': 0.0}
        },
        {
            'Arch': 'MLP',
            'ConfigName': 'MLP Medium',
            'model_class': MLPResidual,
            'model_kwargs': {'input_dim': 2, 'hidden_dims': [32, 32], 'output_dim': 1},
            'train_kwargs': {'grid_update_freq': 0, 'early_stop_patience': 50, 'reg_entropy': 0.0, 'reg_activation': 0.0}
        },
        {
            'Arch': 'MLP',
            'ConfigName': 'MLP Large',
            'model_class': MLPResidual,
            'model_kwargs': {'input_dim': 2, 'hidden_dims': [64, 64], 'output_dim': 1},
            'train_kwargs': {'grid_update_freq': 0, 'early_stop_patience': 50, 'reg_entropy': 0.0, 'reg_activation': 0.0}
        },
    ]
    
    configs.extend(kan_configs)
    configs.extend(mlp_configs)
    
    return configs


# ============================================================================
# 9. EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(system, config, seed, device='cpu'):
    """Run a single experiment with given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:
        # Generate data
        train_loader, test_loader, normalizer, dt = generate_training_data(
            system, seed=seed, device=device
        )
        
        # Create model
        model_class = config['model_class']
        model_kwargs = config['model_kwargs']
        residual_network = model_class(**model_kwargs).to(device)
        
        # Train with hyperparameters
        train_kwargs = config['train_kwargs']
        residual_network, train_losses, test_losses = train_model(
            system, residual_network, train_loader, test_loader, normalizer, dt,
            epochs=100, lr=1e-2, device=device, verbose=False, **train_kwargs
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
        
        # Check for NaNs
        if np.isnan(discovery_r2) or np.isinf(discovery_r2):
            discovery_r2 = 0.0
        
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
            'learned_residual': None,
            'true_residual': None,
            'success': False,
            'error': str(e)
        }


def run_experiments(systems, configs, n_seeds=100, device='cpu', output_dir='parameter_ablation_results'):
    """Run all experiments and save results."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    residuals_dict = {}  # Store residuals for visualization
    
    for system in systems:
        for config in configs:
            print(f"\n{'='*70}")
            print(f"{system.name()} - {config['ConfigName']} ({config['Arch']})")
            print(f"{'='*70}")
            
            # Run experiments with progress bar
            for seed in tqdm(range(1, n_seeds + 1), desc=f"Seeds"):
                result = run_single_experiment(system, config, seed, device)
                
                # Store result
                all_results.append({
                    'System': system.name(),
                    'Arch': config['Arch'],
                    'ConfigName': config['ConfigName'],
                    'Seed': seed,
                    'ParamCount': result['param_count'],
                    'TestMSE': result['test_mse'],
                    'DiscoveryR2': result['discovery_r2'],
                    'BestSymbolicTerm': result['symbolic_candidate'],
                    'Success': result['success']
                })
                
                # Store residuals for visualization (only if successful)
                if result['success'] and result['learned_residual'] is not None:
                    key = (system.name(), config['ConfigName'], seed)
                    residuals_dict[key] = {
                        'learned_residual': result['learned_residual'],
                        'true_residual': result['true_residual']
                    }
            
            # Print summary for this config
            config_results = [r for r in all_results 
                            if r['System'] == system.name() 
                            and r['ConfigName'] == config['ConfigName']]
            if config_results:
                successful = [r for r in config_results if r['Success']]
                if successful:
                    mean_r2 = np.mean([r['DiscoveryR2'] for r in successful])
                    mean_params = np.mean([r['ParamCount'] for r in successful])
                    print(f"  Mean Discovery R²: {mean_r2:.6f}")
                    print(f"  Mean Parameters: {mean_params:.0f}")
                    print(f"  Success Rate: {len(successful)}/{len(config_results)}")
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'experiment_results_full.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")
    
    # Save residuals for visualization
    residuals_path = os.path.join(output_dir, 'residuals_dict.pkl')
    with open(residuals_path, 'wb') as f:
        pickle.dump(residuals_dict, f)
    print(f"Saved residuals to: {residuals_path}")
    
    return all_results, residuals_dict


# ============================================================================
# 10. VISUALIZATION GENERATION
# ============================================================================

def generate_pareto_efficiency_plot(all_results, output_dir='parameter_ablation_results'):
    """Generate Pareto efficiency scatter plot."""
    df = pd.DataFrame(all_results)
    
    # Filter successful experiments only
    df = df[df['Success'] == True]
    
    # Group by System, Arch, ConfigName and compute statistics
    grouped_list = []
    for (system, arch, config), group in df.groupby(['System', 'Arch', 'ConfigName']):
        mean_params = group['ParamCount'].mean()
        mean_r2 = group['DiscoveryR2'].mean()
        n = len(group)
        r2_values = group['DiscoveryR2'].values
        
        # Compute 95% CI using t-distribution
        if n > 1:
            sem = stats.sem(r2_values)
            ci = stats.t.interval(0.95, n-1, loc=mean_r2, scale=sem)
            r2_lower, r2_upper = ci[0], ci[1]
        else:
            r2_lower, r2_upper = mean_r2, mean_r2
        
        grouped_list.append({
            'System': system,
            'Arch': arch,
            'ConfigName': config,
            'MeanParams': mean_params,
            'MeanR2': mean_r2,
            'R2Lower': r2_lower,
            'R2Upper': r2_upper
        })
    
    grouped = pd.DataFrame(grouped_list)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, system_name in enumerate(['Duffing', 'VdP']):
        ax = axes[idx]
        system_data = grouped[grouped['System'] == system_name]
        
        # Separate KAN and MLP
        kan_data = system_data[system_data['Arch'] == 'KAN'].sort_values('MeanParams')
        mlp_data = system_data[system_data['Arch'] == 'MLP'].sort_values('MeanParams')
        
        # Plot KAN with error bars
        if len(kan_data) > 0:
            ax.errorbar(kan_data['MeanParams'], kan_data['MeanR2'],
                       yerr=[kan_data['MeanR2'] - kan_data['R2Lower'], 
                             kan_data['R2Upper'] - kan_data['MeanR2']],
                       fmt='o-', color='blue', label='KAN', linewidth=2, markersize=8,
                       capsize=5, capthick=2)
            # Add labels
            for _, row in kan_data.iterrows():
                ax.annotate(row['ConfigName'].replace('KAN ', ''), 
                           (row['MeanParams'], row['MeanR2']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot MLP with error bars
        if len(mlp_data) > 0:
            ax.errorbar(mlp_data['MeanParams'], mlp_data['MeanR2'],
                       yerr=[mlp_data['MeanR2'] - mlp_data['R2Lower'], 
                             mlp_data['R2Upper'] - mlp_data['MeanR2']],
                       fmt='s-', color='red', label='MLP', linewidth=2, markersize=8,
                       capsize=5, capthick=2)
            # Add labels
            for _, row in mlp_data.iterrows():
                ax.annotate(row['ConfigName'].replace('MLP ', ''), 
                           (row['MeanParams'], row['MeanR2']),
                           xytext=(5, -15), textcoords='offset points', fontsize=8)
        
        ax.set_xscale('log')
        ax.set_xlabel('Parameters (log scale)', fontsize=12)
        ax.set_ylabel('Discovery R²', fontsize=12)
        ax.set_title(f'{system_name} System', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pareto_efficiency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def load_residuals_dict(output_dir='parameter_ablation_results'):
    """Load residuals dictionary from pickle file if it exists."""
    residuals_path = os.path.join(output_dir, 'residuals_dict.pkl')
    if os.path.exists(residuals_path):
        with open(residuals_path, 'rb') as f:
            return pickle.load(f)
    return {}


def generate_ablation_heatmaps(all_results, residuals_dict=None, output_dir='parameter_ablation_results'):
    """Generate heatmap comparison for median seed."""
    if residuals_dict is None:
        residuals_dict = load_residuals_dict(output_dir)
    
    df = pd.DataFrame(all_results)
    
    # Select configurations to visualize
    selected_configs = ['KAN Small', 'KAN Deep', 'MLP Small', 'MLP Large']
    systems = ['Duffing', 'VdP']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    x_range = np.linspace(-2, 2, 50)
    v_range = np.linspace(-2, 2, 50)
    X_grid, V_grid = np.meshgrid(x_range, v_range)
    
    for row_idx, system_name in enumerate(systems):
        for col_idx, config_name in enumerate(selected_configs):
            ax = axes[row_idx, col_idx]
            
            # Get data for this system and config
            config_data = df[(df['System'] == system_name) & 
                            (df['ConfigName'] == config_name) &
                            (df['Success'] == True)]
            
            if len(config_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{config_name}', fontsize=10)
                continue
            
            # Get median seed (by DiscoveryR2)
            config_data = config_data.sort_values('DiscoveryR2')
            median_idx = len(config_data) // 2
            median_seed = int(config_data.iloc[median_idx]['Seed'])
            median_r2 = config_data.iloc[median_idx]['DiscoveryR2']
            
            # Get residuals for median seed
            key = (system_name, config_name, median_seed)
            if key in residuals_dict:
                learned_residual = residuals_dict[key]['learned_residual']
                im = ax.contourf(X_grid, V_grid, learned_residual, levels=20, cmap='RdBu_r')
                ax.set_title(f'{system_name} - {config_name}\n(Seed {median_seed}, R²={median_r2:.4f})', fontsize=10)
                ax.set_xlabel('x')
                ax.set_ylabel('v')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f'Median Seed: {median_seed}\nR²: {median_r2:.4f}\n(Residuals not available)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{system_name} - {config_name}', fontsize=10)
                ax.set_xlabel('x')
                ax.set_ylabel('v')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ablation_heatmaps.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_final_table(all_results, output_dir='parameter_ablation_results'):
    """Generate summary table with Mean R² and Parameter counts."""
    df = pd.DataFrame(all_results)
    
    # Group by System, Arch, ConfigName
    summary = df.groupby(['System', 'Arch', 'ConfigName']).agg({
        'ParamCount': 'mean',
        'DiscoveryR2': ['mean', 'std', 'count']
    }).reset_index()
    
    summary.columns = ['System', 'Arch', 'ConfigName', 'MeanParams', 'MeanR2', 'StdR2', 'Count']
    
    # Compute 95% CI
    summary['R2Lower'] = summary['MeanR2'] - 1.96 * summary['StdR2'] / np.sqrt(summary['Count'])
    summary['R2Upper'] = summary['MeanR2'] + 1.96 * summary['StdR2'] / np.sqrt(summary['Count'])
    
    output_path = os.path.join(output_dir, 'final_table.md')
    with open(output_path, 'w') as f:
        f.write("# Parameter Efficiency Ablation Study - Summary Table\n\n")
        f.write("Results with Mean ± 95% CI (N=100 seeds)\n\n")
        
        for system_name in ['Duffing', 'VdP']:
            f.write(f"## {system_name} System\n\n")
            f.write("| Architecture | Configuration | Parameters | Discovery R² |\n")
            f.write("|--------------|----------------|------------|--------------|\n")
            
            system_data = summary[summary['System'] == system_name]
            for _, row in system_data.iterrows():
                r2_str = f"{row['MeanR2']:.6f} ± {row['MeanR2'] - row['R2Lower']:.6f}"
                f.write(f"| {row['Arch']} | {row['ConfigName']} | {int(row['MeanParams'])} | {r2_str} |\n")
            
            f.write("\n")
    
    print(f"Saved: {output_path}")


# ============================================================================
# 11. MAIN EXPERIMENT
# ============================================================================

def main():
    """Run complete parameter efficiency ablation study."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("PARAMETER EFFICIENCY ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Using device: {device}\n")
    
    # Systems
    systems = [DuffingOscillator(), VanDerPolOscillator()]
    
    # Get experiment configurations
    configs = get_experiment_configs()
    
    print(f"Total configurations: {len(configs)}")
    print(f"Total systems: {len(systems)}")
    print(f"Total seeds per config: 100")
    print(f"Total experiments: {len(configs) * len(systems) * 100}\n")
    
    # Run experiments
    all_results, residuals_dict = run_experiments(systems, configs, n_seeds=100, device=device)
    
    # Generate outputs
    print(f"\n{'='*70}")
    print("GENERATING OUTPUTS")
    print(f"{'='*70}")
    
    generate_pareto_efficiency_plot(all_results, output_dir='parameter_ablation_results')
    generate_ablation_heatmaps(all_results, residuals_dict, output_dir='parameter_ablation_results')
    generate_final_table(all_results, output_dir='parameter_ablation_results')
    
    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == '__main__':
    results = main()
