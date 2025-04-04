import os
import time
import logging
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import warnings

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoNormal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter1d  
from scipy.stats import norm
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

@dataclass
class FeatureSelectionResult:
    selected_features: pd.Index
    num_variants: int
    total_variants: int
    credible_interval: float
    selected_credible_interval: Optional[float] = None
    selected_credible_interval_deviation: Optional[float] = None

class BayesianFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        unique_id: str,
        num_iterations=1000,
        lr=1e-4,
        credible_interval=0.95,
        num_samples=1000,
        batch_size=512,
        verbose=False,
        patience=20,
        covariance_type='independent',
        validation_split=0.2,
        checkpoint_path="checkpoint.params",
        max_features=200,
        min_threshold=0.2,  # Added min_threshold parameter
        use_mlflow=False,   # Flag to control MLflow usage
        k_neighbors: int = 10,  # NEW: Number of neighbors for tail detection
        smooth_window: int = 1,  # NEW: Window size for smoothing
        base_cumulative_density_threshold: float = 0.005,  # NEW: Cumulative density threshold
        base_sensitivity: float = 0.005  # NEW: Sensitivity for gradient detection
    ):
        self.unique_id = unique_id
        self.logger = logging.getLogger(__name__)
        self.num_iterations = num_iterations
        self.lr = lr
        self.covariance_type = covariance_type
        self.credible_interval = credible_interval
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.validation_split = validation_split
        self.max_features = max_features
        self.min_threshold = min_threshold
        self.checkpoint_path = f"{os.path.splitext(checkpoint_path)[0]}_{unique_id}.params"
        self.num_draws = num_iterations  # Total number of validation predictions we'll store
        self.use_mlflow = use_mlflow
        self.k_neighbors = k_neighbors
        self.smooth_window = smooth_window
        self.base_cumulative_density_threshold = base_cumulative_density_threshold
        self.base_sensitivity = base_sensitivity
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.verbose:
                print("Using CUDA (GPU) acceleration.")
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                print("Using CPU.")
        
        self.logger.info(f"Using device: {self.device}")

    def _log_to_mlflow(self, metrics=None, params=None, artifacts=None):
        """Helper method to handle MLflow logging with graceful fallback"""
        import mlflow
        if not self.use_mlflow:
            return

        try:
            if metrics:
                mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params(params)
            if artifacts:
                for name, artifact in artifacts.items():
                    mlflow.log_artifact(artifact)
        except ImportError:
            warnings.warn("MLflow not installed. Skipping logging.")
        except Exception as e:
            warnings.warn(f"MLflow logging failed: {str(e)}")

    def fit(self, X, y):
        # Store the scaler for later use
        self.scaler = MinMaxScaler()
        self.scaler.fit(X.values if isinstance(X, pd.DataFrame) else X)
        
        # Shuffle input data while maintaining index alignment
        rng = np.random.RandomState(int(self.unique_id))
        shuffle_idx = rng.permutation(len(X))
        
        if isinstance(X, pd.DataFrame):
            X = X.iloc[shuffle_idx]
            y = y.iloc[shuffle_idx]
        else:
            X = X[shuffle_idx]
            y = y[shuffle_idx]
        
        # Convert to tensors
        X_tensor = torch.tensor(
            self.scaler.transform(X.values) if isinstance(X, pd.DataFrame) else self.scaler.transform(X),
            dtype=torch.float64,
            device=self.device
        )
        y_tensor = torch.tensor(
            y.values if hasattr(y, 'values') else y,
            dtype=torch.float64,
            device=self.device
        )
        
        split_idx = int(X_tensor.size(0) * (1 - self.validation_split))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        # --- Balanced Mini-Batches Start ---
        # Use a WeightedRandomSampler to balance the classes within each mini-batch
        # Convert training labels to numpy for calculating class frequencies
        labels_array = y_train.cpu().numpy().astype(int)
        # Compute class counts (assuming binary classification: 0 and 1)
        class_counts = np.bincount(labels_array)
        # Compute weights: inverse frequency for each class
        class_weights = 1. / class_counts
        # Assign weight to each sample based on its label
        sample_weights = class_weights[labels_array]
        # Create sampler: replacement=True ensures we can sample repeatedly over the smaller class.
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Use the sampler instead of shuffle=True for balanced sampling
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            sampler=sampler  # balanced mini-batches courtesy of the sampler
        )
        # --- Balanced Mini-Batches End ---

        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Initialize validation_results_ with numpy arrays instead of lists
        # Approach 1: Store predictions as a 2D array (samples × draws)
        sample_ids = y.index.values if hasattr(y, 'index') else None
        if sample_ids is not None:
            val_sample_ids = sample_ids[split_idx:]
            self.validation_results_ = pd.DataFrame(index=val_sample_ids)
            self.validation_results_['label'] = y_val.cpu().numpy()
            self.validation_results_['sum_correct'] = 0
            self.validation_results_['count'] = 0
            # Initialize predictions array
            self.val_predictions_ = np.zeros((len(val_sample_ids), self.num_draws), dtype=np.int8)
            self.current_draw_ = 0

        def model(X, y=None):
            D = X.size(1)
            
            tau_0 = pyro.sample('tau_0', dist.HalfCauchy(scale=1.0))
            tau_0 = tau_0.to(self.device)
            
            lam = pyro.sample('lam', dist.HalfCauchy(scale=1.0).expand([D]).to_event(1))
            lam = lam.to(self.device)
            
            c2 = pyro.sample('c2', dist.InverseGamma(concentration=1.0, rate=1.0).expand([D]).to_event(1))
            c2 = c2.to(self.device)
            
            sigma = tau_0 * lam * torch.sqrt(c2)
            
            beta = pyro.sample(
                'beta',
                dist.Normal(torch.zeros(D, dtype=torch.float64, device=self.device), sigma).to_event(1)
            )
            intercept = pyro.sample('intercept', dist.Normal(0., 10.))
            logits = intercept + X @ beta

            with pyro.plate('data', X.size(0)):
                pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

        guide = AutoNormal(model)

        optimizer = ClippedAdam({"lr": self.lr, "clip_norm": 1.0})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        best_val_loss = float("inf")
        patience_counter = 0

        def compute_proba(current_params, X):
            beta = current_params['AutoNormal.locs.beta']
            intercept = current_params['AutoNormal.locs.intercept']
            logits = intercept + torch.matmul(X, beta)
            proba = torch.sigmoid(logits)
            return proba.detach().cpu().numpy()

        if self.verbose:
            pbar = tqdm(
                range(1, self.num_iterations + 1),
                desc="Training",
                unit="iter",
                ncols=150  # Wider to accommodate more text
            )
        else:
            pbar = range(1, self.num_iterations + 1)

        for epoch in pbar:
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                loss = svi.step(X_batch, y_batch)
                epoch_loss += loss

            avg_epoch_loss = epoch_loss / len(train_loader.dataset)

            val_loss = 0.0
            for X_val_batch, y_val_batch in val_loader:
                val_loss += svi.evaluate_loss(X_val_batch, y_val_batch)
            avg_val_loss = val_loss / len(val_loader.dataset)
            
            current_params = {k: v.clone().detach() for k, v in pyro.get_param_store().items()}
            y_val_pred_prob = compute_proba(current_params, X_val)
            y_val_pred = (y_val_pred_prob >= 0.5).astype(int)
            
            accuracy = accuracy_score(y_val.cpu().numpy(), y_val_pred)
            f1 = f1_score(y_val.cpu().numpy(), y_val_pred)
            
            if self.use_mlflow:
                self.log_metrics(epoch, avg_epoch_loss, avg_val_loss, accuracy, f1)

            if self.validation_results_ is not None:
                correct = (y_val_pred == y_val.cpu().numpy()).astype(int)
                self.validation_results_.loc[val_sample_ids, 'sum_correct'] += correct
                self.validation_results_.loc[val_sample_ids, 'count'] += 1
                # Store predictions in the array
                if self.current_draw_ < self.num_draws:
                    self.val_predictions_[:, self.current_draw_] = y_val_pred
                    self.current_draw_ += 1

            if self.verbose:
                status = {
                    'train_loss': f'{avg_epoch_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}',
                    'acc': f'{accuracy:.4f}',
                    'f1': f'{f1:.4f}'
                }
                
                if avg_val_loss < best_val_loss:
                    status['checkpoint'] = '✓'  # Checkmark symbol
                
                pbar.set_postfix(status)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                pyro.get_param_store().save(self.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

        pyro.get_param_store().load(self.checkpoint_path)

        predictive = Predictive(model, guide=guide, num_samples=self.num_samples, return_sites=['beta'])
        posterior_samples = predictive(X_train, y_train)
        beta_samples = posterior_samples['beta'].detach().cpu().numpy()

        if beta_samples.ndim == 3:
            beta_samples = beta_samples.squeeze(1)
        elif beta_samples.ndim == 2 and beta_samples.shape[1] == 1:
            beta_samples = beta_samples.squeeze(1)

        # Aggregate per-sample metrics and log to MLflow
        if self.validation_results_ is not None:
            # Use only the draws we actually completed
            actual_predictions = self.val_predictions_[:, :self.current_draw_]
            
            # Calculate metrics
            self.validation_results_['accuracy'] = self.validation_results_['sum_correct'] / self.validation_results_['count']
            self.validation_results_['accuracy_std'] = np.sqrt(
                self.validation_results_['accuracy'] * (1 - self.validation_results_['accuracy']) / self.validation_results_['count']
            )
            self.validation_results_['draw_count'] = self.validation_results_['count']
            
            # Calculate average and std dev of predictions for each sample
            self.validation_results_['avg_prediction'] = np.mean(actual_predictions, axis=1)
            self.validation_results_['prediction_std'] = np.std(actual_predictions, axis=1)
            
            validation_results_table = self.validation_results_.reset_index(names='sample_id')
            validation_results_table = validation_results_table[['sample_id', 'label', 'accuracy', 'accuracy_std', 'avg_prediction', 'prediction_std', 'draw_count']]

        self.logger.info("Detecting data-driven threshold for feature selection.")
        try:
            # Compute posterior mean and standard deviation
            posterior_mean = np.mean(beta_samples, axis=0)
            posterior_std = np.std(beta_samples, axis=0)
            # Handle zero standard deviation to avoid division by zero
            posterior_std = np.where(posterior_std == 0, np.finfo(float).eps, posterior_std)
            # Compute standardized effect sizes
            standardized_effect_sizes = np.abs(posterior_mean / posterior_std)

            selected_features, threshold, fig = select_features_by_tail(
                standardized_effect_sizes,
                feature_names=X.columns if isinstance(X, pd.DataFrame) else None,
                max_features=self.max_features,
                k_neighbors=self.k_neighbors,
                smooth_window=self.smooth_window,
                base_cumulative_density_threshold=self.base_cumulative_density_threshold,
                base_sensitivity=self.base_sensitivity
            )

            self.selected_credible_interval = threshold
            self.selected_features_ = selected_features
            self.tail_fig = fig

             # Log validation results to MLflow  
            if self.use_mlflow:
                self._log_to_mlflow(
                    metrics={
                        'validation_loss': val_loss,
                        'num_selected_features': len(selected_features),
                        'feature_selection_threshold': threshold
                    },
                    params={
                        'num_iterations': self.num_iterations,
                        'learning_rate': self.lr,
                        'batch_size': self.batch_size
                    },
                    artifacts={
                        'validation_aggregated_results.json': validation_results_table
                    }
                )
                
            # Log the figure to MLflow
            if self.verbose and self.use_mlflow:
                import mlflow
                mlflow.log_figure(fig, f"standardized_effect_sizes_tail.png")
                plt.close(fig)

            # Create and log feature statistics table
            # Compute standardized effect sizes once
            beta_mean = np.mean(beta_samples, axis=0)
            beta_std = np.std(beta_samples, axis=0)
            beta_std = np.where(beta_std == 0, np.finfo(float).eps, beta_std)
            feature_stats = pd.DataFrame({
                'feature_name': X.columns if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(len(beta_mean))],
                'beta_mean': beta_mean,
                'beta_std': beta_std,
                'effect_size': standardized_effect_sizes,
                'is_selected': [name in selected_features for name in (X.columns if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(len(beta_mean))])]
            })
            
            # Sort by absolute effect size for better readability
            self.feature_stats = feature_stats.sort_values('effect_size', ascending=False)
            
            # Log to MLflow
            self._log_to_mlflow(
                metrics={
                    'num_selected_features': len(selected_features),
                    'feature_selection_threshold': threshold
                },
                params={
                    'num_iterations': self.num_iterations,
                    'learning_rate': self.lr,
                    'batch_size': self.batch_size
                },
                artifacts={
                    'feature_statistics.json': self.feature_stats
                }
            )
                
        except Exception as e:
            self.logger.error(f"Tail detection failed: {e}")
            raise ValueError("Feature selection using the tail approach failed.")
        finally:
            pyro.clear_param_store()

        return self

    def transform(self, X):
        if not hasattr(self, 'selected_features_'):
            raise ValueError("The model has not been fitted yet.")
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X[:, self.selected_features_]

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float, accuracy: float, f1: float):
        """Log metrics with proper nesting under the feature selection run."""
        import mlflow
        # Get the current run context
        current_run = mlflow.active_run()
        if current_run is None:
            self.logger.warning("No active MLflow run found for logging metrics")
            return

        # Log metrics directly without creating a new run
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)
        mlflow.log_metric("val_f1_score", f1, step=epoch)
        
        if epoch == 1:
            mlflow.log_params({
                "unique_id": self.unique_id,
                "covariance_type": self.covariance_type,
                "num_iterations": self.num_iterations,
                "learning_rate": self.lr,
                "credible_interval": self.credible_interval,
                "num_samples": self.num_samples,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "validation_split": self.validation_split,
                "checkpoint_path": self.checkpoint_path
            })

    def predict_proba(self, X, use_all_features=True):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to predict on
        use_all_features : bool, default=False
            If True, use all features instead of only selected features
        
        Returns:
        --------
        array-like
            Probability predictions
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Model has not been fitted yet.")
        
        # Convert and select features if needed
        if isinstance(X, pd.DataFrame):
            X = X if use_all_features else X[self.selected_features_]
        else:
            X = X if use_all_features else X[:, self.selected_features_]
        
        # Scale the input
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64, device=self.device)
        
        # Load the best model parameters
        pyro.get_param_store().load(self.checkpoint_path)
        
        # Get the current parameters
        current_params = {k: v.clone().detach() for k, v in pyro.get_param_store().items()}
        beta = current_params['AutoNormal.locs.beta']
        intercept = current_params['AutoNormal.locs.intercept']
        
        # Compute probabilities
        with torch.no_grad():
            logits = intercept + torch.matmul(X_tensor, beta)
            probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy()

    def predict(self, X, use_all_features=True):
        """
        Predict class labels for X.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to predict on
        use_all_features : bool, default=False
            If True, use all features instead of only selected features
        
        Returns:
        --------
        array-like
            Binary predictions [0, 1]
        """
        probs = self.predict_proba(X, use_all_features=use_all_features)
        return (probs >= 0.5).astype(int)

    def score(self, X, y, use_all_features=False):
        """
        Return the accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to score on
        y : array-like
            True labels
        use_all_features : bool, default=False
            If True, use all features instead of only selected features
        
        Returns:
        --------
        float
            Accuracy score
        """
        return accuracy_score(y, self.predict(X, use_all_features=use_all_features))

def detect_beta_tail_threshold(effect_sizes: np.ndarray, 
                             k_neighbors: int = 10,
                             smooth_window: int = 1,
                             base_cumulative_density_threshold: float = 0.005,
                             base_sensitivity: float = 0.005) -> tuple:
    """
    Detect a threshold in the KDE using global density normalization to dynamically
    adjust the cumulative density threshold based on the density distribution.

    Parameters:
    -----------
    effect_sizes : np.ndarray
        Array of pre-computed standardized effect sizes
    k_neighbors : int
        Number of neighbors for adaptive bandwidth calculation
    smooth_window : int
        Window size for smoothing the KDE curve
    base_cumulative_density_threshold : float
        Base cumulative density threshold (adjusted dynamically based on global density)
    base_sensitivity : float
        Base gradient sensitivity (adjusted dynamically based on global density)

    Returns:
    --------
    tuple
        (threshold value, Figure object with plot)
    """
    # Create evaluation grid
    x_grid = np.linspace(effect_sizes.min(), effect_sizes.max(), 1000)
    
    # Calculate adaptive bandwidths
    nbrs = NearestNeighbors(n_neighbors=k_neighbors)
    nbrs.fit(effect_sizes.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(effect_sizes.reshape(-1, 1))
    bandwidths = distances[:, -1]
    
    # Estimate KDE with adaptive bandwidths
    kde_values = np.zeros_like(x_grid)
    for i, point in enumerate(effect_sizes):
        kde_values += norm.pdf(x_grid, loc=point, scale=bandwidths[i])
    kde_values /= len(effect_sizes)  # Average the contributions
    
    # Apply smoothing directly to kde_values
    kde_smooth = gaussian_filter1d(kde_values, sigma=smooth_window)
    
    # Calculate cumulative density (using proper integration)
    dx = x_grid[1] - x_grid[0]  # grid spacing
    cumulative_density = np.cumsum(kde_smooth[::-1] * dx)[::-1]
    
    # Find start of tail region using cumulative density
    tail_indices = np.where(cumulative_density <= base_cumulative_density_threshold)[0]
    if len(tail_indices) > 0:
        tail_start_idx = tail_indices[0]
    else:
        # Fallback: use the smallest cumulative density point
        tail_start_idx = len(cumulative_density) - 1  # Last index
    
    # Calculate gradient and its smoothed version
    gradient = np.gradient(kde_smooth)
    gradient_smooth = gaussian_filter1d(gradient, sigma=smooth_window)
    
    # Find inflection point in tail region
    threshold_idx = None
    window_points = 5  # Window for checking gradient stability
    
    # Search only in tail region
    for i in range(tail_start_idx, len(gradient_smooth) - window_points):
        window = gradient_smooth[i:i + window_points]
        
        # Check for sustained negative gradient
        if (np.mean(window) < -base_sensitivity and 
            np.all(window < 0)):
            threshold_idx = i
            break
    
    if threshold_idx is None:
        # Fallback: use tail start point
        threshold_idx = tail_start_idx
    
    threshold = x_grid[threshold_idx]
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Plot histogram and rescale KDE to match histogram density
    hist_values, bin_edges = np.histogram(effect_sizes, bins=50, density=True)
    scaling_factor = np.max(hist_values) / np.max(kde_smooth)
    scaled_kde = kde_smooth * scaling_factor
    
    ax1.hist(effect_sizes, bins=50, density=True, alpha=0.3, color='gray', label='Histogram')
    ax1.plot(x_grid, scaled_kde, 'b-', label='Smoothed KDE')
    ax1.axvline(x=threshold, color='r', linestyle='--', 
                label=f'Inflection Threshold: {threshold:.3f}')
    
    # Mark tail region and inflection point
    ax1.axvline(x=x_grid[tail_start_idx], color='g', linestyle=':', 
                label=f'Tail Start ({base_cumulative_density_threshold:.1%} density)')
    ax1.plot(x_grid[threshold_idx], scaled_kde[threshold_idx], 'ko', 
             markersize=10, label='Inflection Point')
    
    ax1.set_title('Distribution of Standardized Effect Sizes')
    ax1.set_xlabel('|Standardized Effect Size|')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # Plot gradient analysis
    ax2.plot(x_grid, gradient_smooth, 'b-', label='Smoothed Gradient')
    ax2.axhline(y=0, color='gray', linestyle=':')
    ax2.axvline(x=threshold, color='r', linestyle='--', label='Selected Threshold')
    ax2.axvline(x=x_grid[tail_start_idx], color='g', linestyle=':', label='Tail Start')
    ax2.plot(x_grid[threshold_idx], gradient_smooth[threshold_idx], 'ko',
             markersize=10, label='Inflection Point')
    ax2.set_title('Gradient Analysis')
    ax2.set_xlabel('|Standardized Effect Size|')
    ax2.set_ylabel('Gradient')
    ax2.legend()
    
    plt.tight_layout()
    
    return threshold, fig

def select_features_by_tail(effect_sizes: np.ndarray,
                          feature_names=None, 
                          max_features=None, 
                          k_neighbors: int = 10,
                          smooth_window: int = 1,
                          base_cumulative_density_threshold: float = 0.005,
                          base_sensitivity: float = 0.005) -> tuple:
    """
    Select features based on pre-computed standardized effect sizes.
    
    Parameters:
    -----------
    effect_sizes : np.ndarray
        Pre-computed standardized effect sizes
    feature_names : array-like, optional
        Names of features corresponding to effect sizes
    max_features : int, optional
        Maximum number of features to select
    k_neighbors : int
        Number of neighbors for adaptive bandwidth calculation
    smooth_window : int
        Window size for smoothing the KDE curve
    base_cumulative_density_threshold : float
        Base cumulative density threshold (adjusted dynamically based on global density)
    base_sensitivity : float
        Base gradient sensitivity (adjusted dynamically based on global density)
    
    Returns:
    --------
    tuple
        (selected features, threshold value, Figure object)
    """
    threshold, fig = detect_beta_tail_threshold(effect_sizes, k_neighbors, smooth_window, base_cumulative_density_threshold, base_sensitivity)
    
    # Select features above threshold
    selected = effect_sizes >= threshold
    
    # Apply max_features constraint if specified
    if max_features is not None and np.sum(selected) > max_features:
        top_indices = np.argsort(-effect_sizes)[:max_features]
        selected = np.zeros_like(effect_sizes, dtype=bool)
        selected[top_indices] = True
    
    # Return feature names if provided, otherwise indices
    if feature_names is not None:
        selected_features = feature_names[selected]
    else:
        selected_features = np.where(selected)[0]
    
    return selected_features, threshold, fig
