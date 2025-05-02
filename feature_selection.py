import os
import time
import logging
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import warnings
import joblib # Added
import json   # Added

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Keep this for validation metrics
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit # Sigmoid function
from scipy.stats import beta

# Conditional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Define dummy mlflow object if mlflow is not available
    class DummyMlflow:
        def __getattr__(self, name):
            def dummy_method(*args, **kwargs):
                # warnings.warn(f"MLflow not installed. Call to mlflow.{name} ignored.")
                pass # Silently ignore if mlflow is not installed
            return dummy_method
    mlflow = DummyMlflow()

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
        self.base_checkpoint_dir = "bfs_checkpoints" # Base directory for saving
        self.save_dir = os.path.join(self.base_checkpoint_dir, self.unique_id) # Directory for this specific run
        os.makedirs(self.save_dir, exist_ok=True) # Create the directory
        self.num_draws = num_iterations  # Total number of validation predictions we'll store
        self.use_mlflow = use_mlflow
        self.k_neighbors = k_neighbors
        self.smooth_window = smooth_window
        self.base_cumulative_density_threshold = base_cumulative_density_threshold
        self.base_sensitivity = base_sensitivity
        self._best_y_true = None
        self._best_y_pred = None
        self._best_y_proba = None
        self.best_validation_sample_details_ = None
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.verbose:
                print("Using CUDA (GPU) acceleration.")
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                print("Using CPU.")
        
        self.logger.info(f"Using device: {self.device}")

        # Initialize attributes that will be loaded/saved
        self.scaler = None
        self.feature_names_ = None
        self.selected_features_ = None

    def _log_to_mlflow(self, metrics=None, params=None, artifacts=None):
        """Helper method to handle MLflow logging with graceful fallback"""
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
        # Store feature names if X is a DataFrame
        self.feature_names_ = X.columns if isinstance(X, pd.DataFrame) else pd.Index([f'feature_{i}' for i in range(X.shape[1])])

        # Shuffle input data while maintaining index alignment
        try:
            timestamp_str = self.unique_id.split('_')[-1]
            seed = int(timestamp_str)
        except (IndexError, ValueError):
            warnings.warn(f"Warning: Could not parse timestamp from unique_id '{self.unique_id}'. Using current time for seeding.")
            seed = int(time.time())
        rng = np.random.RandomState(seed)
        shuffle_idx = rng.permutation(len(X))

        # Ensure X and y are consistently handled (DataFrames or arrays)
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_np = y.values if isinstance(y, pd.Series) else np.asarray(y)

        X_shuffled = X_np[shuffle_idx]
        y_shuffled = y_np[shuffle_idx]

        # --- Split Data BEFORE Scaling ---
        split_idx = int(len(X_shuffled) * (1 - self.validation_split))
        X_train_raw, X_val_raw = X_shuffled[:split_idx], X_shuffled[split_idx:]
        y_train_np, y_val_np = y_shuffled[:split_idx], y_shuffled[split_idx:] # Keep y as numpy for now

        # --- Fit Scaler ONLY on Training Data ---
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train_raw)

        # --- Scale Both Train and Validation Data ---
        X_train_scaled = self.scaler.transform(X_train_raw)
        X_val_scaled = self.scaler.transform(X_val_raw)

        # Convert to tensors AFTER scaling and splitting
        X_train = torch.tensor(X_train_scaled, dtype=torch.float64, device=self.device)
        X_val = torch.tensor(X_val_scaled, dtype=torch.float64, device=self.device)
        y_train = torch.tensor(y_train_np, dtype=torch.float64, device=self.device)
        y_val = torch.tensor(y_val_np, dtype=torch.float64, device=self.device) # Now convert y_val

        # --- Balanced Mini-Batches Start ---
        # Use a WeightedRandomSampler based on the training labels
        labels_array = y_train.cpu().numpy().astype(int) # Use y_train tensor
        class_counts = np.bincount(labels_array)
        # Handle cases where a class might be missing in the training split (unlikely but possible)
        if len(class_counts) < 2:
             warnings.warn("Training split contains only one class. WeightedRandomSampler might not behave as expected.")
             # Fallback: Use standard sampling or raise error? For now, proceed but weights might be off.
             class_weights = np.ones(2) # Dummy weights
        else:
             class_weights = 1. / np.maximum(class_counts, 1) # Avoid division by zero if counts are 0

        sample_weights = class_weights[labels_array]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), # Use scaled X_train tensor
            batch_size=self.batch_size,
            sampler=sampler
        )
        # --- Balanced Mini-Batches End ---

        # val_loader is not used, can be removed or kept commented out
        # val_loader = DataLoader(
        #     TensorDataset(X_val, y_val), # Use scaled X_val tensor
        #     batch_size=self.batch_size,
        #     shuffle=False
        # )

        # --- Pyro Setup ---
        pyro.clear_param_store()
        num_features = X_train.shape[1] # Get shape from scaled train tensor

        def pyro_model(X, y=None):
            num_features = X.size(1) # Get num_features from input tensor

            # --- Horseshoe Prior Definition ---
            tau_0 = pyro.sample('tau_0', dist.HalfCauchy(scale=torch.tensor(1.0, device=self.device, dtype=torch.float64)))
            lam = pyro.sample('lam', dist.HalfCauchy(scale=torch.ones(num_features, device=self.device, dtype=torch.float64)).to_event(1))
            c2 = pyro.sample('c2', dist.InverseGamma(concentration=torch.ones(num_features, device=self.device, dtype=torch.float64),
                                                    rate=torch.ones(num_features, device=self.device, dtype=torch.float64)).to_event(1))
            sigma = tau_0 * lam * torch.sqrt(c2)

            beta = pyro.sample(
                'beta',
                dist.Normal(torch.zeros(num_features, dtype=torch.float64, device=self.device), sigma).to_event(1)
            )
            intercept = pyro.sample('intercept', dist.Normal(torch.tensor(0., device=self.device, dtype=torch.float64),
                                                              torch.tensor(10., device=self.device, dtype=torch.float64)))

            logits = intercept + X @ beta

            with pyro.plate('data', X.size(0)):
                pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

        guide = AutoNormal(pyro_model)

        # --- SVI Training Loop ---
        optimizer = ClippedAdam({"lr": self.lr, "clip_norm": 10.0}) # Pyro optimizer
        svi = SVI(pyro_model, guide, optimizer, loss=Trace_ELBO())

        # Initialize storage for validation results
        self.validation_results_ = [] # Store metrics from validation steps

        best_val_loss = float('inf')
        patience_counter = 0

        # Start MLflow run if enabled
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=f"BFS_{self.unique_id}")
            # Log hyperparameters
            params_to_log = {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, str, bool)) and k != 'unique_id' and k!= 'logger'} # Exclude logger
            # Add save_dir to logged params
            params_to_log['save_dir'] = self.save_dir
            mlflow.log_params(params_to_log)

        start_time = time.time()

        for iter_num in tqdm(range(self.num_iterations), desc="SVI Iterations", disable=not self.verbose):
            epoch_loss = 0.0
            num_batches = 0
            for X_batch, y_batch in train_loader: # Use DataLoader for batching
                 # Ensure batches are on the correct device and dtype
                 X_batch = X_batch.to(device=self.device, dtype=torch.float64)
                 y_batch = y_batch.to(device=self.device, dtype=torch.float64)

                 # Pass necessary args to model and guide
                 loss = svi.step(X_batch, y_batch) # Removed reg_lambda and sensitivity args
                 epoch_loss += loss
                 num_batches += 1

            epoch_loss /= (len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1) # Normalize loss by dataset size


            # --- Validation and Logging ---
            # Perform validation periodically
            # verbose_freq needs to be defined in __init__, e.g., self.verbose_freq = 10
            if not hasattr(self, 'verbose_freq'): self.verbose_freq = 10 # Default if not set
            if (iter_num + 1) % self.verbose_freq == 0 or iter_num == self.num_iterations - 1:
                 # Use Predictive to get samples from the posterior predictive distribution
                 # Only request 'obs' site, as '_RETURN' is None for this model
                 predictive = Predictive(pyro_model, guide=guide, num_samples=100, return_sites=("obs",)) # Get 100 posterior samples for 'obs' # Pass restored model name

                 with torch.no_grad():
                      # Pass the correctly scaled validation tensor X_val
                      posterior_samples = predictive(X_val) # Use the pre-split, pre-scaled X_val
                      # posterior_samples['obs'] shape: (num_samples, num_data)
                      # Average probabilities over samples
                      # Use the posterior samples from predictive which are based on logits
                      # Note: posterior_samples['obs'] might contain sampled outcomes (0/1) or logits depending on Pyro version/model.
                      # Assuming it gives logits or something transformable to probability:
                      # If posterior_samples['obs'] are logits:
                      # y_pred_proba_val = torch.sigmoid(posterior_samples['_RETURN']).mean(0).cpu().numpy() # Often _RETURN contains logits/params
                      # If posterior_samples['obs'] are sampled 0/1s:
                      y_pred_proba_val = posterior_samples['obs'].float().mean(0).cpu().numpy() # Average samples to get proba

                      y_pred_val = (y_pred_proba_val > 0.5).astype(int)
                      # Use the correctly split validation tensor y_val
                      y_true_val = y_val.cpu().numpy() # Use the pre-split y_val

                      # Calculate metrics using sklearn
                      accuracy = accuracy_score(y_true_val, y_pred_val)
                      precision, recall, f1, _ = precision_recall_fscore_support(
                           y_true_val, y_pred_val, average='binary', zero_division=0
                      )
                      # Calculate validation loss (e.g., average log loss)
                      # Use logits directly if available, otherwise calculate from probabilities
                      # We don't have direct logits here easily, let's use prediction probability
                      # A simple proxy: negative log-likelihood (using mean probability)
                      # This is not the ELBO loss, but a classification performance metric
                      val_loss = -np.mean(y_true_val * np.log(y_pred_proba_val + 1e-9) + (1 - y_true_val) * np.log(1 - y_pred_proba_val + 1e-9))


                 # Store metrics
                 current_metrics = {
                      'iteration': iter_num + 1,
                      'train_loss (ELBO)': epoch_loss, # Note: This is ELBO loss
                      'val_loss (NLL)': val_loss,      # Note: This is NLL/Cross-Entropy
                      'val_accuracy': accuracy,
                      'val_precision': precision,
                      'val_recall': recall,
                      'val_f1': f1
                 }
                 self.validation_results_.append(current_metrics)

                 if self.verbose:
                      print(f"Iter {iter_num+1}/{self.num_iterations} - ELBO Loss: {epoch_loss:.4f}, Val NLL: {val_loss:.4f}, Val Acc: {accuracy:.4f}, Val F1: {f1:.4f}")

                 # Log metrics to MLflow if enabled
                 if self.use_mlflow and MLFLOW_AVAILABLE:
                      mlflow.log_metric("train_loss_elbo", epoch_loss, step=iter_num + 1)
                      mlflow.log_metric("val_loss_nll", val_loss, step=iter_num + 1)
                      mlflow.log_metric("val_accuracy", accuracy, step=iter_num + 1)
                      mlflow.log_metric("val_precision", precision, step=iter_num + 1)
                      mlflow.log_metric("val_recall", recall, step=iter_num + 1)
                      mlflow.log_metric("val_f1", f1, step=iter_num + 1)

                 # Early Stopping Check (based on validation NLL loss)
                 if val_loss < best_val_loss:
                      best_val_loss = val_loss
                      patience_counter = 0
                      # Store the sample-wise results for this best iteration
                      self._best_y_true = y_true_val
                      self._best_y_pred = y_pred_val
                      self._best_y_proba = y_pred_proba_val
                      # Save parameters using Pyro's param store
                      # pyro.get_param_store().save(self.checkpoint_path) # Old saving
                      # --- Save model state to directory ---
                      pyro_params_path = os.path.join(self.save_dir, "pyro_params.pt")
                      scaler_path = os.path.join(self.save_dir, "scaler.joblib")
                      feature_names_path = os.path.join(self.save_dir, "feature_names.json")

                      pyro.get_param_store().save(pyro_params_path)
                      joblib.dump(self.scaler, scaler_path)
                      # Convert pd.Index to list for JSON serialization
                      feature_names_list = self.feature_names_.tolist()
                      with open(feature_names_path, 'w') as f:
                          json.dump(feature_names_list, f)
                      # --- End save ---
                 else:
                      patience_counter += 1
                      if patience_counter >= self.patience:
                           if self.verbose:
                                print(f"Early stopping triggered at iteration {iter_num + 1}")
                           if self.use_mlflow and MLFLOW_AVAILABLE:
                                mlflow.set_tag("early_stopping", "True")
                           break # Exit training loop

        # Create DataFrame with best sample-wise validation results if available
        if self._best_y_true is not None:
            self.best_validation_sample_details_ = pd.DataFrame({
                'true_label': self._best_y_true,
                'predicted_label': self._best_y_pred,
                'predicted_probability': self._best_y_proba,
                'is_correct': (self._best_y_true == self._best_y_pred)
            })
            # Clean up temporary storage
            del self._best_y_true, self._best_y_pred, self._best_y_proba
        else:
            warnings.warn("Could not create best_validation_sample_details_ as no best validation state was recorded.")
            self.best_validation_sample_details_ = None # Explicitly set to None


        # --- Feature Selection Post-Training ---
        # Load the best parameters found during training
        try:
            self._load_model_state() # Use the new loading method
            # # pyro.get_param_store().load(self.checkpoint_path) # Original call causing UnpicklingError
            # pyro_params_path = os.path.join(self.save_dir, "pyro_params.pt")
            # scaler_path = os.path.join(self.save_dir, "scaler.joblib")
            # feature_names_path = os.path.join(self.save_dir, "feature_names.json")

            # with open(pyro_params_path, "rb") as f:
            #      # Explicitly set weights_only=False if needed by your Pyro/Torch version
            #      loaded_state = torch.load(f, map_location=self.device) # Removed weights_only=False for now, add back if needed
            # pyro.get_param_store().set_state(loaded_state)

            # self.scaler = joblib.load(scaler_path)
            # with open(feature_names_path, 'r') as f:
            #     feature_names_list = json.load(f)
            # self.feature_names_ = pd.Index(feature_names_list)

        except FileNotFoundError:
             warnings.warn(f"Checkpoint files not found in {self.save_dir}. Using parameters from the last iteration (if available) and model may be unusable.")
             # If loading fails, ensure attributes are in a consistent state (e.g., None)
             # self.scaler = None # Should be handled by _load_model_state or initial state
             # self.feature_names_ = None
             # pyro.clear_param_store() # Clear potentially partially loaded state? Or rely on last iter state.
        except Exception as e:
             warnings.warn(f"Failed to load checkpoint from {self.save_dir}: {e}. Using parameters from the last iteration (if available).")
             # Similar handling as FileNotFoundError

        # Ensure scaler and feature names are loaded before proceeding
        if self.scaler is None or self.feature_names_ is None:
             warnings.warn("Scaler or feature names could not be loaded. Feature selection and transformation might fail.")
             # Assign empty index to prevent downstream errors if feature_names_ needed
             if self.feature_names_ is None: self.feature_names_ = pd.Index([])
        else:
            # Get posterior samples for beta coefficients using the final/best guide
            predictive = Predictive(pyro_model, guide=guide, num_samples=self.num_samples, return_sites=['beta']) # Pass restored model name
            # Need to pass dummy data that matches expected input shape for Predictive
            X_dummy_for_predictive = torch.zeros((1, len(self.feature_names_)), dtype=torch.float64, device=self.device) # Use loaded feature names count
            posterior_samples = predictive(X_dummy_for_predictive) # Removed reg_lambda and sensitivity args # Pass args model expects
            beta_samples = posterior_samples['beta'].detach().cpu().numpy() # Shape: (num_samples, 1, num_features) or (num_samples, num_features)

            # Squeeze dimensions if necessary, typical shape is (num_samples, num_features)
            if beta_samples.ndim == 3:
                beta_samples = beta_samples.squeeze(1) # This might need adjustment based on Horseshoe output shape

            # --- Tail-based Feature Selection ---
            self.logger.info("Detecting data-driven threshold for feature selection.")
            try:
                # Compute posterior mean and standard deviation
                posterior_mean = np.mean(beta_samples, axis=0)
                posterior_std = np.std(beta_samples, axis=0)
                # Handle zero standard deviation to avoid division by zero
                posterior_std = np.where(posterior_std == 0, np.finfo(float).eps, posterior_std)
                # Compute standardized effect sizes
                standardized_effect_sizes = np.abs(posterior_mean / posterior_std)

                # Use stored feature names from loaded state
                feature_names = self.feature_names_ # Use the loaded pd.Index

                selected_feature_names, threshold, fig = select_features_by_tail(
                    standardized_effect_sizes,
                    feature_names=feature_names,
                    max_features=self.max_features,
                    k_neighbors=self.k_neighbors,
                    smooth_window=self.smooth_window,
                    base_cumulative_density_threshold=self.base_cumulative_density_threshold,
                    base_sensitivity=self.base_sensitivity
                )

                self.selected_credible_interval = threshold # Store the threshold used
                self.selected_features_ = selected_feature_names # Store as pd.Index
                self.tail_fig = fig

                # Log the figure to MLflow
                if self.verbose and self.use_mlflow and MLFLOW_AVAILABLE:
                    mlflow.log_figure(fig, f"standardized_effect_sizes_tail_{self.unique_id}.png")
                    plt.close(fig) # Close the figure after logging

                # --- Feature Statistics ---
                # Compute standardized effect sizes once
                beta_mean = np.mean(beta_samples, axis=0)
                beta_std = np.std(beta_samples, axis=0)
                beta_std = np.where(beta_std == 0, np.finfo(float).eps, beta_std) # Avoid division by zero
                effect_size = np.abs(beta_mean / beta_std)

                # Create feature statistics DataFrame
                feature_stats = pd.DataFrame({
                    'feature_name': feature_names, # Use the stored Index
                    'beta_mean': beta_mean,
                    'beta_std': beta_std,
                    'effect_size': effect_size,
                    'is_selected': [name in self.selected_features_ for name in feature_names]
                })

                # Sort by absolute effect size for better readability
                self.feature_stats = feature_stats.sort_values('effect_size', ascending=False)

                # Log feature stats table to MLflow as artifact (e.g., CSV)
                if self.use_mlflow and MLFLOW_AVAILABLE:
                     stats_filename = f"feature_stats_{self.unique_id}.csv"
                     self.feature_stats.to_csv(stats_filename, index=False)
                     mlflow.log_artifact(stats_filename)
                     # Clean up local file after logging
                     os.remove(stats_filename)

                     # Log final metrics
                     mlflow.log_metric("final_num_selected_features", len(self.selected_features_))
                     mlflow.log_metric("final_selection_threshold", self.selected_credible_interval)


            except Exception as e:
                self.logger.error(f"Tail detection or feature stats calculation failed: {e}", exc_info=True) # Log traceback
                # Fallback or re-raise depending on desired behavior
                # Maybe select top N features based on mean beta as a fallback?
                # For now, let's store empty results and log the error
                self.selected_features_ = pd.Index([])
                self.feature_stats = pd.DataFrame()
                self.selected_credible_interval = None
                if self.use_mlflow and MLFLOW_AVAILABLE:
                     mlflow.set_tag("error", f"Feature selection failed: {e}")


        # End MLflow run if enabled and active
        if self.use_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
             mlflow.end_run()

        return self

    def _load_model_state(self):
        """Loads the pyro parameters, scaler, and feature names from the save directory."""
        pyro_params_path = os.path.join(self.save_dir, "pyro_params.pt")
        scaler_path = os.path.join(self.save_dir, "scaler.joblib")
        feature_names_path = os.path.join(self.save_dir, "feature_names.json")

        if not os.path.exists(pyro_params_path) or \
           not os.path.exists(scaler_path) or \
           not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"One or more required checkpoint files not found in {self.save_dir}")

        # Load Pyro params - using load directly, assuming it works. Fallback to torch.load if needed.
        pyro.clear_param_store() # Clear before loading
        pyro.get_param_store().load(pyro_params_path, map_location=self.device)
        # Alternative using torch.load if pyro.load fails:
        # with open(pyro_params_path, "rb") as f:
        #     loaded_state = torch.load(f, map_location=self.device) # Consider weights_only=False if needed
        # pyro.get_param_store().set_state(loaded_state)

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Load feature names
        with open(feature_names_path, 'r') as f:
            feature_names_list = json.load(f)
        self.feature_names_ = pd.Index(feature_names_list)

        self.logger.info(f"Successfully loaded model state from {self.save_dir}")

    def transform(self, X):
        if not hasattr(self, 'selected_features_') or self.selected_features_ is None:
             # Try loading state if not fitted or if selected_features_ is None
             try:
                 self._load_model_state()
                 # We might not have run feature selection yet if loading an older model,
                 # but selected_features_ should be present if fit completed.
                 # Re-check after loading.
                 if self.selected_features_ is None:
                      # This condition might occur if loading a model state saved *before*
                      # feature selection was run/stored. Handle appropriately.
                      # For now, raise error or log warning.
                      raise ValueError("Model state loaded, but selected features are not available. Was fit fully completed?")
             except Exception as e:
                 raise ValueError(f"Model has not been fitted or state could not be loaded: {e}")

        # Ensure X has the original features before selection and scaling
        if isinstance(X, pd.DataFrame):
             # Check if columns match the original feature names
             if not self.feature_names_.equals(X.columns):
                 # Attempt to reindex if columns are a subset or superset, or just error out
                 try:
                     # Ensure all original features are present before selecting
                     X_reindexed = X.reindex(columns=self.feature_names_, fill_value=0) # Or handle missing cols appropriately
                 except Exception as e:
                     raise ValueError(f"Input DataFrame columns do not match original features used for training: {e}")
             else:
                 X_reindexed = X # Columns match
             return X_reindexed[self.selected_features_]
        else:
             # Assuming numpy array has the correct columns in the original order
             if X.shape[1] != len(self.feature_names_):
                 raise ValueError(f"Input array has {X.shape[1]} columns, but model was trained with {len(self.feature_names_)} features.")
             # We need indices corresponding to selected_features_ within feature_names_
             selected_indices = [self.feature_names_.get_loc(feature) for feature in self.selected_features_]
             return X[:, selected_indices]

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float, accuracy: float, f1: float):
        """Log metrics with proper nesting under the feature selection run."""
        if self.use_mlflow:
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
        # Try to load state if scaler or feature_names are missing
        if self.scaler is None or self.feature_names_ is None:
            try:
                self._load_model_state()
            except Exception as e:
                 raise ValueError(f"Model needs to be fitted or state loaded before prediction: {e}")

        # Ensure selected_features_ is available if not using all features
        if not use_all_features and (not hasattr(self, 'selected_features_') or self.selected_features_ is None):
             raise ValueError("Selected features not available. Ensure fit completed or load a model where selection was performed.")

        X_input = X # Keep original for potential column checks

        # Convert and select features if needed
        if isinstance(X_input, pd.DataFrame):
            # Ensure columns match feature_names_ before potential selection/scaling
            if not self.feature_names_.equals(X_input.columns):
                 try:
                      X_input = X_input.reindex(columns=self.feature_names_, fill_value=0) # Ensure correct columns and order
                 except Exception as e:
                      raise ValueError(f"Input DataFrame columns do not match original features: {e}")

            X_processed = X_input if use_all_features else X_input[self.selected_features_]
            feature_subset_names = self.feature_names_ if use_all_features else self.selected_features_
        else: # Assuming numpy array
             if X_input.shape[1] != len(self.feature_names_):
                 raise ValueError(f"Input array has {X_input.shape[1]} columns, but model expects {len(self.feature_names_)}.")
             if use_all_features:
                 X_processed = X_input
                 feature_subset_names = self.feature_names_ # Not directly used, but conceptually represents the features
             else:
                 # Select columns based on selected_features_
                 selected_indices = [self.feature_names_.get_loc(feature) for feature in self.selected_features_]
                 X_processed = X_input[:, selected_indices]
                 feature_subset_names = self.selected_features_ # Conceptually

        # Scale the input (using potentially subsetted features)
        # Important: Scaler was fit on ALL original features. We need to transform using the same features.
        # Therefore, scaling must happen BEFORE feature selection for prediction.
        if isinstance(X, pd.DataFrame):
             if not self.feature_names_.equals(X.columns):
                  try:
                       X_reindexed = X.reindex(columns=self.feature_names_, fill_value=0)
                  except Exception as e:
                       raise ValueError(f"Input DataFrame columns mismatch for scaling: {e}")
             else:
                  X_reindexed = X
             X_scaled_all_features = self.scaler.transform(X_reindexed) # Scale using all features scaler expects
        else: # Numpy array
             if X.shape[1] != len(self.feature_names_):
                  raise ValueError(f"Input array column count mismatch for scaling ({X.shape[1]} vs {len(self.feature_names_)}).")
             X_scaled_all_features = self.scaler.transform(X)

        # Now select features *after* scaling if needed
        if use_all_features:
            X_scaled_processed = X_scaled_all_features
            num_processed_features = len(self.feature_names_)
        else:
            selected_indices = [self.feature_names_.get_loc(feature) for feature in self.selected_features_]
            X_scaled_processed = X_scaled_all_features[:, selected_indices]
            num_processed_features = len(self.selected_features_)

        X_tensor = torch.tensor(X_scaled_processed, dtype=torch.float64, device=self.device)

        # Load the best model parameters (already handled by _load_model_state call at start)
        # We need the guide parameters from the loaded state.
        # Option 1: Use Predictive (more Bayesian)
        # Requires the pyro_model definition and the guide instance.
        # This might be complex if the guide object itself isn't easily reconstituted.

        # Option 2: Use point estimate (mean of posterior) - current approach
        # Get the current parameters from the loaded param store state
        current_params = pyro.get_param_store() # Access the already loaded store
        beta_locs = current_params['AutoNormal.locs.beta'] # Changed prefix to AutoNormal
        intercept_loc = current_params['AutoNormal.locs.intercept'] # Changed prefix to AutoNormal

        # Select the beta parameters corresponding to the features being used
        if use_all_features:
            beta_subset_locs = beta_locs
        else:
            # We need the indices of selected features within the *original* feature list
            selected_indices_in_original = [self.feature_names_.get_loc(feature) for feature in self.selected_features_]
            beta_subset_locs = beta_locs[selected_indices_in_original]

        # Ensure beta_subset_locs matches the number of features in X_tensor
        if beta_subset_locs.shape[0] != X_tensor.shape[1]:
             raise RuntimeError(f"Mismatch between selected beta parameters ({beta_subset_locs.shape[0]}) and input tensor features ({X_tensor.shape[1]})")


        # Compute probabilities using point estimates
        with torch.no_grad():
            # Ensure shapes match: X_tensor (N, k), beta_subset_locs (k,) -> matmul result (N,)
            logits = intercept_loc + torch.matmul(X_tensor, beta_subset_locs)
            probs = torch.sigmoid(logits)

        # Clean up param store? Maybe not if predict might be called multiple times.
        # pyro.clear_param_store()

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
    
    ax1.set_yscale('log') # Use logarithmic scale for y-axis

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
    
    ax2.set_yscale('symlog', linthresh=0.01) # Use symmetric log scale for y-axis

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
