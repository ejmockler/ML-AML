"""
PerturbationEvaluator:
A tool for scoring single-step perturbations using the GEARS model.
This evaluator predicts the effect of a perturbation on a given cell state
and computes several quality metrics using batched predictions.
"""

import logging
from typing import Dict, Optional, List
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from dataclasses import dataclass
import gc  # import the garbage collector
import psutil
import os
from dataclasses import dataclass
import gc  # import the garbage collector
import collections.abc # Import for isinstance check
import csv # Add csv import

# Import GEARS so we can monkey‑patch its __init__
from gears import GEARS
from gears.utils import create_cell_graph_dataset_for_prediction
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from tqdm import tqdm


def create_pert_data(adata: ad.AnnData, data_dir: str):
    """
    Given an AnnData object (already populated with full metadata) and a data directory,
    create and process a PertData instance. This follows the notebook workflow:
      - Instantiates PertData with a data directory.
      - Converts adata.X to a CSR sparse matrix.
      - Forces the condition column to strings.
      - Defines nonzero gene indices.
      - Processes the data, prepares the split, and builds the dataloader.
    
    Parameters:
        adata (AnnData): the AnnData object to use.
        data_dir (str): the directory to be used by PertData.
    
    Returns:
        pert_data: a fully initialized PertData instance.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    from gears import PertData
    pert_data = PertData(data_dir)
    
    # Convert the expression matrix to CSR format.
    adata.X = csr_matrix(adata.X)
    
    # Ensure the condition column is stored as strings.
    adata.obs['condition'] = adata.obs['condition'].astype('str')

    def adata_define_nonzero_gene_idx(adata):
        import scipy.sparse as sp
        # Use scipy.sparse.find to get non-zero entries (rows, cols, values)
        _, non_zero_cols, _ = sp.find(adata.X)
        # Get unique columns with non-zero entries
        unique_non_zero_cols = np.unique(non_zero_cols)
        # Map these indices to gene names
        non_zeros_gene_idx = {adata.var_names[idx]: idx for idx in unique_non_zero_cols}
        # Store the result
        adata.uns['non_zeros_gene_idx'] = non_zeros_gene_idx
    
    adata_define_nonzero_gene_idx(adata)
    
    pert_data.new_data_process(dataset_name='single_cell', adata=adata, skip_calc_de=True)
    pert_data.prepare_split(split='no_split')
    pert_data.dataloader = pert_data.get_dataloader(batch_size=1)
    
    return pert_data


@dataclass
class PerturbationInfo:
    """Track perturbation evaluation results."""
    # state: torch.Tensor  # Removed to save memory
    # direction: torch.Tensor # Removed to save memory
    transport_cost: float
    reliability: float
    scores: Dict[str, float]
    perturbation: Optional[str] = None


def get_system_resources(device):
    resources = {
        'available_ram_gb': psutil.virtual_memory().available / (1024**3),
        'cpu_count': os.cpu_count() or 1, # Default to 1 if detection fails
        'available_gpu_mem_gb': 0,
        'total_gpu_mem_gb': 0,
        'has_gpu': False
    }
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory
            # Estimate available memory: total - reserved
            # Note: torch.cuda.memory_reserved() includes memory allocated + cached by PyTorch
            # This is an approximation of 'free' memory from PyTorch's perspective
            reserved_memory = torch.cuda.memory_reserved(device)
            available_memory = total_memory - reserved_memory
            
            # Sometimes reserved is slightly higher even if nothing allocated? Clamp to 0.
            available_memory = max(0, available_memory) 

            resources['available_gpu_mem_gb'] = available_memory / (1024**3)
            resources['total_gpu_mem_gb'] = total_memory / (1024**3)
            resources['has_gpu'] = True
        except Exception as e:
            warnings.warn(f"Could not query GPU memory: {e}")
            # Fallback if GPU query fails
            resources['has_gpu'] = torch.cuda.is_available() # Still acknowledge GPU exists if possible

    return resources


# Define helper function at the top level
def _pert_to_str(pert) -> str:
    """Converts a perturbation identifier (str or list/tuple) to a canonical string."""
    if isinstance(pert, collections.abc.Sequence) and not isinstance(pert, str):
        # Sort elements for consistency and join with '+'
        # Ensure elements are strings before joining
        return '+'.join(sorted(map(str, pert))) 
    elif isinstance(pert, str):
        return pert
    else:
        # Handle other potential types if necessary, or raise error
        # This ensures even non-list/non-string types become strings
        return str(pert)


class PerturbationEvaluator:
    def __init__(
        self, 
        classifier, 
        reference_state: Optional[ad.AnnData] = None,  # optional AnnData, e.g. healthy cells
        device: str = None, 
        n_quantiles: int = 50,
        invert_classifier_score: bool = True,
        gears_data_dir: str = None,  # path to load the pretrained GEARS model
        template_adata: Optional[ad.AnnData] = None,  # Template AnnData for metadata
        gene_subset: Optional[List[int]] = None,
        output_predictions_path: Optional[str] = None # Add optional path for saving predictions
    ):
        """
        Initialize the evaluator.
        
        Parameters:
          classifier: object with a predict_proba method.
          reference_state: Optional AnnData representing reference cells.
          device: str for torch device.
          n_quantiles: int, used in Wasserstein distance computation.
          invert_classifier_score: bool, controls classifier score conversion.
          gears_data_dir: directory with the pretrained GEARS model.
          template_adata: AnnData containing metadata; used in reinitializing GEARS.
          gene_subset: Optional list of indices of genes used in the upregulated AnnData.
          output_predictions_path: Optional path to save predicted state vectors as CSV.
        """
        self.classifier = classifier
        self.n_quantiles = n_quantiles
        self.invert_classifier_score = invert_classifier_score
        self.gears_data_dir = gears_data_dir
        self.template_adata = template_adata
        self.gene_subset = gene_subset
        self.output_predictions_path = output_predictions_path # Store the path
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Use the explicit reference AnnData if provided.
        if reference_state is not None:
            ref_matrix = reference_state.X.toarray() if hasattr(reference_state.X, 'toarray') else reference_state.X
        self.reference_state = torch.tensor(ref_matrix, dtype=torch.float32, device=self.device)
        self.ref_quantiles = self._compute_marginal_quantiles(self.reference_state)
        self.ref_mean = torch.mean(self.reference_state, dim=0)
        self.ref_std = torch.std(self.reference_state, dim=0)

    def _compute_marginal_quantiles(self, data: torch.Tensor) -> torch.Tensor:
        quantile_points = torch.linspace(0, 1, self.n_quantiles, device=self.device)
        return torch.quantile(data, quantile_points, dim=0)

    def compute_wasserstein_distance(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            state_r = state if state.dim() == 2 else state.unsqueeze(0)
            state_quantiles = self._compute_marginal_quantiles(state_r)
            return torch.mean(torch.abs(state_quantiles - self.ref_quantiles))

    def compute_state_score(self, state: torch.Tensor) -> dict:
        """
        Compute a score for the given state.
        """
        with torch.no_grad():
            # Wasserstein-based score.
            w_distance = self.compute_wasserstein_distance(state)
            w_score = torch.exp(-w_distance / 10).item()
            
            # Stability score using a sigmoid transform on z-scores.
            z_scores = (state - self.ref_mean) / (self.ref_std + 1e-6)
            stability_score = torch.mean(torch.sigmoid(-torch.abs(z_scores) + 5)).item()
            
            # Local density estimation score: assess how close the state is to the reference.
            dists = torch.cdist(state.unsqueeze(0), self.reference_state).squeeze()
            density_score = torch.mean(torch.exp(-dists / dists.mean())).item()

            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            # Filter state for classifier input if gene_subset is provided.
            # Here, we assume self.gene_subset is a list or np.array of indices
            if hasattr(self, "gene_subset") and self.gene_subset is not None:
                # Ensure the gene order matches that used to train the classifier.
                state_for_classifier = state.cpu().numpy()[:,self.gene_subset]
            else:
                state_for_classifier = state.cpu().numpy()
            
            
            # Compute the classifier prediction (e.g., probability) using the filtered state.
            classifier_score = self.classifier.predict_proba(state_for_classifier).mean()
            # If inverting, a lower classifier score (more healthy-like) is desirable.
            adjusted_classifier = (1.0 - classifier_score) if self.invert_classifier_score else classifier_score
            
            total_score = adjusted_classifier * w_score * stability_score * density_score
            return {
                'total': total_score,
                'classifier': classifier_score,
                'wasserstein': w_distance.item(),
                'stability': stability_score,
                'density': density_score
            }

    def _load_gears_wrapper_once(self, input_adata: ad.AnnData):
        """
        Helper to load GEARS model wrapper once and create PertData for metadata.
        """
        print("Loading GEARS model and preparing PertData...")
        # Create PertData instance using the provided input_adata (template)
        # This assumes input_adata has necessary structure or use self.template_adata
        pert_data_adata = self.template_adata.copy() if self.template_adata is not None else input_adata.copy()
        
        # Ensure necessary preprocessing steps from create_pert_data are done
        # if not already present in template_adata/input_adata
        if not isinstance(pert_data_adata.X, csr_matrix):
             pert_data_adata.X = csr_matrix(pert_data_adata.X)
        if 'condition' not in pert_data_adata.obs or pert_data_adata.obs['condition'].dtype != 'object':
             pert_data_adata.obs['condition'] = pert_data_adata.obs['condition'].astype('str')
        if 'non_zeros_gene_idx' not in pert_data_adata.uns:
             def adata_define_nonzero_gene_idx(adata):
                 import scipy.sparse as sp
                 _, non_zero_cols, _ = sp.find(adata.X)
                 unique_non_zero_cols = np.unique(non_zero_cols)
                 non_zeros_gene_idx = {adata.var_names[idx]: idx for idx in unique_non_zero_cols}
                 adata.uns['non_zeros_gene_idx'] = non_zeros_gene_idx
             adata_define_nonzero_gene_idx(pert_data_adata)
             
        # Create PertData - Note: This might still be heavy if new_data_process is complex
        # Consider simplifying if only pert_list/node_map are truly needed.
        pert_data_path = f"{self.gears_data_dir}/pert_data_single_load" # Use a distinct path
        pert_data = create_pert_data(pert_data_adata, pert_data_path)
        pert_list = pert_data.pert_names.tolist() # Extract the perturbation list

        # --- Begin monkey-patching GEARS --- (Needs to happen before GEARS instantiation)
        # This is potentially problematic if GEARS is imported elsewhere without the patch.
        # Consider a more robust patching mechanism if necessary.
        def ctrl_expression_property(self):
            return self._ctrl_expression
        GEARS.ctrl_expression = property(ctrl_expression_property)

        def GEARS_patched_init_(self, pert_data,
                                 device='cuda',
                                 weight_bias_track=False,
                                 proj_name='GEARS',
                                 exp_name='GEARS'):
            import torch
            self.weight_bias_track = weight_bias_track
            if self.weight_bias_track:
                import wandb
                wandb.init(project=proj_name, name=exp_name)
                self.wandb = wandb
            else:
                self.wandb = None
            self.device = device
            self.config = None
            self.dataloader = pert_data.dataloader
            self.adata = pert_data.adata
            self.node_map = pert_data.node_map
            self.node_map_pert = pert_data.node_map_pert
            self.data_path = pert_data.data_path
            self.dataset_name = pert_data.dataset_name
            self.split = pert_data.split
            self.seed = pert_data.seed
            self.train_gene_set_size = pert_data.train_gene_set_size
            self.set2conditions = pert_data.set2conditions
            self.subgroup = pert_data.subgroup
            self.gene_list = pert_data.gene_names.values.tolist()
            self.pert_list = pert_data.pert_names.tolist() # This is the list we need
            self.num_genes = len(self.gene_list)
            self.num_perts = len(self.pert_list)
            self.default_pert_graph = pert_data.default_pert_graph
            self.saved_pred = {}
            self.saved_logvar_sum = {}

            ctrl_mask = (self.adata.obs.condition == 'ctrl').to_numpy()
            # Ensure X is dense numpy array before mean calculation
            ctrl_X = self.adata.X[ctrl_mask]
            if hasattr(ctrl_X, "toarray"):
                 ctrl_X_dense = ctrl_X.toarray()
            else:
                 ctrl_X_dense = ctrl_X
            self._ctrl_expression = torch.tensor(
                np.mean(ctrl_X_dense, axis=0)
            ).reshape(-1,).to(self.device)
            # Ensure 'condition_name' exists
            if 'condition_name' not in self.adata.obs:
                 self.adata.obs['condition_name'] = self.adata.obs['condition']
            pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
            self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                                self.adata.uns['non_zeros_gene_idx'].items() if i in pert_full_id2pert}
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            gene_dict = {g: i for i, g in enumerate(self.gene_list)}
            self.pert2gene = {p: gene_dict[pert] for p, pert in enumerate(self.pert_list) if pert in self.gene_list}

        GEARS.__init__ = GEARS_patched_init_
        # --- End monkey-patching ---

        # Instantiate GEARS wrapper (using the patched __init__)
        gears_wrapper = GEARS(pert_data, device=self.device, weight_bias_track=False)
        
        # Load the pretrained model state
        model_path = f"{self.gears_data_dir}/model_ckpt"
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"GEARS model checkpoint not found at {model_path}")
        gears_wrapper.load_pretrained(model_path)
        
        gears_wrapper.model.eval() # Set model to evaluation mode
        gears_wrapper.model.to(self.device) # Ensure model is on the correct device

        print("GEARS wrapper and model loaded successfully.")
        # Return the entire wrapper instance
        return gears_wrapper

    def evaluate_perturbations(self, current_adata: ad.AnnData, perturbations: List,
                               chunk_size: Optional[int] = None,
                               batch_size: Optional[int] = None,
                               num_workers: Optional[int] = None
                              ) -> List[PerturbationInfo]: # Adjusted return type hint

        # --- Resource Detection ---
        device_obj = torch.device(self.device)
        resources = get_system_resources(device_obj)
        print(f"Detected Resources: RAM Available: {resources['available_ram_gb']:.2f} GB, "
              f"CPU Cores: {resources['cpu_count']}, "
              f"GPU Available: {resources['has_gpu']}, "
              f"GPU Mem Available: {resources['available_gpu_mem_gb']:.2f} GB")

        # --- Load GEARS Wrapper ONCE --- 
        try:
             gears_wrapper = self._load_gears_wrapper_once(current_adata)
             # Get model reference from the loaded wrapper for convenience
             gears_model = gears_wrapper.model 
             gears_model.eval() # Ensure eval mode just in case
        except Exception as e:
             logging.error(f"Failed to load GEARS wrapper: {e}", exc_info=True)
             raise

        # --- Dynamic Parameter Calculation (Example Heuristics - NEEDS TUNING) ---
        baseline_ram_gb = 16.0
        default_chunk_size = 1000
        if chunk_size is None:
            ram_ratio = max(0.5, resources['available_ram_gb'] / baseline_ram_gb)
            dynamic_chunk_size = int(default_chunk_size * ram_ratio)
            dynamic_chunk_size = max(100, min(dynamic_chunk_size, 5000))
            print(f"Dynamically setting chunk_size based on RAM: {dynamic_chunk_size}")
        else:
            dynamic_chunk_size = chunk_size
            print(f"Using user-provided chunk_size: {dynamic_chunk_size}")

        default_batch_size_gpu = 300
        default_batch_size_cpu = 64
        baseline_gpu_mem_gb = 6.0
        if batch_size is None:
            if resources['has_gpu'] and resources['available_gpu_mem_gb'] > 0.5:
                 vram_ratio = max(0.25, resources['available_gpu_mem_gb'] / baseline_gpu_mem_gb)
                 dynamic_batch_size = int(default_batch_size_gpu * vram_ratio)
                 dynamic_batch_size = max(32, min(dynamic_batch_size, 1024))
                 print(f"Dynamically setting batch_size based on GPU VRAM: {dynamic_batch_size}")
            else:
                 dynamic_batch_size = default_batch_size_cpu
                 print(f"Using default CPU batch_size: {dynamic_batch_size}")
        else:
            dynamic_batch_size = batch_size
            print(f"Using user-provided batch_size: {dynamic_batch_size}")

        # --- Setup --- 
        import time
        import gc
        from torch_geometric.data import DataLoader, Data, Batch 

        results = [] # Store final results directly
        processed_pert_count = 0
        prediction_file = None
        csv_writer = None
        gene_list = [] # Initialize gene_list

        # --- Open Prediction Output File (if path provided) ---
        if self.output_predictions_path:
            try:
                gene_list = gears_wrapper.gene_list # Get gene list from loaded wrapper
                print(f"Opening prediction output file: {self.output_predictions_path}")
                # Open in 'w' mode (write), creates new file or overwrites existing
                prediction_file = open(self.output_predictions_path, 'w', newline='')
                csv_writer = csv.writer(prediction_file)
                # Write header
                header = ['perturbation'] + gene_list
                csv_writer.writerow(header)
            except Exception as e:
                warnings.warn(f"Could not open or write header to prediction output file {self.output_predictions_path}: {e}. Predictions will not be saved.")
                if prediction_file: prediction_file.close() # Close if opened
                prediction_file = None
                csv_writer = None # Ensure writer is None if opening failed

        # Compute baseline state tensor once.
        current_matrix = current_adata.X.toarray() if hasattr(current_adata.X, "toarray") else current_adata.X
        current_state_tensor = torch.tensor(current_matrix, dtype=torch.float32, device=self.device)
        if len(current_state_tensor.shape) == 1:
            current_state_tensor = current_state_tensor.reshape(1, -1)

        # Compute baseline classifier score from the unperturbed state.
        if hasattr(self, "gene_subset") and self.gene_subset is not None:
            baseline_for_classifier = current_state_tensor.cpu().numpy()[:, self.gene_subset]
        else:
            baseline_for_classifier = current_state_tensor.cpu().numpy()
        baseline_classifier_score = self.classifier.predict_proba(baseline_for_classifier).mean()

        # Divide the perturbations into chunks.
        if dynamic_chunk_size <= 0:
             chunked_perturbations = [perturbations]
        else:
            chunked_perturbations = [
                perturbations[i:i + dynamic_chunk_size] for i in range(0, len(perturbations), dynamic_chunk_size)
            ]

        num_chunks = len(chunked_perturbations)

        for chunk_idx, chunk_pert_list in enumerate(chunked_perturbations, 1):
            print(f"Processing chunk {chunk_idx}/{num_chunks} with {len(chunk_pert_list)} perturbations...")
            
            # No longer creating all graphs upfront for the chunk
            # chunk_cell_graphs = []
            # original_perts_in_chunk_str = [] 

            # Iterate through perturbations *within* the chunk one by one
            pbar_pert = tqdm(chunk_pert_list, desc=f"Chunk {chunk_idx} Perturbations", leave=False)
            for p in pbar_pert:
                pert_id_str = _pert_to_str(p)
                try:
                    # --- Create Cell Graph for *this* perturbation ---                    
                    cg_list = create_cell_graph_dataset_for_prediction(
                        p,
                        gears_wrapper.adata,
                        gears_wrapper.pert_list,
                        gears_wrapper.device
                    )
                    
                    if not cg_list:
                        warnings.warn(f"No cell graph generated for perturbation {p} (str: {pert_id_str}). Skipping.")
                        continue
                    
                    cg = cg_list[0] # Assuming only one graph per perturbation

                    # --- Create DataLoader for *this* single graph ---                    
                    # Batch size is less critical here, could be 1, but use dynamic for consistency?
                    # Or simply use batch_size=len(cg_list) if it's guaranteed small?
                    # Let's use dynamic_batch_size, it won't hurt significantly for single graph.
                    loader = DataLoader([cg], batch_size=dynamic_batch_size, shuffle=False,
                                      num_workers=0, pin_memory=False) # Still use [cg] list input

                    # --- Process the single batch from the loader ---                    
                    batch = next(iter(loader)) # Get the only batch
                    batch = batch.to(gears_wrapper.device)
                    
                    aggregated_preds = None 
                    p_out = None
                    
                    with torch.no_grad():
                         p_out = gears_model(batch)
                         
                         # --- Aggregation Check (Should ideally match num_graphs=1) --- 
                         num_graphs_in_batch = batch.num_graphs # Should be 1
                         if p_out is not None and p_out.shape[0] == num_graphs_in_batch:
                              aggregated_preds = p_out 
                         elif p_out is not None:
                              # Fallback aggregation if needed (less likely now)
                              warnings.warn(f"Shape mismatch for single graph {pert_id_str}. Aggregating.")
                              num_genes = p_out.shape[1]
                              temp_agg = torch.zeros((num_graphs_in_batch, num_genes), dtype=p_out.dtype, device=p_out.device)
                              try:
                                   from torch_scatter import scatter_mean
                                   temp_agg = scatter_mean(p_out.clone(), batch.batch, dim=0, out=temp_agg)
                                   aggregated_preds = temp_agg
                              except ImportError:
                                   if num_graphs_in_batch == 1: # Simplified fallback for single graph
                                       aggregated_preds = p_out.mean(dim=0, keepdim=True)
                                   else: # Should not happen if loader input is [cg]
                                       warnings.warn("Fallback aggregation failed for unexpected batch structure.")
                                       aggregated_preds = None
                              del temp_agg
                         else:
                              warnings.warn(f"Model output p_out is None for {pert_id_str}. Cannot proceed.")
                              aggregated_preds = None

                    # --- Calculate score and Write Prediction (if enabled) ---
                    if aggregated_preds is None or aggregated_preds.shape[0] != 1:
                         warnings.warn(f"Prediction tensor invalid for {pert_id_str}. Skipping score calculation.")
                    else:
                        # Get the single prediction (index 0)
                        pred_state = aggregated_preds[0]
                        
                        # --- Write prediction to file ---
                        if csv_writer and prediction_file: # Check if file is open and writer exists
                             try:
                                 pred_state_numpy = pred_state.cpu().numpy().flatten() # Ensure it's a 1D array
                                 csv_writer.writerow([pert_id_str] + pred_state_numpy.tolist())
                             except Exception as write_e:
                                 warnings.warn(f"Error writing prediction for {pert_id_str} to CSV: {write_e}")
                                 # Consider stopping writing if errors persist? For now, just warn.

                        # Compute evaluation scores.
                        scores = self.compute_state_score(pred_state)
                        
                        # Direction calculation requires current_state_tensor and pred_state
                        # If direction is truly needed elsewhere, it could be recalculated later
                        # or stored temporarily if memory allows. For now, we skip storing it.
                        # direction = F.normalize(pred_state - current_state_tensor.squeeze(), dim=0) 
                        
                        current_w = self.compute_wasserstein_distance(current_state_tensor)
                        perturbed_w = scores['wasserstein'] # Reuse calculated Wasserstein
                        w_improvement = (current_w - perturbed_w) / current_w if current_w != 0 else 0.0
                        # Clamp reliability to [0, 1] range, sigmoid can sometimes slightly exceed bounds due to precision
                        reliability = torch.clamp(torch.sigmoid(w_improvement * 5), 0.0, 1.0).item()

                        # Append result DIRECTLY to the final list WITHOUT state/direction
                        results.append(PerturbationInfo(
                            # state=pred_state.cpu(), # Don't store state
                            # direction=direction.cpu(), # Don't store direction
                            transport_cost=scores['wasserstein'],
                            reliability=reliability,
                            scores=scores,
                            perturbation=pert_id_str
                        ))
                        processed_pert_count += 1

                    # Clean up tensors specific to this perturbation
                    # Ensure pred_state and direction (if calculated) are cleaned
                    del cg, loader, batch, p_out, aggregated_preds
                    if 'pred_state' in locals(): del pred_state # Delete if it exists
                    # if 'direction' in locals(): del direction # Delete if it exists
                    
                    # Optional: More frequent cache clearing? Might help if GPU memory is the constraint.
                    # if self.device == 'cuda': torch.cuda.empty_cache()
                    # gc.collect() # Manual GC might still be useful here

                except Exception as pert_e:
                    warnings.warn(f"Error processing perturbation {pert_id_str}: {str(pert_e)}")
                    continue # Continue to the next perturbation in the chunk
            
            pbar_pert.close()
            # Cleanup at end of chunk (less critical now, but good practice)
            if self.device == 'cuda':
                 torch.cuda.empty_cache()
            gc.collect()
            if chunk_idx < num_chunks:
                 print(f"Chunk {chunk_idx} finished ({processed_pert_count} total perts processed). Waiting 1s...")
                 time.sleep(1)

        # --- Close Prediction Output File ---
        if prediction_file:
            try:
                print(f"Closing prediction output file: {self.output_predictions_path}")
                prediction_file.close()
            except Exception as close_e:
                warnings.warn(f"Error closing prediction output file: {close_e}")

        print(f"Finished evaluating {len(results)} perturbations.")
        if processed_pert_count != len(perturbations):
             warnings.warn(f"Processed {processed_pert_count} perturbations, but expected {len(perturbations)}. Some may have failed.")
        if not results:
             print("Warning: Evaluation finished but the results list is empty.")
             
        return results
    
def predict_and_save_per_cell_structured(
    gears_wrapper: GEARS,
    adata_to_predict: ad.AnnData,
    perturbations: List,
    output_base_dir: str,
    sample_id_col: str = 'sample_id',
    batch_size: int = 64,
    mapping: Optional[Dict] = None
):
    """
    Predicts perturbation effects for every cell and saves results in structured directories,
    optimized for lower memory usage by processing sample by sample.

    Args:
        gears_wrapper: Initialized GEARS model wrapper.
        adata_to_predict: AnnData object with cells for prediction (obs must contain sample_id_col).
        perturbations: List of perturbations.
        output_base_dir: Root directory to save predictions.
        sample_id_col: Column name in adata_to_predict.obs with sample IDs.
        batch_size: Batch size for prediction within each sample.
        mapping: Optional dictionary to map perturbation symbols to Ensembl IDs.
    """
    print(f"Starting per-cell prediction and structured saving to: {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)

    if sample_id_col not in adata_to_predict.obs.columns:
        raise ValueError(f"Sample ID column '{sample_id_col}' not found in adata_to_predict.obs")

    # --- Get Global Metadata ---
    gene_names = adata_to_predict.var_names.tolist()
    all_sample_ids = adata_to_predict.obs[sample_id_col].unique().tolist()
    n_genes = adata_to_predict.n_vars
    print(f"Processing {len(all_sample_ids)} samples.")

    # --- Filter/Map Perturbations (Placeholder - Implement as needed) ---
    perturbations_to_run = perturbations # Assume input perturbations are runnable directly
    available_symbol_to_ensembl = None
    if mapping:
        print("Warning: Perturbation checking/mapping logic is simplified/skipped.")
        # Add robust checking/mapping logic here if necessary
        # available_symbols, _, available_symbol_to_ensembl = check_available_perturbations(...)
        # perturbations_to_run = [p for p in perturbations if _is_pert_valid(p, available_symbols)]
        pass

    if not perturbations_to_run:
        print("No valid perturbations to run after filtering/mapping.")
        return

    # --- Outer Loop: Iterate through Samples ---
    for sample_id in tqdm(all_sample_ids, desc="Samples"):
        sample_dir = os.path.join(output_base_dir, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)

        # Get data specific to this sample
        sample_mask = (adata_to_predict.obs[sample_id_col] == sample_id).values
        sample_cell_ids = adata_to_predict.obs_names[sample_mask].tolist()
        n_sample_cells = len(sample_cell_ids)

        if n_sample_cells == 0:
            warnings.warn(f"No cells found for sample ID '{sample_id}'. Skipping.")
            continue

        print(f"\nProcessing Sample: {sample_id} ({n_sample_cells} cells)")

        # --- Middle Loop: Iterate through Perturbations for this sample ---
        for pert_input in tqdm(perturbations_to_run, desc=f"Perturbations for Sample {sample_id}", leave=False):

            # --- Map perturbation and prepare filename ---
            pert_for_predict = pert_input # Placeholder
            if mapping and available_symbol_to_ensembl:
                 # Add mapping logic (symbol -> ensembl) here
                 if isinstance(pert_input, (list, tuple)):
                     pert_for_predict = tuple(available_symbol_to_ensembl[symbol] for symbol in pert_input)
                 elif isinstance(pert_input, str):
                     pert_for_predict = available_symbol_to_ensembl[pert_input]

            pert_filename_str = _pert_to_str(pert_input).replace('+', '_')
            output_csv_path = os.path.join(sample_dir, f"{pert_filename_str}_pred.csv")

            # --- Inner Loop: Batch predict *only for cells in this sample* ---
            sample_preds_list = []
            try:
                with torch.no_grad():
                    # Iterate using sample-specific cell indices
                    for i in range(0, n_sample_cells, batch_size):
                        batch_end = min(i + batch_size, n_sample_cells)
                        batch_size_actual = batch_end - i

                        # Provide the *same* perturbation ID for all cells in the sample's batch
                        batch_input_pert_list = [pert_for_predict] * batch_size_actual

                        # Prepare batch input specifically for the cells in this sample batch
                        # Note: gears_wrapper.predict needs a list of perturbations,
                        # one for each cell state you want to predict *from*.
                        # We assume predict works correctly when given a list of identical perturbations,
                        # implying it uses the *underlying cell states* associated with the
                        # gears_wrapper's internal adata indices matching the batch.
                        # THIS IS A CRITICAL ASSUMPTION about gears_wrapper.predict behaviour.
                        # We need the indices from the *original* adata used to init the wrapper.
                        # Let's find the original indices for the current sample batch cells.
                        
                        # Get original adata indices for the current batch of sample cells
                        current_batch_cell_ids = sample_cell_ids[i:batch_end]
                        # Find where these cell IDs appear in the *original* adata's obs_names
                        # This assumes gears_wrapper.adata retains the original cell order/indices
                        try:
                            # More robust index finding:
                            original_indices = pd.Index(gears_wrapper.adata.obs_names).get_indexer_for(current_batch_cell_ids)
                            if -1 in original_indices:
                                raise ValueError(f"Some cell IDs from sample {sample_id} batch not found in gears_wrapper.adata")
                        except Exception as idx_e:
                            warnings.warn(f"Error finding original indices for cells in batch {i} for sample {sample_id}: {idx_e}. Trying direct slicing (ASSUMES ORDER MATCH).")
                            # Fallback (less safe, assumes perfect order match between adata_to_predict and gears_wrapper.adata)
                            # Find the absolute start index of the sample in the original adata
                            sample_start_abs_index = np.where(adata_to_predict.obs_names == sample_cell_ids[0])[0][0]
                            original_indices = slice(sample_start_abs_index + i, sample_start_abs_index + batch_end)


                        # Call predict using the original indices/slice and the perturbation list
                        # The exact mechanism might depend on how gears_wrapper.predict handles indices.
                        # Common patterns:
                        # 1) Pass indices explicitly: batch_preds_dict = gears_wrapper.predict(batch_input_pert_list, indices=original_indices)
                        # 2) Rely on implicit order (if predict works on the whole dataset internally): Run predict on full dataset and slice after. (INEFFICIENT)
                        # 3) Predict needs a dataloader-like structure specific to the indices. (COMPLEX)

                        # Let's ASSUME Scenario 1 is possible or predict handles lists intelligently based on underlying adata.
                        # If predict ONLY takes perturbation lists, we might need to rethink or use a different predict method.
                        # --> Checking GEARS documentation/source for predict is essential here <--

                        # Assuming predict works by just passing the perturbation list corresponding to the batch size:
                        batch_preds_dict = gears_wrapper.predict(batch_input_pert_list, c_idx=original_indices) # Try passing indices via c_idx (common in some GEARS versions/forks)

                        # Extract prediction
                        pred_key = pert_for_predict
                        if pred_key not in batch_preds_dict and str(pred_key) in batch_preds_dict:
                             pred_key = str(pred_key)

                        if pred_key in batch_preds_dict:
                            batch_pred_array = batch_preds_dict[pred_key]
                            if batch_pred_array.ndim == 1 and batch_size_actual == 1: batch_pred_array = batch_pred_array.reshape(1, -1)
                            elif batch_pred_array.shape[0] != batch_size_actual: raise ValueError(f"Prediction shape mismatch")
                            sample_preds_list.append(batch_pred_array)
                        else:
                            warnings.warn(f"Pred key '{pred_key}' not found for sample {sample_id}, pert {pert_input}, batch {i}.")
                            sample_preds_list.append(np.full((batch_size_actual, n_genes), np.nan))

                if not sample_preds_list:
                    warnings.warn(f"No predictions for sample {sample_id}, pert {pert_input}.")
                    continue

                # Concatenate predictions *for this sample only*
                sample_preds_concat = np.concatenate(sample_preds_list, axis=0)

                if sample_preds_concat.shape != (n_sample_cells, n_genes):
                     warnings.warn(f"Concat shape mismatch for sample {sample_id}, pert {pert_input}. Expected ({n_sample_cells},{n_genes}), got {sample_preds_concat.shape}.")
                     continue

                # --- Create DataFrame and Save Immediately ---
                sample_df = pd.DataFrame(sample_preds_concat, index=sample_cell_ids, columns=gene_names)
                sample_df.to_csv(output_csv_path)

            except Exception as e:
                warnings.warn(f"Error processing sample {sample_id}, perturbation {pert_input}: {e}")
                # Optionally: print traceback using `import traceback; traceback.print_exc()`
                continue # Skip to next perturbation for this sample

            finally:
                # --- Cleanup for this sample/perturbation ---
                del sample_preds_list # Delete list of arrays
                if 'sample_preds_concat' in locals(): del sample_preds_concat
                if 'sample_df' in locals(): del sample_df
                if 'batch_preds_dict' in locals(): del batch_preds_dict
                gc.collect() # Collect garbage more frequently

    print("Finished per-cell prediction and structured saving.")
