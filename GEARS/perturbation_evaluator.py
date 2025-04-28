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
    state: torch.Tensor
    direction: torch.Tensor
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
        gene_subset: Optional[List[int]] = None
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
        """
        self.classifier = classifier
        self.n_quantiles = n_quantiles
        self.invert_classifier_score = invert_classifier_score
        self.gears_data_dir = gears_data_dir
        self.template_adata = template_adata
        self.gene_subset = gene_subset
        
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
                              ) -> List:

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

                    # --- Calculate score for *this* perturbation ---                    
                    if aggregated_preds is None or aggregated_preds.shape[0] != 1:
                         warnings.warn(f"Prediction tensor invalid for {pert_id_str}. Skipping score calculation.")
                    else:
                        # Get the single prediction (index 0)
                        pred_state = aggregated_preds[0]
                        
                        # Compute evaluation scores.
                        scores = self.compute_state_score(pred_state)
                        direction = F.normalize(pred_state - current_state_tensor.squeeze(), dim=0)
                        current_w = self.compute_wasserstein_distance(current_state_tensor)
                        perturbed_w = self.compute_wasserstein_distance(pred_state)
                        w_improvement = (current_w - perturbed_w) / current_w if current_w != 0 else 0.0
                        reliability = torch.sigmoid(w_improvement * 5).item()

                        # Append result DIRECTLY to the final list
                        results.append(PerturbationInfo(
                            state=pred_state.cpu(),
                            direction=direction.cpu(),
                            transport_cost=scores['wasserstein'],
                            reliability=reliability,
                            scores=scores,
                            perturbation=pert_id_str
                        ))
                        processed_pert_count += 1

                    # Clean up tensors specific to this perturbation
                    del cg, loader, batch, p_out, aggregated_preds, pred_state # Add others if created
                    # Optional: More frequent cache clearing? Probably not needed now.
                    # if self.device == 'cuda': torch.cuda.empty_cache()
                    # gc.collect() 

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

        print(f"Finished evaluating {len(results)} perturbations.")
        if processed_pert_count != len(perturbations):
             warnings.warn(f"Processed {processed_pert_count} perturbations, but expected {len(perturbations)}. Some may have failed.")
        if not results:
             print("Warning: Evaluation finished but the results list is empty.")
             
        return results