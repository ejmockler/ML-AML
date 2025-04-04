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

    def reinit_gears(self, input_adata: ad.AnnData):
        """
        Reinitialize GEARS with a PertData built from the provided AnnData.
        Since input_adata already contains all the necessary annotations and metadata,
        we simply make a copy and pass it to create_pert_data.
        """
        new_adata = input_adata.copy()
        new_pert_data = create_pert_data(new_adata, f"{self.gears_data_dir}/pert_data")
        
        # --- Begin monkey-patching GEARS ---
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
            self.pert_list = pert_data.pert_names.tolist()
            self.num_genes = len(self.gene_list)
            self.num_perts = len(self.pert_list)
            self.default_pert_graph = pert_data.default_pert_graph
            self.saved_pred = {}
            self.saved_logvar_sum = {}
            
            ctrl_mask = (self.adata.obs.condition == 'ctrl').to_numpy()
            self._ctrl_expression = torch.tensor(
                np.mean(self.adata.X[ctrl_mask], axis=0)
            ).reshape(-1,).to(self.device)
            pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
            self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                                self.adata.uns['non_zeros_gene_idx'].items() if i in pert_full_id2pert}
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            gene_dict = {g: i for i, g in enumerate(self.gene_list)}
            self.pert2gene = {p: gene_dict[pert] for p, pert in enumerate(self.pert_list) if pert in self.gene_list}
        
        GEARS.__init__ = GEARS_patched_init_
        # --- End monkey-patching ---
        
        gears_wrapper = GEARS(new_pert_data, device=self.device, weight_bias_track=False)
        gears_wrapper.load_pretrained(f"{self.gears_data_dir}/model_ckpt")
        gears_wrapper.model.eval()
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return gears_wrapper

    def evaluate_perturbations(self, current_adata: ad.AnnData, perturbations: List, chunk_size: int = 1000) -> List:
        """
        For each perturbation, reinitialize GEARS with a PertData built from the provided AnnData,
        run the perturbation prediction, and yield a PerturbationInfo object.
        
        Parameters:
            current_adata (AnnData): Baseline cell state(s) to perturb.
            perturbations (list): List of perturbation identifiers.
            chunk_size (int, optional): If provided and > 0, the perturbations are processed in chunks of this size.
        
        Yields:
            PerturbationInfo: Evaluation result for each perturbation.
        """
        import time  # for sleep between chunks
        import gc

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
        
        # Divide the perturbations into chunks if chunk_size is provided.
        if chunk_size is None or chunk_size <= 0:
            chunked_perturbations = [perturbations]
        else:
            chunked_perturbations = [
                perturbations[i:i + chunk_size] for i in range(0, len(perturbations), chunk_size)
            ]
        
        from torch_geometric.data import DataLoader
        num_chunks = len(chunked_perturbations)
        for chunk_idx, chunk in enumerate(chunked_perturbations, 1):
            # Reinitialize GEARS for each chunk to help free memory.
            gears_instance = self.reinit_gears(current_adata)
            print(f"Processing chunk {chunk_idx}/{num_chunks} with {len(chunk)} perturbations...")
            for p in tqdm(chunk, desc=f"Chunk {chunk_idx}/{num_chunks}"):
                try:
                    # Prepare the graph dataset using GEARS's API.
                    cg = create_cell_graph_dataset_for_prediction(
                        p, gears_instance.adata, gears_instance.pert_list, gears_instance.device
                    )
                    # Set num_workers=0 to help avoid lingering processes.
                    loader = DataLoader(cg, batch_size=300, shuffle=False, num_workers=0)
                    batch = next(iter(loader))
                    batch = batch.to(gears_instance.device)
    
                    with torch.no_grad():
                        if gears_instance.config.get('uncertainty', False):
                            p_out, _ = gears_instance.best_model(batch)
                        else:
                            p_out = gears_instance.best_model(batch)
                    
                    # Compute the predicted state from the model output.
                    pred_state = torch.tensor(
                        np.mean(p_out.detach().cpu().numpy(), axis=0),
                        device=gears_instance.device
                    )
                    
                    # Compute evaluation scores.
                    scores = self.compute_state_score(pred_state)
                    scores['baseline_classifier'] = baseline_classifier_score
                    
                    direction = F.normalize(pred_state - current_state_tensor, dim=0)
                    current_w = self.compute_wasserstein_distance(current_state_tensor)
                    perturbed_w = self.compute_wasserstein_distance(pred_state)
                    w_improvement = (current_w - perturbed_w) / current_w
                    reliability = torch.sigmoid(w_improvement * 5).item()
        
                    # Yield the result promptly rather than storing it.
                    yield PerturbationInfo(
                        state=pred_state.cpu(),
                        direction=direction.cpu(),
                        transport_cost=scores['wasserstein'],
                        reliability=reliability,
                        scores=scores,
                        perturbation=p
                    )
        
                    # Clean up intermediate variables.
                    del batch, p_out, pred_state, direction, loader, cg
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    warnings.warn(f"Error evaluating perturbation {p}: {str(e)}")
                    continue
            
            # Aggressively clean up the GEARS instance.
            try:
                for attr in ["model", "best_model", "dataloader", "adata"]:
                    if hasattr(gears_instance, attr):
                        delattr(gears_instance, attr)
            except Exception:
                pass
            del gears_instance
            torch.cuda.empty_cache()
            gc.collect()