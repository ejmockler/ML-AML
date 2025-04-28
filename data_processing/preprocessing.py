"""Functions for preprocessing AnnData objects."""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import issparse

def normalize_log_transform(adata):
    """Normalize total counts per cell and log-transform the data."""
    print("Normalizing total counts and log-transforming...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    print("Normalization and log-transform complete.")

def align_genes(adata, target_gene_list):
    """
    Aligns the genes in adata.var_names to the target_gene_list.
    Adds missing genes with zero expression.
    Removes genes not present in the target_gene_list.

    Args:
        adata (AnnData): The AnnData object to modify.
        target_gene_list (list): The list of genes to align to.

    Returns:
        AnnData: The AnnData object with aligned genes. Returns None if target_gene_list is empty.
    """
    if not target_gene_list:
        print("Warning: Target gene list is empty. Skipping gene alignment.")
        return adata # Or raise error, depending on desired behavior

    print(f"Aligning genes to a target list of {len(target_gene_list)} genes...")
    original_genes = set(adata.var_names)
    target_gene_set = set(target_gene_list)

    missing_genes = list(target_gene_set - original_genes)
    extra_genes = list(original_genes - target_gene_set)

    # Remove extra genes first
    if extra_genes:
        print(f"Removing {len(extra_genes)} genes not in the target list.")
        adata = adata[:, list(target_gene_set.intersection(original_genes))].copy()

    # Add missing genes
    if missing_genes:
        print(f"Adding {len(missing_genes)} missing genes with 0 expression.")
        # Create a zero matrix for missing genes
        zeros = pd.DataFrame(0, index=adata.obs_names, columns=missing_genes)
        # Create an AnnData object for the missing genes
        missing_genes_adata = ad.AnnData(
            X=zeros.values,
            obs=adata.obs.copy(), # Copy obs metadata
            var=pd.DataFrame(index=missing_genes)
        )
        missing_genes_adata.var['gene_name'] = missing_genes

        # Concatenate - ensure axis=1 for adding genes (variables)
        # Important: Ensure the order matches target_gene_list if needed downstream
        # Concatenating adds missing genes to the end, might need reordering.
        adata = ad.concat([adata, missing_genes_adata], axis=1, join='outer', index_unique=None, merge='first')
        # Reorder var to match target_gene_list
        adata = adata[:, target_gene_list].copy()
        print(f"After adding missing genes, adata shape: {adata.shape}")

    print(f"Gene alignment complete. Final var count: {adata.n_vars}")
    # Verify
    if not all(gene in adata.var_names for gene in target_gene_list):
         print("Error: Post-alignment verification failed. Not all target genes are present.")
         # Handle error appropriately

    return adata

def filter_cells_by_prediction(adata, condition_col='sample_type', prediction_col='PredictionRefined', case_label='case', healthy_label='healthy', malignant_label='malignant', normal_label='normal'):
    """
    Filters cells based on sample type and prediction status.
    Keeps malignant cells from case samples and normal cells from healthy samples.

    Args:
        adata (AnnData): The AnnData object to filter.
        condition_col (str): Column in adata.obs indicating sample type (e.g., 'case', 'healthy').
        prediction_col (str): Column in adata.obs with cell prediction (e.g., 'malignant', 'normal').
        case_label (str): Label for case samples in condition_col.
        healthy_label (str): Label for healthy samples in condition_col.
        malignant_label (str): Label for malignant cells in prediction_col.
        normal_label (str): Label for normal cells in prediction_col.

    Returns:
        AnnData: The filtered AnnData object.
    """
    print("Filtering cells based on condition and prediction...")
    if condition_col not in adata.obs.columns or prediction_col not in adata.obs.columns:
        print(f"Warning: Required columns ('{condition_col}' or '{prediction_col}') not found. Skipping filtering.")
        return adata

    initial_cells = adata.n_obs
    
    # Create masks
    is_case = adata.obs[condition_col] == case_label
    is_healthy = adata.obs[condition_col] == healthy_label
    is_malignant = adata.obs[prediction_col] == malignant_label
    is_normal = adata.obs[prediction_col] == normal_label

    # Combine masks: (case AND malignant) OR (healthy AND normal)
    keep_mask = (is_case & is_malignant) | (is_healthy & is_normal)

    adata_filtered = adata[keep_mask].copy() # Use .copy() to avoid view issues
    final_cells = adata_filtered.n_obs
    print(f"Filtering complete. Kept {final_cells} out of {initial_cells} cells.")

    return adata_filtered

def run_pca(adata, n_comps=50, use_highly_variable=False, layer=None, **kwargs):
    """Runs PCA on the AnnData object."""
    print(f"Running PCA (n_comps={n_comps}, use_highly_variable={use_highly_variable})...")
    # Ensure svd_solver is suitable for the data size
    if 'svd_solver' not in kwargs:
         kwargs['svd_solver'] = 'arpack' # Default used in notebook, good for reproducibility

    layer_data = adata.X if layer is None else adata.layers[layer]
    if np.any(np.isnan(layer_data)):
        print("Warning: NaN values found in data matrix before PCA. Replacing with zeros...")
        if layer is not None:
            # Make a copy of the layer to avoid modifying the original in-place
            if issparse(adata.layers[layer]):
                # For sparse matrices
                adata.layers[layer] = adata.layers[layer].copy()
                adata.layers[layer].data = np.nan_to_num(adata.layers[layer].data)
            else:
                # For dense matrices
                adata.layers[layer] = np.nan_to_num(adata.layers[layer])
        else:
            # For X matrix
            if issparse(adata.X):
                adata.X = adata.X.copy()
                adata.X.data = np.nan_to_num(adata.X.data)
            else:
                adata.X = np.nan_to_num(adata.X)
        print("NaN replacement complete.")

    sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=use_highly_variable, layer=layer, **kwargs)
    print("PCA complete. Results stored in adata.obsm['X_pca'] and related fields.")

def run_harmony(adata, batch_key, basis='X_pca', adjusted_basis='X_pca_harmony', **kwargs):
    """Runs Harmony batch correction on the AnnData object."""
    print(f"Running Harmony batch correction (batch_key='{batch_key}', basis='{basis}')...")
    if basis not in adata.obsm:
         print(f"Error: Basis '{basis}' not found in adata.obsm. Run PCA or other dimensionality reduction first.")
         return
    if batch_key not in adata.obs:
         print(f"Error: Batch key '{batch_key}' not found in adata.obs.")
         return
         
    # Ensure batch key is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[batch_key]):
        print(f"Converting batch key '{batch_key}' to categorical for Harmony.")
        adata.obs[batch_key] = adata.obs[batch_key].astype('category')
    
    # Check for NaNs in the basis matrix and replace them with zeros
    if np.any(np.isnan(adata.obsm[basis])):
        print(f"Warning: NaN values found in {basis}. Replacing with zeros for Harmony...")
        adata.obsm[basis] = np.nan_to_num(adata.obsm[basis])
        print("NaN replacement complete.")

    try:
        sc.external.pp.harmony_integrate(adata, key=batch_key, basis=basis, adjusted_basis=adjusted_basis, **kwargs)
        print(f"Harmony complete. Corrected basis stored in adata.obsm['{adjusted_basis}'].")
    except ImportError:
         print("Error: Harmony integration requires the 'harmony-pytorch' package. Please install it.")
    except Exception as e:
         print(f"Error during Harmony integration: {e}")
         print("Harmony integration failed. You may need to use a different batch correction method.")
