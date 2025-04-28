"""Utility functions for creating, processing, and combining AnnData objects."""

import anndata as ad
import scanpy as sc
from tqdm import tqdm
import pandas as pd

# Import functions from other modules in the package
from .data_loader import load_anndata_from_files
from .preprocessing import normalize_log_transform, align_genes, filter_cells_by_prediction

def build_processed_adata(sample_id, paths, target_gene_list, sample_type,
                          condition_col='sample_type', prediction_col='PredictionRefined', 
                          filter_cells=True):
    """
    Loads, aligns genes, normalizes, and potentially filters an AnnData object for a single sample.

    Args:
        sample_id (str): The ID of the sample.
        paths (dict): Dictionary containing 'dem_path' and 'anno_path'.
        target_gene_list (list): The list of genes to align to.
        sample_type (str): Type of the sample (e.g., 'case', 'healthy').
        condition_col (str): Column to store sample_type in adata.obs.
        prediction_col (str): Column used for cell filtering (if filter_cells=True).
        filter_cells (bool): Whether to filter cells based on prediction/condition.

    Returns:
        AnnData: The processed AnnData object for the sample, or None if loading fails.
    """
    print(f"Processing sample: {sample_id}")
    try:
        adata = load_anndata_from_files(paths['dem_path'], paths['anno_path'])
        adata.obs[condition_col] = sample_type
        adata.obs['sample_id'] = sample_id # Store original sample ID

        # Store raw counts before normalization
        adata.layers['counts'] = adata.X.copy()
        
        # 1. Align Genes (before normalization to avoid issues with adding zeros post-log)
        if target_gene_list:
             adata = align_genes(adata, target_gene_list)
        else:
             print("Warning: No target gene list provided. Skipping gene alignment.")

        # 2. Normalize & Log Transform
        normalize_log_transform(adata)
        # Store log-normalized data before potential filtering/correction
        adata.layers['lognorm'] = adata.X.copy()

        # 3. Filter Cells (Optional)
        if filter_cells:
            if prediction_col in adata.obs.columns:
                 adata = filter_cells_by_prediction(
                    adata, 
                    condition_col=condition_col,
                    prediction_col=prediction_col,
                    # Pass labels explicitly or define them centrally
                    case_label='case', 
                    healthy_label='healthy',
                    malignant_label='malignant', 
                    normal_label='normal' 
                 )
            else:
                print(f"Warning: Prediction column '{prediction_col}' not found for sample {sample_id}. Skipping cell filtering.")
        
        if adata.n_obs == 0:
             print(f"Warning: Sample {sample_id} has 0 cells after processing/filtering. Skipping.")
             return None
             
        print(f"Finished processing sample {sample_id}. Shape: {adata.shape}")
        return adata

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error processing sample {sample_id}: {e}. Skipping this sample.")
        return None

def combine_anndatas(sample_dict, target_gene_list, case_keys):
    """
    Loads, processes, and combines multiple samples into a single AnnData object.

    Args:
        sample_dict (dict): Dictionary mapping sample IDs to their file paths.
        target_gene_list (list): Master list of genes for alignment.
        case_keys (list or set): List of sample IDs belonging to the 'case' group.

    Returns:
        tuple: (combined_adata, batch_to_sample) 
               combined_adata: The final combined AnnData object, or None if failed.
               batch_to_sample: Dictionary mapping batch string IDs to sample IDs.
    """
    processed_adata_list = []
    sample_to_batch = {}
    batch_to_sample = {}
    failed_samples = []

    print(f"Starting processing and combination for {len(sample_dict)} samples.")

    # Determine a common gene list if not provided (use intersection? union? first sample?)
    # For now, requiring target_gene_list based on previous steps.
    if not target_gene_list:
         print("Error: A target_gene_list is required for consistent feature sets.")
         # Alternative: could define target_gene_list from the first loaded sample,
         # but this might exclude genes present only in later samples.
         return None, None

    # Process each sample
    for batch_id, (sample_id, paths) in enumerate(tqdm(sample_dict.items(), desc="Processing samples")):
        sample_type = 'case' if sample_id in case_keys else 'healthy'
        
        # Call the processing function for a single sample
        adata = build_processed_adata(
            sample_id=sample_id,
            paths=paths,
            target_gene_list=target_gene_list,
            sample_type=sample_type,
            filter_cells=True # Set based on notebook workflow
        )
        
        if adata is not None and adata.n_obs > 0:
            # Store batch info before potential concatenation modifications
            batch_id_str = str(batch_id) # Use string representation
            adata.obs['batch_id_str'] = batch_id_str
            processed_adata_list.append(adata)
            sample_to_batch[sample_id] = batch_id_str
            batch_to_sample[batch_id_str] = sample_id
        else:
            failed_samples.append(sample_id)

    if failed_samples:
        print(f"Warning: Failed to process or resulted in zero cells for samples: {failed_samples}")

    # Combine all processed samples
    if not processed_adata_list:
        print("Error: No AnnData objects successfully processed to combine.")
        return None, None

    print(f"Concatenating {len(processed_adata_list)} processed AnnData objects...")
    
    try:
        # Use the first adata as the reference for concatenation
        combined_adata = ad.concat(
            processed_adata_list,
            axis=0, # Concatenate along observations (cells)
            join='outer', # Keep all cells, should have same vars due to align_genes
            label='batch', # Add 'batch' column with keys '0', '1', ...
            keys=[adata.obs['batch_id_str'][0] for adata in processed_adata_list], # Use the string IDs
            index_unique=None, # Add sample ID prefix if needed: f"{sample_id}_"
            merge='first' # Strategy for conflicting annotations if any
        )
    except Exception as e:
        print(f"Error during AnnData concatenation: {e}")
        # Debugging: Print shapes and keys of list items
        for i, adt in enumerate(processed_adata_list):
            print(f"  adata {i}: shape={adt.shape}, batch_id_str={adt.obs.get('batch_id_str', 'N/A')}, sample_id={adt.obs.get('sample_id', 'N/A')}")
        return None, None

    # Ensure batch column is suitable for downstream tasks
    if 'batch' in combined_adata.obs:
        combined_adata.obs['batch'] = combined_adata.obs['batch'].astype('category')
    else:
        print("Warning: 'batch' column not automatically created during concatenation. Check anndata version or concat arguments.")
        # Attempt to create it from batch_id_str if needed
        if 'batch_id_str' in combined_adata.obs:
             print("Creating 'batch' column from 'batch_id_str'.")
             combined_adata.obs['batch'] = combined_adata.obs['batch_id_str'].astype('category')

    # Verify final object
    print(f"Data loading and combining finished. Final shape: {combined_adata.shape}")
    print(f"Combined object obs columns: {combined_adata.obs.columns.tolist()}")
    print(f"Combined object var names head: {combined_adata.var_names[:5].tolist()}...")
    print(f"Combined object layers: {list(combined_adata.layers.keys())}")
    
    return combined_adata, batch_to_sample 