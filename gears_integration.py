import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import anndata as ad
from typing import List, Dict, Union, Optional

# Import gene ID mapping functions
from data_processing.gene_id_mapping import (
    create_gene_id_mapping,
    convert_adata_var_to_ensembl,
    is_ensembl_id
)

def prepare_adata_for_gears(combined_adata, trained_genes, cache_file=None, max_workers=3, show_progress=True):
    """
    Prepares the combined_adata object for GEARS by handling gene ID conversion 
    and alignment between gene symbols and Ensembl IDs.
    Uses the Ensembl REST API for gene identifier mapping with multithreading.
    
    Args:
        combined_adata (AnnData): The AnnData object to prepare, with gene symbols in var_names
        trained_genes (list): List of genes from GEARS (usually Ensembl IDs)
        cache_file (str, optional): Path to cache the gene ID mapping
        max_workers (int, optional): Maximum number of concurrent API threads (default: 3, recommended: 3-5)
        show_progress (bool, optional): Whether to show progress bars with tqdm (default: True)
        
    Returns:
        AnnData: The prepared AnnData object with genes aligned to trained_genes
    """
    from data_processing.preprocessing import align_genes
    
    # Ensure max_workers is reasonable to avoid API rate limits
    if max_workers > 5:
        print(f"Warning: Reducing max_workers from {max_workers} to 5 to avoid Ensembl API rate limits")
        max_workers = 5
    
    print("Preparing combined_adata for GEARS...")
    
    # 1. Check if we need to convert IDs (are combined_adata and trained_genes using different ID systems?)
    sample_combined = list(combined_adata.var_names)[:5]
    sample_trained = trained_genes[:5]
    
    combined_has_ensembl = all(is_ensembl_id(gene) for gene in sample_combined)
    trained_has_ensembl = all(is_ensembl_id(gene) for gene in sample_trained)
    
    print(f"Combined AnnData uses Ensembl IDs: {combined_has_ensembl}")
    print(f"Trained genes use Ensembl IDs: {trained_has_ensembl}")
    
    # 2. If ID systems don't match, perform conversion
    if trained_has_ensembl and not combined_has_ensembl:
        print(f"Converting combined_adata gene symbols to Ensembl IDs using Ensembl REST API ({max_workers} threads)...")
        # Create a mapping between combined_adata var_names and trained_genes
        mapping_genes = list(combined_adata.var_names) + trained_genes
        mapping = create_gene_id_mapping(mapping_genes, cache_file=cache_file, max_workers=max_workers, show_progress=show_progress)
        
        # Convert combined_adata var_names to Ensembl IDs
        combined_adata_ensembl = convert_adata_var_to_ensembl(combined_adata, mapping_dict=mapping, max_workers=max_workers, show_progress=show_progress)
        print(f"Conversion complete. AnnData shape: {combined_adata_ensembl.shape}")
        
        # Now align with trained_genes
        print(f"Aligning converted AnnData to {len(trained_genes)} GEARS trained genes...")
        gears_adata = align_genes(combined_adata_ensembl, trained_genes)
    else:
        # No conversion needed, directly align
        print(f"Aligning AnnData directly to {len(trained_genes)} GEARS trained genes...")
        gears_adata = align_genes(combined_adata, trained_genes)
    
    if gears_adata is not None:
        print(f"Alignment complete. GEARS AnnData shape: {gears_adata.shape}")
        # Verification
        if gears_adata.var_names.tolist() == trained_genes:
            print("Verification successful: gears_adata.var_names matches trained_genes order.")
        else:
            print("Verification FAILED: gears_adata.var_names does NOT match trained_genes order.")
    else:
        print("Alignment failed.")
    
    return gears_adata

# Example of how to use in the notebook:
"""
# --- GEARS Preparation with Ensembl ID Mapping ---

# 1. Load the GEARS trained gene list
gears_data_dir = 'norman_umi_go'
adata_path = os.path.join(gears_data_dir, 'perturb_processed.h5ad')
print(f"\nLoading GEARS trained gene list from: {adata_path}")

if not os.path.exists(adata_path):
    raise FileNotFoundError(f"Trained AnnData file not found at: {adata_path}")
try:
    trained_genes = sc.read_h5ad(adata_path, backed='r').var_names.tolist()
    print(f"GEARS target gene list defined with {len(trained_genes)} genes.")
except Exception as e:
    raise RuntimeError(f"Failed to load trained genes from {adata_path}: {e}")
if not trained_genes:
    raise ValueError("Loaded trained_genes list is empty.")

# 2. Set up cache for gene ID mapping
cache_dir = os.path.join(os.getcwd(), 'cache')
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(cache_dir, 'ensembl_gene_mapping_cache.pkl')

# 3. Install tqdm if you want progress tracking
# !pip install tqdm

# 4. Configure multithreading for API calls
# IMPORTANT: Using more than 5 threads is not recommended as it may trigger rate limiting
max_workers = 3  # Conservative setting for Ensembl API
show_progress = True  # Set to False if you don't want progress bars

# 5. Prepare combined_adata for GEARS with automatic ID conversion
if combined_adata is not None:
    print("Preparing combined_adata for GEARS with parallel Ensembl ID mapping...")
    
    # Clean up previous dimensionality reduction results before gene alignment
    print("Cleaning up previous dimensionality reduction results before gene alignment...")
    keys_to_del = {'obsm': [], 'varm': [], 'uns': [], 'obsp': []}
    if 'X_pca' in combined_adata.obsm: keys_to_del['obsm'].append('X_pca')
    if 'X_pca_harmony' in combined_adata.obsm: keys_to_del['obsm'].append('X_pca_harmony')
    if 'X_umap' in combined_adata.obsm: keys_to_del['obsm'].append('X_umap')
    if 'PCs' in combined_adata.varm: keys_to_del['varm'].append('PCs')
    if 'pca' in combined_adata.uns: keys_to_del['uns'].append('pca')
    if 'harmony' in combined_adata.uns: keys_to_del['uns'].append('harmony')
    if 'neighbors' in combined_adata.uns:
        keys_to_del['uns'].append('neighbors')
        if 'distances' in combined_adata.obsp: keys_to_del['obsp'].append('distances')
        if 'connectivities' in combined_adata.obsp: keys_to_del['obsp'].append('connectivities')

    for key_type, keys in keys_to_del.items():
        for key in keys:
            try:
                print(f"  Deleting combined_adata.{key_type}['{key}']")
                del getattr(combined_adata, key_type)[key]
            except KeyError:
                print(f"  Key combined_adata.{key_type}['{key}'] not found, skipping deletion.")
    
    # Prepare the data with automatic gene ID conversion using multithreaded Ensembl API
    gears_adata = prepare_adata_for_gears(combined_adata, trained_genes, 
                                          cache_file=cache_file,
                                          max_workers=max_workers,
                                          show_progress=show_progress)
    
    # Optional: Clean up original combined_adata if memory is a concern
    # del combined_adata
else:
    print("Skipping GEARS preparation: combined_adata is None.")
    gears_adata = None

# Now 'gears_adata' is ready for subsequent GEARS-specific steps.
""" 