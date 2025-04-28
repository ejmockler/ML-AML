"""Functions for performing analysis, such as differential expression."""

import numpy as np
import pandas as pd
from scipy.sparse import issparse

def find_differentially_expressed_genes(
    adata,
    groupby,
    group1,
    group2,
    layer=None,
    method='t-test_overestim_var', # Default method in scanpy
    corr_method='benjamini-hochberg', # Default correction
    n_genes=None # Number of top genes to return, None for all significant
):
    """
    Performs differential expression analysis between two groups using scanpy.tl.rank_genes_groups.

    Args:
        adata (AnnData): The AnnData object containing the data.
        groupby (str): The key in adata.obs to group by (e.g., 'PredictionRefined').
        group1 (str): The name of the first group (e.g., 'malignant').
        group2 (str): The name of the second group (e.g., 'normal').
        layer (str, optional): The layer in adata.layers to use. If None, uses adata.X.
        method (str): The statistical method for DE testing (see scanpy docs).
        corr_method (str): Method for multiple testing correction.
        n_genes (int, optional): Return only the top N genes ranked by significance.

    Returns:
        pd.DataFrame: DataFrame containing the DE results for group1 vs group2,
                      sorted by score/significance. Returns None if analysis fails.
        list: List of gene names identified as upregulated in group1 compared to group2
              based on positive log2 fold change and significance. Returns empty list if analysis fails.
    """
    import scanpy as sc # Local import as it might not be needed everywhere

    print(f"Running DE analysis: {group1} vs {group2} using groupby='{groupby}', method='{method}'")

    if groupby not in adata.obs:
        print(f"Error: Grouping variable '{groupby}' not found in adata.obs.")
        return None, []

    # Check if groups exist
    # Convert to categorical if not already, to check categories safely
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        adata.obs[groupby] = adata.obs[groupby].astype('category')

    if group1 not in adata.obs[groupby].cat.categories:
        print(f"Error: Group '{group1}' not found in adata.obs['{groupby}']. Categories: {adata.obs[groupby].cat.categories}")
        return None, []
    if group2 not in adata.obs[groupby].cat.categories:
        print(f"Error: Group '{group2}' not found in adata.obs['{groupby}']. Categories: {adata.obs[groupby].cat.categories}")
        return None, []

    # Ensure enough cells per group
    n_group1 = (adata.obs[groupby] == group1).sum()
    n_group2 = (adata.obs[groupby] == group2).sum()
    if n_group1 < 3 or n_group2 < 3:
        print(f"Warning: Not enough cells for DE analysis. Group1 ({group1}): {n_group1}, Group2 ({group2}): {n_group2}. Need at least 3.")
        return None, []

    try:
        # rank_genes_groups modifies adata inplace by default
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            groups=[group1],
            reference=group2,
            layer=layer,
            method=method,
            corr_method=corr_method,
            use_raw=False # Use the specified layer or .X
        )

        # Extract results into a DataFrame
        # group1 name is used as the key in the results dictionary
        result = sc.get.rank_genes_groups_df(adata, group=group1)

        # Filter for upregulated genes (positive logfoldchange and significant p-value adjusted)
        # Ensure 'logfoldchanges' and 'pvals_adj' columns exist
        if 'logfoldchanges' not in result.columns or 'pvals_adj' not in result.columns:
             print("Warning: DE results DataFrame missing expected columns ('logfoldchanges', 'pvals_adj'). Cannot determine upregulated genes.")
             upregulated_genes = []
        else:
             significant_upregulated = result[(result['logfoldchanges'] > 0) & (result['pvals_adj'] < 0.05)] # Common significance threshold
             upregulated_genes = significant_upregulated['names'].tolist()
             print(f"Found {len(upregulated_genes)} significantly upregulated genes in {group1} vs {group2} (logFC > 0, padj < 0.05).")

        # Optionally subset to top N genes
        if n_genes is not None and n_genes > 0:
            result = result.head(n_genes)
            # Update upregulated_genes list if n_genes is applied
            significant_upregulated_top = result[(result['logfoldchanges'] > 0) & (result['pvals_adj'] < 0.05)]
            upregulated_genes = significant_upregulated_top['names'].tolist()

        return result, upregulated_genes

    except Exception as e:
        print(f"Error during differential expression analysis: {e}")
        return None, []

# --- Alternative manual DE calculation (similar to original notebook) --- 
# This is kept for reference but using scanpy's rank_genes_groups is generally preferred.

def find_upregulated_genes_manual(
    adata,
    condition_col,
    group1,
    group2,
    layer=None
):
    """
    Finds genes with higher mean expression in group1 compared to group2.
    This is a simpler comparison than statistical DE tests.

    Args:
        adata (AnnData): The AnnData object.
        condition_col (str): Column in adata.obs differentiating the groups.
        group1 (str): Name of the first group.
        group2 (str): Name of the second group (reference).
        layer (str, optional): Layer to use for expression data. Uses adata.X if None.

    Returns:
        tuple: (upregulated_adata, upregulated_gene_list) or (None, []) if failed.
               upregulated_adata contains only the upregulated genes.
    """
    print(f"Finding upregulated genes comparing mean expression: {group1} vs {group2} using column '{condition_col}'")
    if condition_col not in adata.obs:
        print(f"Error: Condition column '{condition_col}' not found.")
        return None, []

    mask1 = adata.obs[condition_col] == group1
    mask2 = adata.obs[condition_col] == group2

    if mask1.sum() == 0 or mask2.sum() == 0:
        print(f"Warning: Not enough cells in one or both groups ({group1}: {mask1.sum()}, {group2}: {mask2.sum()}).")
        return None, []

    # Select the data matrix (layer or X)
    if layer:
        if layer not in adata.layers:
            print(f"Error: Layer '{layer}' not found.")
            return None, []
        X_data = adata.layers[layer]
    else:
        X_data = adata.X

    # Calculate means
    if issparse(X_data):
        means1 = np.array(X_data[mask1, :].mean(axis=0)).flatten()
        means2 = np.array(X_data[mask2, :].mean(axis=0)).flatten()
    else:
        means1 = X_data[mask1, :].mean(axis=0)
        means2 = X_data[mask2, :].mean(axis=0)

    # Identify upregulated genes
    upregulated_bool = means1 > means2
    upregulated_genes = adata.var_names[upregulated_bool].tolist()

    print(f"Number of genes with mean expression > in {group1} vs {group2}: {len(upregulated_genes)}")

    if not upregulated_genes:
        print("No genes found with higher mean expression in group1.")
        return None, []

    # Create subset AnnData
    adata_upregulated = adata[:, upregulated_genes].copy()

    return adata_upregulated, upregulated_genes 