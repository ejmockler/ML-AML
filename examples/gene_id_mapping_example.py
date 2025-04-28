import scanpy as sc
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import gene ID mapping functions
from data_processing.gene_id_mapping import (
    create_gene_id_mapping,
    convert_adata_var_to_ensembl,
    convert_adata_var_to_symbol,
    convert_gene_list
)

# Set up cache directory for mapping results
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(cache_dir, 'ensembl_gene_mapping_cache.pkl')

def example_combined_to_gears():
    """
    Example of mapping combined_adata (with gene symbols) to GEARS (with Ensembl IDs)
    using the Ensembl REST API for gene ID conversion.
    """
    # Example: Load your combined_adata
    # combined_adata = sc.read_h5ad('path_to_combined_adata.h5ad')
    
    # For demonstration, let's create a simple AnnData with gene symbols
    print("\n=== Example: Converting gene symbols to Ensembl IDs via Ensembl API ===")
    gene_symbols = ['TP53', 'BRCA1', 'EGFR', 'KRAS', 'PTEN']
    print(f"Original gene symbols: {gene_symbols}")
    
    # Create mapping
    mapping = create_gene_id_mapping(gene_symbols, cache_file=cache_file)
    
    # Convert symbols to Ensembl IDs
    symbols_to_ensembl = convert_gene_list(gene_symbols, target_type='ensembl', mapping_dict=mapping)
    
    # Print results
    print("\nMapping results:")
    for symbol, ensembl_id in symbols_to_ensembl.items():
        print(f"  {symbol} -> {ensembl_id}")
    
    # With a real AnnData object, you would do:
    # gears_adata = convert_adata_var_to_ensembl(combined_adata, mapping_dict=mapping)
    
    return mapping

def example_gears_to_combined(mapping=None):
    """
    Example of mapping GEARS (with Ensembl IDs) to combined_adata (with gene symbols)
    using the Ensembl REST API for gene ID conversion.
    """
    # Example: Load your GEARS trained gene list
    # trained_genes = sc.read_h5ad('path_to_gears.h5ad', backed='r').var_names.tolist()
    
    # For demonstration, let's create sample Ensembl IDs
    print("\n=== Example: Converting Ensembl IDs to gene symbols via Ensembl API ===")
    ensembl_ids = ['ENSG00000141510', 'ENSG00000012048', 'ENSG00000146648', 'ENSG00000133703', 'ENSG00000171862']
    print(f"Original Ensembl IDs: {ensembl_ids}")
    
    # Create or reuse mapping
    if mapping is None:
        mapping = create_gene_id_mapping(ensembl_ids, cache_file=cache_file)
    
    # Convert Ensembl IDs to symbols
    ensembl_to_symbols = convert_gene_list(ensembl_ids, target_type='symbol', mapping_dict=mapping)
    
    # Print results
    print("\nMapping results:")
    for ensembl_id, symbol in ensembl_to_symbols.items():
        print(f"  {ensembl_id} -> {symbol}")
    
    # With a real AnnData object, you would do:
    # converted_adata = convert_adata_var_to_symbol(gears_adata, mapping_dict=mapping)
    
    return mapping

def real_example_workflow():
    """
    Realistic workflow based on the actual combined_adata and trained_genes
    using the Ensembl REST API for gene ID conversion.
    """
    print("\n=== Real workflow example with Ensembl API ===")
    
    # Load combined_adata and trained_genes
    # Note: Replace with actual code to load your data
    # combined_adata = sc.read_h5ad('path_to_combined_adata.h5ad')
    # trained_genes = sc.read_h5ad('path_to_gears.h5ad', backed='r').var_names.tolist()
    
    # For demonstration, simulate these variables
    combined_gene_symbols = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'PTEN', 'GAPDH']
    trained_genes_ensembl = ['ENSG00000141510', 'ENSG00000012048', 'ENSG00000146648', 'ENSG00000136997', 'ENSG00000171862']
    
    print(f"Combined AnnData genes (symbols): {combined_gene_symbols}")
    print(f"GEARS trained genes (Ensembl): {trained_genes_ensembl}")
    
    # Create a unified mapping using both gene lists
    all_genes = combined_gene_symbols + trained_genes_ensembl
    mapping = create_gene_id_mapping(all_genes, cache_file=cache_file)
    
    # 1. Approach: Convert combined_adata to Ensembl IDs
    print("\nApproach 1: Convert combined_adata genes to Ensembl IDs")
    combined_to_ensembl = convert_gene_list(combined_gene_symbols, target_type='ensembl', mapping_dict=mapping)
    converted_genes_ensembl = [combined_to_ensembl[symbol] for symbol in combined_gene_symbols]
    print(f"Converted genes: {converted_genes_ensembl}")
    
    # 2. Approach: Convert trained_genes to symbols
    print("\nApproach 2: Convert trained_genes to symbols")
    trained_to_symbols = convert_gene_list(trained_genes_ensembl, target_type='symbol', mapping_dict=mapping)
    converted_trained_symbols = [trained_to_symbols[ensembl] for ensembl in trained_genes_ensembl]
    print(f"Converted trained genes: {converted_trained_symbols}")
    
    # 3. Find common genes using Ensembl IDs as the reference
    print("\nFinding common genes using Ensembl IDs")
    valid_combined_ensembl = [ens_id for ens_id in converted_genes_ensembl if ens_id is not None]
    trained_ensembl_set = set(trained_genes_ensembl)
    common_ensembl = [ens_id for ens_id in valid_combined_ensembl if ens_id in trained_ensembl_set]
    print(f"Common genes (Ensembl): {common_ensembl}")
    
    # 4. Align genes for GEARS
    print("\nAligning genes for GEARS")
    print("For a real workflow, you would do:")
    print("1. Convert combined_adata var_names to Ensembl IDs using Ensembl API")
    print("2. Use align_genes() to match with trained_genes")
    print("3. This ensures gene order and presence matches exactly")
    
    return mapping

if __name__ == "__main__":
    # Run the example functions
    mapping = example_combined_to_gears()
    example_gears_to_combined(mapping)
    real_example_workflow() 