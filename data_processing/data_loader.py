"""Functions for finding, categorizing, and loading sample data."""

import os
import gzip
import pandas as pd
import anndata as ad
from collections import defaultdict

def load_gzipped_txt(file_path):
    """Loads a gzipped tab-separated file into a pandas DataFrame."""
    print(f"Loading gzipped file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t', index_col=0)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def load_anndata_from_files(dem_path, anno_path):
    """Loads DEM and Annotation data into an AnnData object."""
    print(f"Loading AnnData: DEM='{os.path.basename(dem_path)}', ANNO='{os.path.basename(anno_path)}'")
    try:
        dem_df = load_gzipped_txt(dem_path)
        anno_df = load_gzipped_txt(anno_path)

        # Basic checks for compatibility
        if not dem_df.columns.equals(anno_df.index):
             print(f"Warning: Cell ID mismatch between DEM ({dem_path}) columns and ANNO ({anno_path}) index. Attempting intersection.")
             common_cells = dem_df.columns.intersection(anno_df.index)
             if len(common_cells) == 0:
                  raise ValueError("No common cells found between DEM and ANNO files.")
             print(f"Found {len(common_cells)} common cells.")
             dem_df = dem_df[common_cells]
             anno_df = anno_df.loc[common_cells]

        # Standardize column names if possible (handle variations)
        anno_df = anno_df.rename(columns={
            'CellType': 'cell_type', # Original notebook
            'celltype': 'cell_type', # Common variations
            'Cell_Type': 'cell_type',
            'PredictionRefined': 'PredictionRefined' # Keep if exists
        })

        # Create var_df from DEM index (genes)
        var_df = pd.DataFrame(index=dem_df.index)
        var_df['gene_name'] = var_df.index # Ensure gene names are stored

        # Create AnnData (transpose DEM for AnnData convention: obs x var)
        adata = ad.AnnData(X=dem_df.T.values, obs=anno_df, var=var_df)

        return adata

    except FileNotFoundError:
        # Error already printed in load_gzipped_txt, re-raise
        raise
    except ValueError as ve:
        print(f"Data loading error for {dem_path}/{anno_path}: {ve}")
        raise
    except Exception as e:
        print(f"Unexpected error loading AnnData for {dem_path}/{anno_path}: {e}")
        raise

def find_sample_files(data_dir):
    """
    Scans a directory to find DEM and ANNO files for samples,
    categorizes them, and handles missing file lookups.

    Args:
        data_dir (str): Path to the directory containing sample folders.

    Returns:
        tuple: (aml_dict, control_dict, cell_line_dict)
               Dictionaries mapping sample IDs to their file paths.
               e.g., {'SampleID': {'dem_path': '/path/to/dem', 'anno_path': '/path/to/anno'}}
    """
    aml_dict = {}
    control_dict = {}
    cell_line_dict = {}
    visited_folders = set()
    all_sample_info = defaultdict(lambda: {'dem_path': None, 'anno_path': None})

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return {}, {}, {}

    data_folders = os.listdir(data_dir)

    # First pass: Collect potential DEM and ANNO file locations
    for folder in data_folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path) or "nanopore" in folder: # Skip files and specific folders
            continue

        # Extract potential sample ID and GEO ID
        parts = folder.split('_')
        if len(parts) < 2: continue # Skip folders not matching expected format

        geo_id = parts[1]
        is_anno_folder = folder.endswith('-anno')
        sample_id = folder.split('_')[-1].replace('-anno', '')
        base_fileName = f"{geo_id}_{sample_id}"

        potential_dem = os.path.join(folder_path, f"{base_fileName}.dem.txt.gz")
        potential_anno = os.path.join(folder_path, f"{base_fileName}.anno.txt.gz")

        if not is_anno_folder and os.path.exists(potential_dem):
            if all_sample_info[sample_id]['dem_path']:
                print(f"Warning: Duplicate DEM file found for sample {sample_id} in {folder}. Keeping previous one.")
            else:
                all_sample_info[sample_id]['dem_path'] = potential_dem
                all_sample_info[sample_id]['dem_folder'] = folder # Store folder for potential cross-lookup

        if os.path.exists(potential_anno):
             if all_sample_info[sample_id]['anno_path']:
                 # Prioritize anno file from a dedicated -anno folder if available
                 if is_anno_folder:
                     all_sample_info[sample_id]['anno_path'] = potential_anno
                     all_sample_info[sample_id]['anno_folder'] = folder
                 else:
                     # Don't overwrite if the existing one came from an -anno folder
                     if not all_sample_info[sample_id].get('anno_folder', '').endswith('-anno'):
                          all_sample_info[sample_id]['anno_path'] = potential_anno
                          all_sample_info[sample_id]['anno_folder'] = folder
                     # else: keep the one from the -anno folder found previously
             else:
                 all_sample_info[sample_id]['anno_path'] = potential_anno
                 all_sample_info[sample_id]['anno_folder'] = folder

    # Second pass: Resolve missing files and categorize
    final_samples = {}
    for sample_id, paths in all_sample_info.items():
        if paths['dem_path'] and paths['anno_path']:
            # Categorize based on sample ID
            if sample_id.startswith('BM'):
                sample_type = 'control'
            elif sample_id in ['MUTZ3', 'OCI-AML3']:
                sample_type = 'cell_line'
            else:
                sample_type = 'aml'

            sample_info = {'dem_path': paths['dem_path'], 'anno_path': paths['anno_path']}

            if sample_type == 'aml':
                aml_dict[sample_id] = sample_info
            elif sample_type == 'control':
                control_dict[sample_id] = sample_info
            elif sample_type == 'cell_line':
                cell_line_dict[sample_id] = sample_info
        else:
             print(f"Warning: Skipping sample {sample_id} due to missing file(s). DEM: {paths['dem_path']}, ANNO: {paths['anno_path']}")

    print(f"Found {len(aml_dict)} AML, {len(control_dict)} control, {len(cell_line_dict)} cell line samples with both files.")
    return aml_dict, control_dict, cell_line_dict

def select_unique_samples(sample_dict):
    """
    Selects one representative sample per base ID (part before hyphen).
    Chooses the alphabetically first sample ID among replicates.

    Args:
        sample_dict (dict): Dictionary mapping full sample IDs to file paths.
                           e.g., {'GSM123-A': {...}, 'GSM123-B': {...}}

    Returns:
        dict: Dictionary mapping base IDs to the selected representative sample ID.
              e.g., {'GSM123': 'GSM123-A'}
        dict: Dictionary mapping the selected representative sample IDs back to their paths.
              e.g., {'GSM123-A': {...}}
    """
    unique_sample_map = defaultdict(list)
    for sample_id in sample_dict.keys():
        base_id = sample_id.split('-')[0]
        unique_sample_map[base_id].append(sample_id)

    selected_samples_by_base = {}
    selected_samples_paths = {}
    print(f"\nSelecting unique samples from {len(sample_dict)} total samples:")
    for base_id, samples in sorted(unique_sample_map.items()):
        selected_sample_id = min(samples) # Choose alphabetically first
        selected_samples_by_base[base_id] = selected_sample_id
        selected_samples_paths[selected_sample_id] = sample_dict[selected_sample_id]
        if len(samples) > 1:
            print(f"  Base ID {base_id}: Found {samples}, selected {selected_sample_id}")
        else:
             print(f"  Base ID {base_id}: Found {samples}, selected {selected_sample_id}")

    print(f"Selected {len(selected_samples_paths)} unique samples.")
    return selected_samples_by_base, selected_samples_paths 