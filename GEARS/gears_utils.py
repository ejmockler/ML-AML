import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch_geometric.data import Data, DataLoader
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import scipy.sparse as sp


from gears.data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter
from gears.utils import print_sys, zip_data_download_wrapper, dataverse_download, \
                  filter_pert_in_go, get_genes_from_perts, tar_data_download_wrapper


# NOTE: This is a custom PertData class derived from the one in finetuning.ipynb
# Consider potential differences or updates in the official cell-gears library.
class PertData:
    """
    Class for loading and processing perturbation data

    Attributes
    ----------
    data_path: str
        Path to save/load data
    gene_set_path: str
        Path to gene set to use for perturbation graph
    default_pert_graph: bool
        Whether to use default perturbation graph or not
    dataset_name: str
        Name of dataset
    dataset_path: str
        Path to dataset
    adata: AnnData
        AnnData object containing dataset
    dataset_processed: bool
        Whether dataset has been processed or not
    ctrl_adata: AnnData
        AnnData object containing control samples
    gene_names: list
        List of gene names
    node_map: dict
        Dictionary mapping gene names to indices
    split: str
        Split type
    seed: int
        Seed for splitting
    subgroup: str
        Subgroup for splitting
    train_gene_set_size: int
        Number of genes to use for training

    """

    def __init__(self, data_path,
                 gene_set_path=None,
                 default_pert_graph=True):
        """
        Parameters
        ----------

        data_path: str
            Path to save/load data
        gene_set_path: str
            Path to gene set to use for perturbation graph
        default_pert_graph: bool
            Whether to use default perturbation graph or not

        """


        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        # Download default gene2go if not present
        gene2go_path = os.path.join(self.data_path, 'gene2go_all.pkl')
        if not os.path.exists(gene2go_path):
            print_sys(f"Downloading gene2go mapping to {gene2go_path}")
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
            dataverse_download(server_path, gene2go_path)
        with open(gene2go_path, 'rb') as f:
            self.gene2go = pickle.load(f)

    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in
        perturbation graph
        """

        if self.gene_set_path is not None and os.path.exists(self.gene_set_path):
            # If gene set specified for perturbation graph, use that
            print_sys(f"Using custom gene set from: {self.gene_set_path}")
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)

        elif self.default_pert_graph is False:
             # Use genes in the data + perturbed genes as graph nodes
            print_sys("Using genes from dataset and perturbations for graph.")
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            essential_genes = list(set(essential_genes)) # Ensure unique

        else:
            # Otherwise, use a large default set of essential genes
            print_sys("Using default essential gene set for graph.")
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                     'essential_all_data_pert_genes.pkl')
            if not os.path.exists(path_):
                 print_sys(f"Downloading default essential genes to {path_}")
                 dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)

        # Filter gene2go mapping to include only essential genes
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
        print_sys(f"Perturbation graph created with {len(self.pert_names)} genes.")


    def load(self, data_name = None, data_path = None):
        """
        Load existing dataloader
        Use data_name for loading 'norman', 'adamson', 'dixit' datasets
        For other datasets use data_path

        Parameters
        ----------
        data_name: str
            Name of dataset
        data_path: str
            Path to dataset

        Returns
        -------\n        None

        """

        if data_name in ['norman', 'adamson', 'dixit',
                         'replogle_k562_essential',
                         'replogle_rpe1_essential']:
            ## load from harvard dataverse
            if data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            elif data_name == 'replogle_k562_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
            elif data_name == 'replogle_rpe1_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
            data_path_ = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path_, self.data_path)
            self.dataset_name = data_path_.split('/')[-1]
            self.dataset_path = data_path_
            adata_path = os.path.join(data_path_, 'perturb_processed.h5ad')
            print_sys(f"Loading AnnData from {adata_path}")
            self.adata = sc.read_h5ad(adata_path)

        elif data_path is not None and os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            if not os.path.exists(adata_path):
                 raise ValueError(f"AnnData file not found at {adata_path}")
            print_sys(f"Loading AnnData from {adata_path}")
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("Must provide either a supported data_name "
                             "('norman', 'adamson', 'dixit', "
                             "'replogle_k562_essential', 'replogle_rpe1_essential') "
                             "or a valid data_path containing 'perturb_processed.h5ad'")

        # Ensure adata has required fields
        if 'condition' not in self.adata.obs:
            raise ValueError("AnnData object must have 'condition' in obs")
        if 'gene_name' not in self.adata.var:
            raise ValueError("AnnData object must have 'gene_name' in var")
        # Adding 'cell_type' if not present, defaulting to 'unknown'
        if 'cell_type' not in self.adata.obs:
            print_sys("Warning: 'cell_type' not found in AnnData obs. Defaulting to 'unknown'.")
            self.adata.obs['cell_type'] = 'unknown'
        # Convert sparse matrix to dense if necessary, handle potential format issues
        if sp.issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()

        self.set_pert_genes()

        # Identify perturbations not in the GO graph
        unique_conditions = self.adata.obs['condition'].unique()
        pert_genes_in_data = set(get_genes_from_perts(unique_conditions))
        pert_genes_in_graph = set(self.pert_names)
        not_in_graph_data_perts = pert_genes_in_data - pert_genes_in_graph

        if not_in_graph_data_perts:
            print_sys('These perturbation genes are in the data but not in the '
                      'GO graph and will be ignored for predictions:')
            print_sys(list(not_in_graph_data_perts))

        # Filter AnnData to keep only perturbations present in the graph
        original_obs_count = self.adata.n_obs
        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :].copy() # Use .copy()
        print_sys(f"Filtered AnnData from {original_obs_count} to {self.adata.n_obs} observations "
                  "based on perturbations available in the GO graph.")

        # --- PyG Dataset Handling ---
        pyg_path = os.path.join(self.dataset_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.makedirs(pyg_path, exist_ok=True) # Use makedirs with exist_ok=True
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')

        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            with open(dataset_fname, "rb") as f:
                self.dataset_processed = pickle.load(f)
            print_sys("Done!")
        else:
            print_sys("Local pyg dataset not found. Creating...")
            # Ensure ctrl_adata is defined before creating dataset file
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl'].copy() # Use .copy()
            if self.ctrl_adata.n_obs == 0:
                 print_sys("Warning: No control cells ('ctrl') found in the dataset.")
            self.gene_names = list(self.adata.var.gene_name) # Ensure it's a list

            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname)
            with open(dataset_fname, "wb") as f:
                pickle.dump(self.dataset_processed, f)
            print_sys("Done!")

    def new_data_process(self, dataset_name,
                         adata = None,
                         skip_calc_de = False):
        """
        Process new dataset

        Parameters
        ----------
        dataset_name: str
            Name of dataset
        adata: AnnData object
            AnnData object containing gene expression data
        skip_calc_de: bool
            If True, skip differential expression calculation

        Returns
        -------\n        None

        """

        if adata is None:
             raise ValueError("AnnData object must be provided for new_data_process")
        if 'condition' not in adata.obs.columns.values:
            raise ValueError("Please specify condition")
        if 'gene_name' not in adata.var.columns.values:
            # If 'gene_name' is not a column but is the index, create the column
            if adata.var.index.name == 'gene_name' or 'gene_name' in adata.var.index.names:
                 adata.var['gene_name'] = adata.var.index
            else:
                # Try to find a common gene identifier, e.g., 'gene_symbol', 'Symbol'
                potential_gene_cols = ['gene_symbol', 'Symbol', 'gene']
                found_col = None
                for col in potential_gene_cols:
                    if col in adata.var.columns:
                        adata.var['gene_name'] = adata.var[col]
                        found_col = col
                        print_sys(f"Using '{col}' column as 'gene_name'.")
                        break
                if not found_col:
                    raise ValueError("Please specify gene name column as 'gene_name' in AnnData var")

        if 'cell_type' not in adata.obs.columns.values:
             print_sys("Warning: 'cell_type' not found in AnnData obs. Defaulting to 'unknown'.")
             adata.obs['cell_type'] = 'unknown'

        # Ensure adata.X is in CSR format if sparse, otherwise keep as numpy array
        if sp.issparse(adata.X) and not isinstance(adata.X, sp.csr_matrix):
             print_sys("Converting AnnData.X to CSR format.")
             adata.X = adata.X.tocsr()
        elif not sp.issparse(adata.X) and not isinstance(adata.X, np.ndarray):
             # Handle other potential types like DataFrames
             try:
                 adata.X = adata.X.values if hasattr(adata.X, 'values') else np.array(adata.X)
             except Exception as e:
                 raise TypeError(f"AnnData.X must be a NumPy array or SciPy sparse matrix. Got {type(adata.X)}. Error: {e}")


        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)

        if not os.path.exists(save_data_folder):
            os.makedirs(save_data_folder, exist_ok=True)
        self.dataset_path = save_data_folder

        # --- DE gene calculation (optional) ---
        if not skip_calc_de:
            print_sys("Calculating Differentially Expressed genes...")
            # Ensure 'condition_name' exists or create it from 'condition'
            if 'condition_name' not in adata.obs:
                 adata.obs['condition_name'] = adata.obs['condition']
            self.adata = get_DE_genes(adata) # removed skip_calc_de=False, as it's default
            print_sys("Calculating dropout non-zero genes...")
            self.adata = get_dropout_non_zero_genes(self.adata)
        else:
            print_sys("Skipping DE gene calculation.")
            self.adata = adata.copy() # Important to work on a copy

        processed_adata_path = os.path.join(save_data_folder, 'perturb_processed.h5ad')
        print_sys(f"Saving processed AnnData to {processed_adata_path}")
        # Before writing, ensure X is array or supported sparse format
        if sp.issparse(self.adata.X) and not isinstance(self.adata.X, (sp.csr_matrix, sp.csc_matrix)):
             self.adata.X = self.adata.X.tocsr() # Convert to CSR before saving if needed
        self.adata.write_h5ad(processed_adata_path)

        self.set_pert_genes()

        # Define ctrl_adata and gene_names after adata is finalized
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl'].copy() # Use .copy()
        if self.ctrl_adata.n_obs == 0:
             print_sys("Warning: No control cells ('ctrl') found in the processed dataset.")
        self.gene_names = list(self.adata.var.gene_name) # Ensure it's a list

        # --- PyG Dataset Handling ---
        pyg_path = os.path.join(save_data_folder, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.makedirs(pyg_path, exist_ok=True)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')

        # Always recreate if new_data_process is called
        # if os.path.isfile(dataset_fname):
        #     print_sys("Removing existing pyg dataset to recreate...")
        #     os.remove(dataset_fname)

        print_sys("Creating pyg object for each cell in the data...")
        self.create_dataset_file()
        print_sys("Saving new dataset pyg object at " + dataset_fname)
        with open(dataset_fname, "wb") as f:
            pickle.dump(self.dataset_processed, f)
        print_sys("Done!")


    def prepare_split(self, split = 'simulation',
                      seed = 1,
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None,
                      split_dict_path=None):

        """
        Prepare splits for training and testing

        Parameters
        ----------
        split: str
            Type of split to use. Currently, we support 'simulation',
            'simulation_single', 'combo_seen0', 'combo_seen1', 'combo_seen2',
            'single', 'no_test', 'no_split', 'custom'
        seed: int
            Random seed
        train_gene_set_size: float
            Fraction of genes to use for training
        combo_seen2_train_frac: float
            Fraction of combo seen2 perturbations to use for training
        combo_single_split_test_set_fraction: float
            Fraction of combo single perturbations to use for testing
        test_perts: list
            List of perturbations to use for testing
        only_test_set_perts: bool
            If True, only use test set perturbations for testing
        test_pert_genes: list
            List of genes to use for testing
        split_dict_path: str
            Path to dictionary used for custom split. Sample format:
                {'train': [X, Y], 'val': [P, Q], 'test': [Z]}

        Returns
        -------\n        None

        """
        if self.adata is None:
            raise RuntimeError("AnnData object not loaded or processed. Call .load() or .new_data_process() first.")

        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split', 'custom']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None # Reset subgroup info

        if split == 'custom':
            if split_dict_path is None or not os.path.exists(split_dict_path):
                raise ValueError('Please set a valid split_dict_path for custom split')
            try:
                print_sys(f"Loading custom split dictionary from {split_dict_path}")
                with open(split_dict_path, 'rb') as f:
                    self.set2conditions = pickle.load(f)
                # Validate custom split dictionary
                if not isinstance(self.set2conditions, dict) or not all(k in ['train', 'val', 'test'] for k in self.set2conditions):
                     raise ValueError("Custom split dictionary must be a dict with keys 'train', 'val', 'test'.")
                if 'test' not in self.set2conditions:
                    print_sys("Warning: Custom split dictionary does not contain a 'test' key.")
                # Ensure conditions are valid
                all_split_conditions = set(cond for cond_list in self.set2conditions.values() for cond in cond_list)
                valid_conditions = set(self.adata.obs['condition'].unique())
                invalid_conditions = all_split_conditions - valid_conditions
                if invalid_conditions:
                     print_sys(f"Warning: Custom split contains conditions not present in AnnData: {invalid_conditions}")
                self.train_gene_set_size = None # Not applicable for custom split

            except Exception as e:
                    raise ValueError(f'Error loading or validating custom split dictionary: {e}')
            print_sys("Custom split loaded successfully.")
            return # Exit after loading custom split

        # --- Standard Split Logic ---
        self.train_gene_set_size = train_gene_set_size # Store for non-custom splits
        split_folder = os.path.join(self.dataset_path, 'splits')
        os.makedirs(split_folder, exist_ok=True) # Use makedirs

        # Construct split file name
        split_file_parts = [
            self.dataset_name,
            split,
            str(seed)
        ]
        # Add parameters relevant to the split type to the filename
        if split in ['simulation', 'simulation_single']:
            split_file_parts.append(str(train_gene_set_size))
        elif split.startswith('combo'):
             split_file_parts.append(str(combo_single_split_test_set_fraction))
        #elif split == 'single': # single uses combo_single_split_test_set_fraction too
        #     split_file_parts.append(str(combo_single_split_test_set_fraction))

        split_file = '_'.join(split_file_parts) + '.pkl'
        split_path = os.path.join(split_folder, split_file)

        # Handle test_perts in filename if provided
        if test_perts:
            test_perts_str = '_'.join(sorted(test_perts)) if isinstance(test_perts, list) else test_perts
            split_path = split_path[:-4] + '_testperts-' + test_perts_str + '.pkl'
        if test_pert_genes:
            test_pert_genes_str = '_'.join(sorted(test_pert_genes)) if isinstance(test_pert_genes, list) else test_pert_genes
            split_path = split_path[:-4] + '_testpertgenes-' + test_pert_genes_str + '.pkl'


        if os.path.exists(split_path):
            print_sys(f"Local copy of split is detected. Loading from {split_path}")
            with open(split_path, "rb") as f:
                set2conditions = pickle.load(f)
            if split == 'simulation': # Load subgroup info for simulation split
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                if os.path.exists(subgroup_path):
                     with open(subgroup_path, "rb") as f:
                         subgroup = pickle.load(f)
                     self.subgroup = subgroup
                else:
                    print_sys("Warning: Subgroup file not found for existing simulation split.")
        else:
            print_sys(f"Creating new splits and saving to {split_path}")
            adata_copy = self.adata.copy() # Work on a copy for splitting

            if test_perts and isinstance(test_perts, str): # Convert string to list if needed
                test_perts = test_perts.split('_')
            if test_pert_genes and isinstance(test_pert_genes, str):
                 test_pert_genes = test_pert_genes.split('_')

            DS = DataSplitter(adata_copy, split_type=split) # Pass split type here

            if split in ['simulation', 'simulation_single']:
                # simulation split specific parameters
                adata_split, subgroup = DS.split_data(
                                                train_gene_set_size = train_gene_set_size,
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                with open(subgroup_path, "wb") as f:
                    pickle.dump(subgroup, f)
                self.subgroup = subgroup

            elif split.startswith('combo'):
                # combo perturbation specific parameters
                seen = int(split[-1])
                DS.seen = seen # Set the 'seen' level for the splitter

                adata_split = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)

            elif split == 'single':
                # single perturbation specific parameters
                 adata_split = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                       seed=seed,
                                       test_perts=test_perts) # Added test_perts handling

            elif split == 'no_test':
                # no test set
                adata_split = DS.split_data(seed=seed)

            elif split == 'no_split':
                # no split
                adata_split = adata_copy
                adata_split.obs['split'] = 'test' # Assign all to test

            # --- Generate set2conditions from the 'split' column ---
            if 'split' not in adata_split.obs:
                 raise RuntimeError("DataSplitter did not assign a 'split' column to AnnData obs.")

            # Group by 'split' and get unique conditions for each group
            set2conditions = dict(adata_split.obs.groupby('split')['condition'].unique())
            # Convert numpy arrays to lists for pickle compatibility
            set2conditions = {k: v.tolist() for k, v in set2conditions.items()}

            # Ensure all splits are present, even if empty
            for s in ['train', 'val', 'test']:
                 if s not in set2conditions:
                      set2conditions[s] = []

            with open(split_path, "wb") as f:
                pickle.dump(set2conditions, f)
            print_sys("Saved new splits.")

        self.set2conditions = set2conditions

        # Print info about the split
        print_sys("Split Summary:")
        for s, conds in self.set2conditions.items():
            num_conds = len(conds)
            if self.adata is not None and 'split' in self.adata.obs: # Check if 'split' column exists
                num_cells = self.adata[self.adata.obs['split'] == s].n_obs if s in self.adata.obs['split'].unique() else 0
            else:
                num_cells = "N/A (split not applied to adata yet)" # Or calculate based on dataset_processed later
            print_sys(f"  {s}: {num_conds} conditions, {num_cells} cells")


        if split == 'simulation' and self.subgroup:
            print_sys('Simulation split test composition:')
            for i,j in self.subgroup['test_subgroup'].items():
                print_sys(f"  {i}: {len(j)} conditions")
        print_sys("Split preparation done!")


    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        test_batch_size: int
            Batch size for testing. If None, uses batch_size.

        Returns
        -------\n        dict
            Dictionary of dataloaders

        """
        if self.adata is None:
             raise RuntimeError("AnnData object not loaded. Call .load() or .new_data_process() first.")
        if self.dataset_processed is None:
             raise RuntimeError("Processed PyG dataset not available. Check .load() or .new_data_process().")
        if self.set2conditions is None:
            raise RuntimeError("Splits not prepared. Call .prepare_split() first.")

        if test_batch_size is None:
            test_batch_size = batch_size

        # Use gene names from the final processed AnnData object
        self.gene_names = list(self.adata.var.gene_name)
        self.node_map = {x: it for it, x in enumerate(self.gene_names)}

        # Create cell graphs dictionary based on the split
        cell_graphs = {'train': [], 'val': [], 'test': []}

        print_sys("Collecting cell graphs for each split...")
        found_conditions = set()
        skipped_conditions = set()

        for split_name, conditions in self.set2conditions.items():
            if split_name not in cell_graphs: # Handle cases like 'no_test' where 'test' might be missing
                continue
            for p in conditions:
                if p == 'ctrl': # Skip adding 'ctrl' directly to splits unless specified (usually handled differently)
                     continue
                if p in self.dataset_processed:
                    cell_graphs[split_name].extend(self.dataset_processed[p])
                    found_conditions.add(p)
                else:
                    skipped_conditions.add(p)
                    # print_sys(f"Warning: Condition '{p}' from split '{split_name}' not found in processed dataset. Skipping.")

        if skipped_conditions:
            print_sys(f"Warning: Skipped {len(skipped_conditions)} conditions not found in the processed dataset: {list(skipped_conditions)}")

        print_sys("Creating dataloaders....")
        self.dataloader = {}

        # Set up dataloaders, handling empty splits gracefully
        if cell_graphs['train']:
             self.dataloader['train_loader'] = DataLoader(cell_graphs['train'],
                                         batch_size=batch_size, shuffle=True, drop_last = True,
                                         num_workers=0, pin_memory=False) # Added common DataLoader args
        else:
             print_sys("Warning: No training data found for the current split.")
             self.dataloader['train_loader'] = None # Or an empty loader

        if cell_graphs['val']:
            self.dataloader['val_loader'] = DataLoader(cell_graphs['val'],
                                      batch_size=test_batch_size, shuffle=False, # Shuffle=False for validation
                                      num_workers=0, pin_memory=False)
        else:
             print_sys("Warning: No validation data found for the current split.")
             self.dataloader['val_loader'] = None

        if self.split != 'no_test':
            if cell_graphs['test']:
                 self.dataloader['test_loader'] = DataLoader(cell_graphs['test'],
                                         batch_size=test_batch_size, shuffle=False,
                                         num_workers=0, pin_memory=False)
            else:
                 print_sys("Warning: No test data found for the current split (and split is not 'no_test').")
                 self.dataloader['test_loader'] = None
        elif 'test_loader' in self.dataloader:
            del self.dataloader['test_loader'] # Ensure no test loader if split is no_test


        print_sys("Dataloaders created:")
        for name, loader in self.dataloader.items():
            if loader:
                 print_sys(f"  {name}: {len(loader.dataset)} samples, {len(loader)} batches")
            else:
                 print_sys(f"  {name}: Empty")

        return self.dataloader # Return the dict


    def get_pert_idx(self, pert_category):
        """
        Get perturbation graph node indices for a given perturbation category string.

        Parameters
        ----------
        pert_category: str
            Perturbation category (e.g., 'GENE1+GENE2', 'GENE3', 'ctrl')

        Returns
        -------\n        list or None
            List of integer indices corresponding to perturbed genes in the pert graph.
            Returns None if any gene in pert_category is not found in node_map_pert.
            Returns [-1] for 'ctrl'.
        """
        if pert_category == 'ctrl':
            return [-1] # Special index for control

        pert_genes = pert_category.split('+')
        pert_idx = []
        valid = True
        for p in pert_genes:
            if p in self.node_map_pert:
                pert_idx.append(self.node_map_pert[p])
            else:
                # print_sys(f"Warning: Perturbed gene '{p}' in condition '{pert_category}' not found in node_map_pert. Skipping this condition in get_pert_idx.")
                valid = False
                break # Stop processing this category if a gene is missing

        return pert_idx if valid else None


    def create_cell_graph(self, X_input, y_input, de_idx, pert, pert_idx=None):
        """
        Create a cell graph from a given cell's expression data.
        Handles both sparse and dense input arrays.

        Parameters
        ----------
        X_input: scipy.sparse matrix row or np.ndarray
            Basal gene expression (control cell).
        y_input: scipy.sparse matrix row or np.ndarray
            Target gene expression (perturbed cell).
        de_idx: list or np.ndarray
            DE gene indices.
        pert: str
            Perturbation category.
        pert_idx: list
            List of perturbation indices.

        Returns
        -------
        torch_geometric.data.Data
            Cell graph to be used in dataloader.
        """

        # Convert input to dense NumPy arrays if they are sparse
        if sp.issparse(X_input):
            X = X_input.toarray().flatten()
        elif isinstance(X_input, np.ndarray):
            X = X_input.flatten()
        else:
            raise TypeError(f"Unsupported type for X_input: {type(X_input)}")

        if sp.issparse(y_input):
            y = y_input.toarray().flatten()
        elif isinstance(y_input, np.ndarray):
            y = y_input.flatten()
        else:
            raise TypeError(f"Unsupported type for y_input: {type(y_input)}")

        if X.shape != y.shape or X.ndim != 1:
             raise ValueError(f"Shape mismatch or not 1D after processing: X shape {X.shape}, y shape {y.shape}")
        if len(self.gene_names) != X.shape[0]:
             print_sys(f"!!! DEBUG: Shape mismatch in create_cell_graph for pert='{pert}'!")
             print_sys(f"!!! DEBUG: len(self.gene_names)={len(self.gene_names)}, X.shape[0]={X.shape[0]}")
             raise ValueError(f"Gene dimension mismatch: X shape {X.shape[0]} != num gene_names {len(self.gene_names)}")

        feature_mat = torch.Tensor(X).unsqueeze(1)
        target_vec = torch.Tensor(y)

        if pert_idx is None: pert_idx = [-1]
        elif not isinstance(pert_idx, list): pert_idx = [-1]

        if not isinstance(de_idx, (list, np.ndarray)): de_idx = [-1] * 20
        de_idx = np.array(de_idx).astype(int).tolist()

        return Data(x=feature_mat, pert_idx=pert_idx, y=target_vec, de_idx=de_idx, pert=pert)


    def create_cell_graph_dataset(self, split_adata, pert_category, num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs.
        MODIFIED TO PASS RAW EXPRESSION VECTORS TO create_cell_graph.
        """
        num_de_genes = 20
        adata_pert = split_adata[split_adata.obs['condition'] == pert_category]

        # --- >>> DEBUG PRINT <<< ---
        print_sys(f"--- DEBUG: Entering create_cell_graph_dataset for pert_category='{pert_category}' ---")

        # Determine DE genes (logic remains the same)
        pert_condition_name = adata_pert.obs.get('condition_name', adata_pert.obs['condition'])[0]
        de_genes_dict_key = 'rank_genes_groups_cov_all'
        de_idx = [-1] * num_de_genes # Default
        if de_genes_dict_key in split_adata.uns and pert_condition_name in split_adata.uns[de_genes_dict_key]:
            de_gene_names = split_adata.uns[de_genes_dict_key][pert_condition_name][:num_de_genes]
            found_indices = np.where(split_adata.var_names.isin(de_gene_names))[0]
            if len(found_indices) > 0:
                de_idx[:len(found_indices)] = found_indices # Fill with found indices
                if len(found_indices) < num_de_genes:
                     de_idx[len(found_indices):] = [-1] * (num_de_genes - len(found_indices)) # Pad rest
            # else de_idx remains default [-1] * 20

        # --- Get perturbation graph indices (logic remains the same) ---
        pert_idx = self.get_pert_idx(pert_category)
        if pert_category != 'ctrl' and pert_idx is None: # Check only if not control
             print_sys(f"Skipping {pert_category}: Perturbation index not found.")
             return []


        Xs = []
        ys = []

        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            if self.ctrl_adata is None or self.ctrl_adata.n_obs == 0:
                print_sys(f"Warning: No control data available. Cannot create graphs for '{pert_category}'.")
                return []

            # Sample control indices
            ctrl_indices = np.random.randint(0, len(self.ctrl_adata), num_samples * len(adata_pert))
            ctrl_samples_iter = iter(self.ctrl_adata[ctrl_indices, :].X) # Efficiently get control expressions

            for i in range(len(adata_pert)): # Iterate through perturbed cells
                pert_cell_expr = adata_pert.X[i] # Get i-th perturbed cell expression
                for _ in range(num_samples):
                    try:
                        ctrl_cell_expr = next(ctrl_samples_iter)
                        Xs.append(ctrl_cell_expr) # Append raw control expression (sparse or dense)
                        ys.append(pert_cell_expr) # Append raw perturbed expression (sparse or dense)
                    except StopIteration:
                         # Should not happen if sampling logic is correct
                         print_sys("Error: StopIteration unexpectedly reached in control samples iterator.")
                         break


        # When considering a control perturbation
        else:
            # --- >>> DEBUG PRINT <<< ---
            print_sys(f"--- DEBUG: Processing 'ctrl' branch ---")
            pert_idx = [-1] # Explicitly set for control
            de_idx = [-1] * 20 # Explicitly set default for control
            adata_pert = split_adata[split_adata.obs['condition'] == pert_category]
            # --- >>> DEBUG PRINT <<< ---
            print_sys(f"--- DEBUG: Found {len(adata_pert)} control cells in split_adata ---")
            if len(adata_pert) == 0:
                print_sys(f"--- DEBUG: No control cells found for condition '{pert_category}' in the provided split_adata. Returning empty list. ---")
                return []

            for i in range(len(adata_pert)):
                ctrl_cell_expr = adata_pert.X[i]
                # --- >>> DEBUG PRINT <<< ---
                # Avoid printing large arrays, just print type and shape
                if i == 0: # Print only for the first cell
                    print_sys(f"--- DEBUG: Control cell {i} expression type: {type(ctrl_cell_expr)}, shape: {getattr(ctrl_cell_expr, 'shape', 'N/A')} ---")
                Xs.append(ctrl_cell_expr)
                ys.append(ctrl_cell_expr)

        # Create cell graphs
        cell_graphs = []
        if not Xs: # If no pairs were created (e.g., no control data for non-ctrl pert)
             return []

        # --- >>> DEBUG PRINT <<< ---
        print_sys(f"--- DEBUG: Starting graph creation loop for {len(Xs)} pairs for pert_category='{pert_category}' ---")
        num_errors = 0
        for i in range(len(Xs)):
            try:
                # --- >>> DEBUG PRINT <<< ---
                # Print info about the inputs just before calling create_cell_graph for the first pair
                if i == 0:
                     print_sys(f"--- DEBUG: Input to create_cell_graph (pair 0):")
                     print_sys(f"--- DEBUG:   Xs[0] type: {type(Xs[0])}, shape: {getattr(Xs[0], 'shape', 'N/A')}")
                     print_sys(f"--- DEBUG:   ys[0] type: {type(ys[0])}, shape: {getattr(ys[0], 'shape', 'N/A')}")
                     print_sys(f"--- DEBUG:   de_idx: {de_idx[:5]}... (len={len(de_idx)})") # Print first few DE indices
                     print_sys(f"--- DEBUG:   pert: {pert_category}")
                     print_sys(f"--- DEBUG:   pert_idx: {pert_idx}")

                graph = self.create_cell_graph(Xs[i], ys[i], de_idx, pert_category, pert_idx)
                cell_graphs.append(graph)
            except Exception as e:
                 # --- >>> DEBUG PRINT <<< ---
                 print_sys(f"--- DEBUG: *** Error in create_cell_graph for pert '{pert_category}', sample pair {i}: {e} ***")
                 num_errors += 1
                 # Decide whether to continue or raise
                 # For debugging, let's continue to see if it affects all pairs
                 if num_errors <= 5: # Print details for the first few errors
                     import traceback
                     traceback.print_exc() # Print full traceback for the first few errors
                 elif num_errors == 6:
                     print_sys("--- DEBUG: Suppressing further error messages for this perturbation ---")

        # --- >>> DEBUG PRINT <<< ---
        print_sys(f"--- DEBUG: Finished graph creation loop for pert_category='{pert_category}'. Generated {len(cell_graphs)} graphs. Encountered {num_errors} errors. ---")
        if num_errors == len(Xs) and len(Xs) > 0:
            print_sys(f"--- DEBUG: *** All graph creations failed for pert_category='{pert_category}' ***")
            # Consider returning empty list or raising an error here if all fail
            # return [] # Returning empty list if all fail

        return cell_graphs


    def create_dataset_file(self):
        """
        Create the processed dataset file containing PyG Data objects for all perturbations.
        Uses self.adata. CORRECTED TO PROCESS 'ctrl' CONDITION.
        """
        if self.adata is None:
            raise RuntimeError("AnnData object not loaded. Call .load() or .new_data_process() first.")

        print_sys("Creating dataset file by generating cell graphs...")
        self.dataset_processed = {}
        unique_perts = self.adata.obs['condition'].unique()
        print_sys(f"Found {len(unique_perts)} unique conditions (perturbations): {unique_perts}") # Print unique conditions

        for p in tqdm(unique_perts, desc="Processing perturbations"):
            # --- >>> DEBUG PRINT <<< ---
            print_sys(f"---> DEBUG: Processing condition '{p}' in create_dataset_file <---")
            try:
                generated_graphs = self.create_cell_graph_dataset(self.adata, p)
                 # --- >>> DEBUG PRINT <<< ---
                print_sys(f"---> DEBUG: create_cell_graph_dataset for '{p}' returned {len(generated_graphs) if generated_graphs is not None else 'None'} graphs <---")
                if generated_graphs is not None:
                    self.dataset_processed[p] = generated_graphs
                else:
                    self.dataset_processed[p] = []
                    print_sys(f"Warning: create_cell_graph_dataset returned None for condition '{p}'. Storing empty list.")

            except Exception as e:
                 print_sys(f"Error processing condition '{p}' in create_dataset_file: {e}")
                 self.dataset_processed[p] = []

        num_graphs = sum(len(graphs) for graphs in self.dataset_processed.values())
        num_processed_conditions = len(self.dataset_processed)
        print_sys(f"Generated a total of {num_graphs} cell graphs for {num_processed_conditions} conditions.")

        # Explicitly check if 'ctrl' graphs were generated if 'ctrl' was a unique condition
        if 'ctrl' in unique_perts:
            # --- >>> DEBUG PRINT <<< ---
            print_sys(f"---> DEBUG: Final check: dataset_processed['ctrl'] contains {len(self.dataset_processed.get('ctrl', []))} graphs <---")
            if 'ctrl' not in self.dataset_processed or not self.dataset_processed['ctrl']:
                print_sys("Warning: 'ctrl' condition was present but no control graphs were generated or stored.")
            else:
                print_sys(f"Successfully generated {len(self.dataset_processed['ctrl'])} graphs for 'ctrl' condition.")

        print_sys("Dataset file creation done!")


def save_split_dict(split_dict, file_path):
    """
    Save a generated train/validation/test split dictionary to a file.

    Parameters
    ----------
    split_dict: dict
        The split dictionary to save. Should have keys like 'train', 'val', 'test'
        with lists of condition strings as values.
    file_path: str
        Path to the file where the dictionary will be saved (e.g., using pickle).
    """
    if not isinstance(split_dict, dict):
        raise TypeError("split_dict must be a dictionary.")
    if not all(isinstance(v, list) for v in split_dict.values()):
        raise TypeError("All values in split_dict must be lists of condition strings.")

    print_sys(f"Saving split dictionary to {file_path}")
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(split_dict, f)
        print_sys("Successfully saved split dictionary.")
    except Exception as e:
        print_sys(f"Error saving split dictionary: {e}")
        raise # Re-raise the exception after logging 