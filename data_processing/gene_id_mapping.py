import pandas as pd
import numpy as np
import re
import os
import json
import requests
import concurrent.futures
import time
import random
from typing import List, Dict, Union, Set, Tuple, Any, Callable
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def is_ensembl_id(gene_id: str) -> bool:
    """Check if a gene identifier is an Ensembl ID."""
    return gene_id.startswith('ENSG') and re.match(r'ENSG\d+', gene_id) is not None

class BackoffHandler:
    """Implements exponential backoff with jitter for API rate limiting."""
    
    def __init__(self, initial_delay=0.5, max_delay=60, factor=2, jitter=0.1):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.jitter = jitter
        self.retry_count = {}  # Track retries per endpoint
        
    def get_delay(self, endpoint):
        """Calculate delay with exponential backoff and jitter."""
        if endpoint not in self.retry_count:
            self.retry_count[endpoint] = 0
            return 0
        
        count = self.retry_count[endpoint]
        delay = min(self.max_delay, self.initial_delay * (self.factor ** count))
        
        # Add jitter to avoid thundering herd problem
        jitter_amount = self.jitter * delay
        delay = delay + random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def increment(self, endpoint):
        """Increment retry count for an endpoint."""
        if endpoint not in self.retry_count:
            self.retry_count[endpoint] = 0
        self.retry_count[endpoint] += 1
    
    def reset(self, endpoint):
        """Reset retry count for an endpoint."""
        self.retry_count[endpoint] = 0

# Global backoff handler for API calls
backoff_handler = BackoffHandler()

def ensembl_rest_query(endpoint, params=None, headers=None, data=None, method='GET', max_retries=5):
    """
    Make a request to the Ensembl REST API with exponential backoff for rate limiting.
    
    Args:
        endpoint: API endpoint (without base URL)
        params: GET parameters
        headers: HTTP headers
        data: POST data
        method: HTTP method (GET or POST)
        max_retries: Maximum number of retries before giving up
        
    Returns:
        Response data as JSON
    """
    base_url = "https://rest.ensembl.org"
    full_url = f"{base_url}{endpoint}"
    
    default_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    if headers:
        default_headers.update(headers)
    
    for attempt in range(max_retries):
        # Apply backoff if this is a retry
        if attempt > 0:
            delay = backoff_handler.get_delay(endpoint)
            print(f"Retry {attempt}/{max_retries} for {endpoint}, waiting {delay:.2f}s...")
            time.sleep(delay)
        
        try:
            if method.upper() == 'GET':
                response = requests.get(full_url, params=params, headers=default_headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(full_url, params=params, headers=default_headers, data=data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle rate limiting (HTTP 429)
            if response.status_code == 429:
                backoff_handler.increment(endpoint)
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limit exceeded for {endpoint}. Retry-After: {retry_after}s")
                time.sleep(retry_after)
                continue
            
            # Success - reset backoff
            response.raise_for_status()
            backoff_handler.reset(endpoint)
            
            if response.text:
                return response.json()
            return {}
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error for {endpoint}: {e}")
            backoff_handler.increment(endpoint)
            if attempt == max_retries - 1:  # Last attempt
                print(f"Max retries ({max_retries}) reached for {endpoint}")
                return {}
        
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"Request Error for {endpoint}: {e}")
            backoff_handler.increment(endpoint)
            if attempt == max_retries - 1:  # Last attempt
                print(f"Max retries ({max_retries}) reached for {endpoint}")
                return {}
        
        except json.JSONDecodeError:
            print(f"Invalid JSON response from Ensembl API for {endpoint}")
            return {}

def process_gene_batch(batch_data, pbar=None):
    """
    Process a batch of genes (used by thread pool).
    
    Args:
        batch_data: Dictionary with batch info
        pbar: Optional tqdm progress bar
        
    Returns:
        Dictionary with batch results
    """
    batch_type = batch_data['type']
    batch = batch_data['batch']
    species = batch_data.get('species', 'human')
    
    try:
        if batch_type == 'symbols_to_ids':
            species_map = {"human": "homo_sapiens", "mouse": "mus_musculus"}
            species_name = species_map.get(species.lower(), species.lower())
            
            # Format the batch for the API
            query = {"symbols": batch}
            
            # Use the Ensembl lookup endpoint
            endpoint = f"/lookup/symbol/{species_name}"
            response = ensembl_rest_query(endpoint, data=json.dumps(query), method='POST')
            
            # Extract mappings
            results = {}
            if response:
                for symbol, data in response.items():
                    if 'id' in data:
                        results[symbol] = data['id']
            
            result = {'type': 'symbols_to_ids', 'results': results}
        
        elif batch_type == 'ids_to_symbols':
            # Use the Ensembl post lookup endpoint
            endpoint = "/lookup/id"
            query = {"ids": batch, "expand": 0}  # expand=0 for minimal info
            
            response = ensembl_rest_query(endpoint, data=json.dumps(query), method='POST')
            
            # Extract mappings
            results = {}
            if response:  # Ensure response is not None
                for ensembl_id, data in response.items():
                    if 'display_name' in data:
                        results[ensembl_id] = data['display_name']
            
            result = {'type': 'ids_to_symbols', 'results': results}
        
        else:
            result = {'type': batch_type, 'results': {}}
    
    except Exception as e:
        print(f"Error processing {batch_type} batch: {e}")
        result = {'type': batch_type, 'results': {}}
    
    finally:
        # Update progress bar if provided
        if pbar is not None:
            pbar.update(1)
    
    return result

def batch_query_with_threads(batches, max_workers=3, show_progress=True):
    """
    Query Ensembl API in parallel using multiple threads.
    
    Args:
        batches: List of batch data dictionaries
        max_workers: Maximum number of parallel workers
        show_progress: Whether to show a progress bar
        
    Returns:
        Dictionary of combined results
    """
    symbols_to_ids = {}
    ids_to_symbols = {}
    
    # Add a small delay between submissions to prevent overwhelming the API
    submit_delay = 0.5  # seconds
    
    # Initialize progress bar if requested and tqdm is available
    pbar = None
    if show_progress and TQDM_AVAILABLE:
        pbar = tqdm(total=len(batches), desc="API Batches", unit="batch")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks with a small delay between them
        futures = []
        for batch in batches:
            future = executor.submit(process_gene_batch, batch, pbar)
            futures.append(future)
            time.sleep(submit_delay)  # Add delay between submissions
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result and 'type' in result and 'results' in result:
                    if result['type'] == 'symbols_to_ids':
                        symbols_to_ids.update(result['results'])
                    elif result['type'] == 'ids_to_symbols':
                        ids_to_symbols.update(result['results'])
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    # Close progress bar if it was created
    if pbar is not None:
        pbar.close()
    
    return {'symbols_to_ids': symbols_to_ids, 'ids_to_symbols': ids_to_symbols}

def create_gene_id_mapping(gene_list: List[str], cache_file: str = None, species: str = "human", max_workers: int = 3, show_progress: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Create a mapping between Ensembl IDs and gene symbols using multithreaded API calls.
    
    Args:
        gene_list: List of gene identifiers (Ensembl IDs or gene symbols)
        cache_file: Path to save/load cache of mapping results
        species: Species name (default: "human")
        max_workers: Maximum number of concurrent API threads (default: 3, recommended: 3-5)
        show_progress: Whether to show a progress bar (default: True)
        
    Returns:
        Dictionary with mappings: {
            'ensembl_to_symbol': {ensembl_id: symbol, ...},
            'symbol_to_ensembl': {symbol: ensembl_id, ...}
        }
    """
    # Check if cache exists and load it
    if cache_file and os.path.exists(cache_file):
        print(f"Loading gene mapping from cache: {cache_file}")
        try:
            cached_mapping = pd.read_pickle(cache_file)
            return cached_mapping
        except Exception as e:
            print(f"Error loading cache: {e}. Generating new mapping.")
    
    # Separate Ensembl IDs from symbols
    ensembl_ids = []
    symbols = []
    
    # Use tqdm for progress if available
    gene_list_iterator = gene_list
    if show_progress and TQDM_AVAILABLE:
        gene_list_iterator = tqdm(gene_list, desc="Classifying gene IDs", unit="gene")
    
    for gene in gene_list_iterator:
        if is_ensembl_id(gene):
            ensembl_ids.append(gene)
        else:
            symbols.append(gene)
    
    print(f"Found {len(symbols)} gene symbols and {len(ensembl_ids)} Ensembl IDs")
    
    # Prepare batches for API calls
    batches = []
    batch_size = 25  # Using a smaller batch size to be safer with API limits
    
    # Symbol batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        batches.append({
            'type': 'symbols_to_ids',
            'batch': batch,
            'species': species
        })
    
    # Ensembl ID batches
    for i in range(0, len(ensembl_ids), batch_size):
        batch = ensembl_ids[i:i+batch_size]
        batches.append({
            'type': 'ids_to_symbols',
            'batch': batch
        })
    
    # Execute API calls in parallel
    if batches:
        print(f"Querying Ensembl API with {len(batches)} batches using {max_workers} threads...")
        
        # Limit max_workers to a reasonable number
        if max_workers > 5:
            print(f"Warning: Reducing max_workers from {max_workers} to 5 to avoid hitting API rate limits")
            max_workers = 5
            
        api_results = batch_query_with_threads(batches, max_workers=max_workers, show_progress=show_progress)
        
        symbol_to_ensembl = api_results['symbols_to_ids']
        # Create reverse mapping from symbols_to_ids
        ensembl_to_symbol = {ensembl_id: symbol for symbol, ensembl_id in symbol_to_ensembl.items()}
        
        # Add mappings from ids_to_symbols
        ensembl_to_symbol_new = api_results['ids_to_symbols']
        ensembl_to_symbol.update(ensembl_to_symbol_new)
        
        # Add reverse mappings
        for ensembl_id, symbol in ensembl_to_symbol_new.items():
            symbol_to_ensembl[symbol] = ensembl_id
    else:
        symbol_to_ensembl = {}
        ensembl_to_symbol = {}
    
    # Create the mapping dictionary
    mapping = {
        'ensembl_to_symbol': ensembl_to_symbol,
        'symbol_to_ensembl': symbol_to_ensembl
    }
    
    # Save cache if specified
    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        pd.to_pickle(mapping, cache_file)
        print(f"Saved gene mapping to cache: {cache_file}")
    
    print(f"Created mapping with {len(ensembl_to_symbol)} ensembl→symbol and {len(symbol_to_ensembl)} symbol→ensembl entries")
    return mapping

def convert_adata_var_to_ensembl(adata, mapping_dict=None, cache_file=None, max_workers=3, show_progress=True):
    """
    Convert gene symbols in AnnData var_names to Ensembl IDs.
    
    Args:
        adata: AnnData object with gene symbols in var_names
        mapping_dict: Optional pre-existing mapping dictionary
        cache_file: Path to save/load cache of mapping results
        max_workers: Maximum number of concurrent API threads
        show_progress: Whether to show progress bars
        
    Returns:
        New AnnData object with Ensembl IDs as var_names
    """
    import scanpy as sc
    import anndata as ad
    
    # Get the mapping if not provided
    if mapping_dict is None:
        mapping_dict = create_gene_id_mapping(adata.var_names.tolist(), cache_file=cache_file, max_workers=max_workers, show_progress=show_progress)
    
    # Create a copy of the AnnData object
    adata_ensembl = adata.copy()
    
    # Map gene symbols to Ensembl IDs
    new_var_names = []
    symbol_to_ensembl = mapping_dict['symbol_to_ensembl']
    
    # Use tqdm for progress if available
    var_names_iterator = adata.var_names
    if show_progress and TQDM_AVAILABLE:
        var_names_iterator = tqdm(adata.var_names, desc="Converting to Ensembl IDs", unit="gene")
    
    for symbol in var_names_iterator:
        if is_ensembl_id(symbol):
            # Already an Ensembl ID
            new_var_names.append(symbol)
        else:
            ensembl_id = symbol_to_ensembl.get(symbol, None)
            if ensembl_id:
                new_var_names.append(ensembl_id)
            else:
                # Keep original symbol if no mapping found
                new_var_names.append(symbol)
    
    # Set new var_names
    adata_ensembl.var_names = pd.Index(new_var_names)
    
    # Store original symbols in var
    adata_ensembl.var['gene_symbol'] = adata.var_names.tolist()
    
    return adata_ensembl

def convert_adata_var_to_symbol(adata, mapping_dict=None, cache_file=None, max_workers=3, show_progress=True):
    """
    Convert Ensembl IDs in AnnData var_names to gene symbols.
    
    Args:
        adata: AnnData object with Ensembl IDs in var_names
        mapping_dict: Optional pre-existing mapping dictionary
        cache_file: Path to save/load cache of mapping results
        max_workers: Maximum number of concurrent API threads
        show_progress: Whether to show progress bars
        
    Returns:
        New AnnData object with gene symbols as var_names
    """
    import scanpy as sc
    import anndata as ad
    
    # Get the mapping if not provided
    if mapping_dict is None:
        mapping_dict = create_gene_id_mapping(adata.var_names.tolist(), cache_file=cache_file, max_workers=max_workers, show_progress=show_progress)
    
    # Create a copy of the AnnData object
    adata_symbol = adata.copy()
    
    # Map Ensembl IDs to gene symbols
    new_var_names = []
    ensembl_to_symbol = mapping_dict['ensembl_to_symbol']
    
    # Use tqdm for progress if available
    var_names_iterator = adata.var_names
    if show_progress and TQDM_AVAILABLE:
        var_names_iterator = tqdm(adata.var_names, desc="Converting to gene symbols", unit="gene")
    
    for ensembl_id in var_names_iterator:
        if not is_ensembl_id(ensembl_id):
            # Already a symbol
            new_var_names.append(ensembl_id)
        else:
            symbol = ensembl_to_symbol.get(ensembl_id, None)
            if symbol:
                new_var_names.append(symbol)
            else:
                # Keep original Ensembl ID if no mapping found
                new_var_names.append(ensembl_id)
    
    # Set new var_names
    adata_symbol.var_names = pd.Index(new_var_names)
    
    # Store original Ensembl IDs in var
    adata_symbol.var['ensembl_id'] = adata.var_names.tolist()
    
    return adata_symbol

def convert_gene_list(gene_list, target_type='ensembl', mapping_dict=None, cache_file=None, max_workers=3, show_progress=True):
    """
    Convert a list of genes from Ensembl IDs to gene symbols or vice versa.
    
    Args:
        gene_list: List of gene identifiers
        target_type: 'ensembl' or 'symbol' to specify conversion direction
        mapping_dict: Optional pre-existing mapping dictionary
        cache_file: Path to save/load cache of mapping results
        max_workers: Maximum number of concurrent API threads
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary mapping original IDs to target IDs
    """
    # Get the mapping if not provided
    if mapping_dict is None:
        mapping_dict = create_gene_id_mapping(gene_list, cache_file=cache_file, max_workers=max_workers, show_progress=show_progress)
    
    result = {}
    
    # Create iterator with progress bar if requested
    gene_iterator = gene_list
    if show_progress and TQDM_AVAILABLE:
        gene_iterator = tqdm(gene_list, desc=f"Converting to {target_type}", unit="gene")
    
    if target_type == 'ensembl':
        # Convert symbols to Ensembl IDs
        for gene in gene_iterator:
            if is_ensembl_id(gene):
                # Already Ensembl ID
                result[gene] = gene
            else:
                # Try to convert symbol to Ensembl
                ensembl_id = mapping_dict['symbol_to_ensembl'].get(gene, None)
                if ensembl_id:
                    result[gene] = ensembl_id
                else:
                    result[gene] = None
    else:
        # Convert Ensembl IDs to symbols
        for gene in gene_iterator:
            if is_ensembl_id(gene):
                # Convert Ensembl to symbol
                symbol = mapping_dict['ensembl_to_symbol'].get(gene, None)
                if symbol:
                    result[gene] = symbol
                else:
                    result[gene] = None
            else:
                # Already a symbol
                result[gene] = gene
    
    return result 