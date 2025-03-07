#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spoticell Data Preprocessing Module
-----------------------------------
Handles downloading, preprocessing, and transforming scRNA-seq data for Spoticell model training.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
import math
import json
import gc
import argparse
from typing import List, Dict, Optional, Union, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

# Set figure parameters
sc.settings.set_figure_params(dpi=100, facecolor='white')
plt.rcParams['figure.figsize'] = (10, 8)


def download_h5ad_file(url: str, output_path: str) -> str:
    """
    Download h5ad file from URL
    
    Parameters:
    -----------
    url: str
        URL of the h5ad file
    output_path: str
        Path to save the downloaded file
        
    Returns:
    --------
    str
        Path to the downloaded file
    """
    if os.path.exists(output_path):
        print(f"File already exists at {output_path}, skipping download")
        return output_path
    
    print(f"Downloading file from {url} to {output_path}")
    
    # Stream download to handle large files
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 8192  # 8 Kibibytes
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as t:
                for chunk in r.iter_content(chunk_size=block_size):
                    t.update(len(chunk))
                    f.write(chunk)
    
    print(f"Download complete: {output_path}")
    return output_path


def explore_h5ad(file_path: str, verbose: bool = True) -> sc.AnnData:
    """
    Load and explore h5ad file
    
    Parameters:
    -----------
    file_path: str
        Path to h5ad file
    verbose: bool
        Whether to print summary information
        
    Returns:
    --------
    anndata.AnnData
        Loaded h5ad file
    """
    print(f"Loading {file_path}...")
    adata = sc.read_h5ad(file_path)
    
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Dataset summary: {os.path.basename(file_path)}")
        print(f"{'=' * 50}")
        print(f"AnnData object with {adata.n_obs} cells and {adata.n_vars} genes")
        print(f"\nObservation (cell) metadata:")
        for col in adata.obs.columns:
            n_unique = adata.obs[col].nunique()
            print(f"  - {col}: {n_unique} unique values")
            
            # For columns with fewer than 20 unique values, show them
            if n_unique < 20:
                value_counts = adata.obs[col].value_counts()
                print(f"    Values: {dict(value_counts)}")
            else:
                print(f"    Top 5 values: {dict(adata.obs[col].value_counts().head(5))}")
        
        print(f"\nVariable (gene) metadata:")
        for col in adata.var.columns:
            print(f"  - {col}: {adata.var[col].nunique()} unique values")
        
        # Check for existing quality control metrics
        if 'n_counts' in adata.obs:
            print("\nQuality control metrics already present")
        else:
            print("\nNo standard QC metrics found")
        
        # Check layers
        if adata.layers:
            print(f"\nLayers: {list(adata.layers.keys())}")
        else:
            print("\nNo layers present")
        
        # Check if X is raw counts or normalized
        x_mean = adata.X.mean()
        if isinstance(x_mean, np.matrix):
            x_mean = x_mean[0, 0]
        print(f"\nMean value in X matrix: {x_mean:.4f}")
        if x_mean < 5:
            print("X matrix appears to be normalized (mean < 5)")
        else:
            print("X matrix may contain raw counts (mean >= 5)")
            
    return adata


def explore_metadata_column(adata: sc.AnnData, column: str) -> pd.Series:
    """
    Explore unique values in a metadata column
    
    Parameters:
    -----------
    adata: anndata.AnnData
        AnnData object
    column: str
        Column name to explore
        
    Returns:
    --------
    pd.Series
        Value counts for the column
    """
    if column not in adata.obs.columns:
        print(f"Column '{column}' not found in metadata")
        print(f"Available columns: {list(adata.obs.columns)}")
        return pd.Series()
    
    value_counts = adata.obs[column].value_counts()
    print(f"Value counts for '{column}':")
    print(value_counts)
    
    # Plot the distribution
    plt.figure(figsize=(12, 6))
    if len(value_counts) > 20:
        # For many values, show top 20
        value_counts.head(20).plot(kind='bar')
        plt.title(f"Top 20 values for '{column}' (out of {len(value_counts)} total)")
    else:
        value_counts.plot(kind='bar')
        plt.title(f"All values for '{column}'")
    plt.ylabel("Number of cells")
    plt.xlabel(column)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return value_counts


def filter_cells(adata: sc.AnnData, 
                column: str, 
                values_to_keep: List[str],
                operation: str = 'include') -> sc.AnnData:
    """
    Filter cells based on metadata values
    
    Parameters:
    -----------
    adata: anndata.AnnData
        AnnData object
    column: str
        Column name for filtering
    values_to_keep: List[str]
        Values to keep or exclude
    operation: str
        'include' to keep cells with values in values_to_keep
        'exclude' to remove cells with values in values_to_keep
        
    Returns:
    --------
    anndata.AnnData
        Filtered AnnData object
    """
    if column not in adata.obs.columns:
        print(f"Column '{column}' not found in metadata")
        return adata
    
    n_cells_before = adata.n_obs
    
    if operation == 'include':
        # Keep only cells with specified values
        mask = adata.obs[column].isin(values_to_keep)
        filtered_adata = adata[mask].copy()
        print(f"Kept {filtered_adata.n_obs} cells with {column} in {values_to_keep}")
        print(f"Removed {n_cells_before - filtered_adata.n_obs} cells")
    elif operation == 'exclude':
        # Remove cells with specified values
        mask = ~adata.obs[column].isin(values_to_keep)
        filtered_adata = adata[mask].copy()
        print(f"Removed {n_cells_before - filtered_adata.n_obs} cells with {column} in {values_to_keep}")
        print(f"Kept {filtered_adata.n_obs} cells")
    else:
        print(f"Unknown operation: {operation}")
        return adata
    
    return filtered_adata


def preprocess_adata(adata: sc.AnnData, 
                    min_genes: int = 200,
                    min_cells: int = 3,
                    target_sum: Optional[int] = None,
                    normalize: bool = True,
                    log_transform: bool = True,
                    highly_variable: bool = True,
                    n_top_genes: int = 2000) -> sc.AnnData:
    """
    Basic preprocessing of AnnData object
    
    Parameters:
    -----------
    adata: anndata.AnnData
        AnnData object
    min_genes: int
        Minimum number of genes expressed for a cell to be kept
    min_cells: int
        Minimum number of cells expressing a gene for the gene to be kept
    target_sum: Optional[int]
        Target sum for normalization, None for default (10,000)
    normalize: bool
        Whether to normalize total counts per cell
    log_transform: bool
        Whether to log transform expression values
    highly_variable: bool
        Whether to identify highly variable genes
    n_top_genes: int
        Number of highly variable genes to keep
        
    Returns:
    --------
    anndata.AnnData
        Preprocessed AnnData object
    """
    print("Starting preprocessing...")
    
    # Create a copy to avoid modifying the original
    adata_pp = adata.copy()
    
    # Calculate QC metrics if not present
    if 'n_genes_by_counts' not in adata_pp.obs:
        sc.pp.calculate_qc_metrics(adata_pp, inplace=True)
    
    # Filter cells with too few genes
    n_cells_before = adata_pp.n_obs
    sc.pp.filter_cells(adata_pp, min_genes=min_genes)
    print(f"Filtered out {n_cells_before - adata_pp.n_obs} cells with fewer than {min_genes} genes")
    
    # Filter genes expressed in too few cells
    n_genes_before = adata_pp.n_vars
    sc.pp.filter_genes(adata_pp, min_cells=min_cells)
    print(f"Filtered out {n_genes_before - adata_pp.n_vars} genes expressed in fewer than {min_cells} cells")
    
    if normalize:
        # Normalize total counts per cell
        sc.pp.normalize_total(adata_pp, target_sum=target_sum)
        print(f"Normalized counts per cell")
    
    if log_transform:
        # Log transform the data
        sc.pp.log1p(adata_pp)
        print(f"Log-transformed data")
    
    if highly_variable:
        # Identify highly variable genes
        sc.pp.highly_variable_genes(adata_pp, n_top_genes=n_top_genes)
        print(f"Identified {sum(adata_pp.var.highly_variable)} highly variable genes")
    
    print("Preprocessing complete")
    return adata_pp


def combine_adata_objects(adata_list: List[sc.AnnData], 
                         cell_type_column: str = 'cell_type',
                         batch_key: str = 'dataset',
                         join_vars: str = 'intersection') -> sc.AnnData:
    """
    Combine multiple AnnData objects
    
    Parameters:
    -----------
    adata_list: List[anndata.AnnData]
        List of AnnData objects to combine
    cell_type_column: str
        Column name for cell type information
    batch_key: str
        Column name to store the batch information
    join_vars: str
        How to join the variables (genes): 'intersection' or 'union'
        
    Returns:
    --------
    anndata.AnnData
        Combined AnnData object
    """
    if not adata_list:
        print("No AnnData objects to combine")
        return None
    
    # Ensure cell_type_column is present in all objects
    for i, adata in enumerate(adata_list):
        if cell_type_column not in adata.obs.columns:
            print(f"Warning: '{cell_type_column}' not found in dataset {i}")
            print(f"Available columns: {list(adata.obs.columns)}")
            print("You might need to map another column to 'cell_type'")
    
    # Add batch information
    for i, adata in enumerate(adata_list):
        adata.obs[batch_key] = f"batch_{i}"
    
    print("Combining AnnData objects...")
    # Concatenate along the observations (cells) axis
    combined = sc.concat(
        adata_list, 
        join=join_vars,
        merge='same',  # Same variables
        index_unique='-'  # Make observation names unique by adding a suffix
    )
    
    print(f"Combined dataset has {combined.n_obs} cells and {combined.n_vars} genes")
    return combined


def standardize_cell_types(adata: sc.AnnData,
                          source_column: str,
                          target_column: str = 'standardized_cell_type',
                          mapping_dict: Optional[Dict[str, str]] = None) -> sc.AnnData:
    """
    Standardize cell type labels
    
    Parameters:
    -----------
    adata: anndata.AnnData
        AnnData object
    source_column: str
        Column containing original cell type labels
    target_column: str
        Column to store standardized cell type labels
    mapping_dict: Optional[Dict[str, str]]
        Dictionary mapping original labels to standardized labels
        If None, the original labels are copied to the target column
        
    Returns:
    --------
    anndata.AnnData
        AnnData object with standardized cell type labels
    """
    if source_column not in adata.obs.columns:
        print(f"Source column '{source_column}' not found in metadata")
        return adata
    
    # Copy AnnData to avoid modifying the original
    adata_out = adata.copy()
    
    if mapping_dict is None:
        # Use original labels
        adata_out.obs[target_column] = adata_out.obs[source_column]
        print(f"Copied '{source_column}' to '{target_column}' without modification")
    else:
        # Map labels using the provided dictionary
        unique_labels = adata_out.obs[source_column].unique()
        unmapped_labels = [label for label in unique_labels if label not in mapping_dict]
        
        if unmapped_labels:
            print(f"Warning: {len(unmapped_labels)} labels have no mapping:")
            print(unmapped_labels[:10])
            print("These will be set to 'unknown'")
            
            # Create a new mapping with 'unknown' for unmapped labels
            full_mapping = mapping_dict.copy()
            for label in unmapped_labels:
                full_mapping[label] = 'unknown'
                
            adata_out.obs[target_column] = adata_out.obs[source_column].map(full_mapping)
        else:
            adata_out.obs[target_column] = adata_out.obs[source_column].map(mapping_dict)
            
        print(f"Mapped '{source_column}' to '{target_column}' using provided dictionary")
        
        # Print statistics of the mapping
        print("\nCell type mapping statistics:")
        value_counts = adata_out.obs[target_column].value_counts()
        for cell_type, count in value_counts.items():
            percentage = 100 * count / adata_out.n_obs
            print(f"  - {cell_type}: {count} cells ({percentage:.2f}%)")
    
    return adata_out


def save_final_dataset(adata: sc.AnnData, 
                       output_path: str,
                       compress: bool = True) -> str:
    """
    Save the final combined dataset
    
    Parameters:
    -----------
    adata: anndata.AnnData
        AnnData object to save
    output_path: str
        Path to save the dataset
    compress: bool
        Whether to compress the dataset
        
    Returns:
    --------
    str
        Path to the saved dataset
    """
    print(f"Saving dataset to {output_path}...")
    if compress:
        adata.write_h5ad(output_path, compression='gzip')
    else:
        adata.write_h5ad(output_path)
    
    print(f"Dataset saved to {output_path}")
    return output_path


def prepare_spoticell_data(adata: sc.AnnData, 
                         output_dir: str,
                         cell_type_column: str = 'standardized_cell_type') -> None:
    """
    Prepare data in efficient format for Spoticell model training
    
    Parameters:
    -----------
    adata: sc.AnnData
        Preprocessed AnnData object
    output_dir: str
        Directory to save the processed data
    cell_type_column: str
        Column containing standardized cell type labels
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Start timing
    start_time = time.time()
    print("Starting efficient preprocessing for Spoticell...")

    # Get gene names and create position mapping
    gene_names = adata.var_names.tolist()
    matrix_size = math.ceil(math.sqrt(len(gene_names)))
    print(f"Matrix size will be {matrix_size}x{matrix_size} for {len(gene_names)} genes")

    # Create mapping of genes to positions (alphabetical order)
    print("Creating gene position mapping...")
    sorted_genes = sorted(gene_names)
    gene_to_idx = {gene: i for i, gene in enumerate(sorted_genes)}
    gene_positions = np.zeros((len(sorted_genes), 2), dtype=np.int32)

    for i, gene in enumerate(sorted_genes):
        row = i // matrix_size
        col = i % matrix_size
        gene_positions[i] = [row, col]

    # Create mapping from original gene index to sorted position
    gene_idx_map = np.zeros(len(gene_names), dtype=np.int32)
    for i, gene in enumerate(gene_names):
        gene_idx_map[i] = gene_to_idx.get(gene, 0)

    # Create cell type mappings
    print("Creating cell type mappings...")
    unique_cell_types = sorted(adata.obs[cell_type_column].unique())
    cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(unique_cell_types)}
    n_cell_types = len(unique_cell_types)

    # Save the sparse expression matrix directly
    print("Saving sparse expression matrix...")
    output_file = os.path.join(output_dir, "expression_data.npz")
    if sparse.issparse(adata.X):
        sparse.save_npz(output_file, adata.X)
    else:
        sparse.save_npz(output_file, sparse.csr_matrix(adata.X))

    # Save gene position mapping
    print("Saving gene position mapping...")
    np.save(os.path.join(output_dir, "gene_positions.npy"), gene_positions)
    np.save(os.path.join(output_dir, "gene_idx_map.npy"), gene_idx_map)

    # Save cell type labels
    print("Saving cell type labels...")
    cell_labels = np.zeros((adata.n_obs, n_cell_types), dtype=np.float32)
    for i, cell_type in enumerate(adata.obs[cell_type_column]):
        if cell_type in cell_type_to_idx:
            cell_labels[i, cell_type_to_idx[cell_type]] = 1.0
    np.save(os.path.join(output_dir, "cell_labels.npy"), cell_labels)

    # Save cell IDs
    print("Saving cell IDs...")
    np.save(os.path.join(output_dir, "cell_ids.npy"), np.array(adata.obs_names))

    # Save sorted gene names for reference
    print("Saving gene reference data...")
    np.save(os.path.join(output_dir, "sorted_genes.npy"), np.array(sorted_genes))
    np.save(os.path.join(output_dir, "original_genes.npy"), np.array(gene_names))

    # Save metadata
    metadata = {
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'matrix_size': int(matrix_size),
        'cell_type_to_idx': {k: int(v) for k, v in cell_type_to_idx.items()},
        'idx_to_cell_type': {str(idx): cell_type for idx, cell_type in enumerate(unique_cell_types)}
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    elapsed_time = time.time() - start_time
    print(f"Data preparation completed in {elapsed_time:.2f} seconds.")
    print(f"Files saved to {output_dir}")

    # Print a small test to confirm proper matrix creation
    print("\nTesting matrix creation for first cell...")
    if sparse.issparse(adata.X):
        test_expr = adata.X[0].toarray().flatten()
    else:
        test_expr = adata.X[0]

    # Create test matrix
    test_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    non_zero_count = 0

    # Only add non-zero genes
    non_zero_indices = np.nonzero(test_expr)[0]
    for gene_idx in non_zero_indices:
        # Map original gene index to sorted position
        matrix_idx = gene_idx_map[gene_idx]
        row, col = gene_positions[matrix_idx]
        test_matrix[row, col] = test_expr[gene_idx]
        non_zero_count += 1

    print(f"First cell has {non_zero_count} non-zero genes out of {len(gene_names)}")
    print(f"Matrix shape: {test_matrix.shape}")
    print(f"Sum of expression values: {np.sum(test_matrix):.4f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process scRNA-seq data for Spoticell')
    parser.add_argument('--output_dir', type=str, default='spoticell_data_efficient',
                        help='Directory to save processed data')
    parser.add_argument('--combined_h5ad', type=str, default='data/processed/combined_data.h5ad',
                        help='Path to save combined h5ad file')
    parser.add_argument('--cell_type_col', type=str, default='standardized_cell_type',
                        help='Column name for standardized cell types')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip downloading datasets (use existing files)')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # URLs for datasets
    h5ad_urls = [
        "https://datasets.cellxgene.cziscience.com/e4ac88b4-5eec-4519-900b-2482b19ef65b.h5ad", # human cHSPCs - Ultima 1652MB 516,121 cells
        "https://datasets.cellxgene.cziscience.com/5311ca08-a915-4bea-a83c-5f2231ba18ef.h5ad", # Lung 2644MB 318,426
        "https://datasets.cellxgene.cziscience.com/03107eab-699e-4c39-977c-25d685345a33.h5ad", # Liver 1627MB 259,678
        "https://datasets.cellxgene.cziscience.com/89619149-162f-4839-8e97-24735924417c.h5ad", # PBMC 4256MB 422,220
        "https://datasets.cellxgene.cziscience.com/b2e5ab44-6df3-4136-b452-2f3a6d4b4662.h5ad", # A cell atlas of human thymic development defines T cell repertoire formation 2511MB 255,901
        "https://datasets.cellxgene.cziscience.com/07405240-4f64-4dd9-83c3-b3db3405b05c.h5ad", # A multi-tissue single-cell tumor microenvironment atlas 2494MB 391,963
        "https://datasets.cellxgene.cziscience.com/369e3ca7-5e0f-417a-ac06-641e9f274b10.h5ad", # Blood 2125MB 335,916
    ]
    
    # Download data if not skipping
    downloaded_files = []
    if not args.skip_download:
        for i, url in enumerate(h5ad_urls):
            filename = f"dataset_{i}.h5ad"
            output_path = os.path.join('data/raw', filename)
            downloaded_file = download_h5ad_file(url, output_path)
            downloaded_files.append(downloaded_file)
    else:
        # If skipping download, just use existing files
        for i in range(len(h5ad_urls)):
            filename = f"dataset_{i}.h5ad"
            file_path = os.path.join('data/raw', filename)
            if os.path.exists(file_path):
                downloaded_files.append(file_path)
        
        if not downloaded_files:
            print("No downloaded files found. Cannot skip download.")
            return
    
    # Load and process datasets
    print(f"Processing {len(downloaded_files)} datasets...")
    adatas = []
    
    # Load datasets
    for file_path in downloaded_files:
        try:
            adata = sc.read_h5ad(file_path)
            print(f"Loaded {file_path}: {adata.n_obs} cells, {adata.n_vars} genes")
            adatas.append(adata)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Filter datasets for relevant cell types
    filtered_adatas = []
    
    # Dataset 0: hematopoietic cells
    adata_0 = adatas[0]
    adata_0 = adata_0[adata_0.obs['cell_type'].isin([
        'hematopoietic precursor cell', 
        'hematopoietic multipotent progenitor cell', 
        'B cell', 
        'monocyte', 
        'dendritic cell'
    ])].copy()
    adata_0 = adata_0[adata_0.obs['disease'].isin(['normal'])].copy()
    adata_0.obs['dataset'] = 'dataset_0'
    filtered_adatas.append(adata_0)
    print(f"Filtered dataset 0: {adata_0.n_obs} cells remaining")
    
    # Dataset 1: Lung
    adata_1 = adatas[1]
    adata_1 = adata_1[adata_1.obs['cell_type'].isin([
        'alveolar macrophage', 'macrophage', 'CD16-positive, CD56-dim natural killer cell, human', 
        'pulmonary alveolar type 2 cell', 'classical monocyte', 'capillary endothelial cell', 
        'effector memory CD4-positive, alpha-beta T cell', 'effector memory CD8-positive, alpha-beta T cell', 
        'fibroblast', 'alveolar capillary type 2 endothelial cell', 'mast cell', 
        'effector memory CD8-positive, alpha-beta T cell, terminally differentiated', 
        'vein endothelial cell', 'fibroblast of lung', 'lung ciliated cell', 
        'non-classical monocyte', 'endothelial cell of lymphatic vessel', 
        'endothelial cell of artery', 'pulmonary alveolar type 1 cell', 
        'smooth muscle cell of the pulmonary artery', 'memory B cell'
    ])].copy()
    adata_1 = adata_1[adata_1.obs['disease'].isin(['normal'])].copy()
    adata_1.obs['dataset'] = 'dataset_1'
    filtered_adatas.append(adata_1)
    print(f"Filtered dataset 1: {adata_1.n_obs} cells remaining")
    
    # Dataset 2: Liver
    adata_2 = adatas[2]
    adata_2 = adata_2[adata_2.obs['cell_type'].isin([
        'effector memory CD8-positive, alpha-beta T cell', 'periportal region hepatocyte', 
        'centrilobular region hepatocyte', 'mucosal invariant T cell', 'classical monocyte', 
        'effector memory CD8-positive, alpha-beta T cell, terminally differentiated', 
        'CD16-positive, CD56-dim natural killer cell, human', 
        'CD16-negative, CD56-bright natural killer cell, human', 'non-classical monocyte', 
        'endothelial cell of pericentral hepatic sinusoid', 'midzonal region hepatocyte', 
        'inflammatory macrophage', 'macrophage', 'CD4-positive, alpha-beta T cell', 
        'leukocyte', 'cholangiocyte', 'neutrophil'
    ])].copy()
    adata_2 = adata_2[adata_2.obs['disease'].isin(['normal'])].copy()
    adata_2.obs['dataset'] = 'dataset_2'
    filtered_adatas.append(adata_2)
    print(f"Filtered dataset 2: {adata_2.n_obs} cells remaining")
    
    # Dataset 3: PBMC
    adata_3 = adatas[3]
    adata_3 = adata_3[adata_3.obs['cell_type'].isin([
        'naive thymus-derived CD4-positive, alpha-beta T cell', 'classical monocyte', 
        'natural killer cell', 'naive B cell', 'CD4-positive helper T cell', 
        'CD8-positive, alpha-beta cytotoxic T cell', 
        'naive thymus-derived CD8-positive, alpha-beta T cell', 
        'central memory CD8-positive, alpha-beta T cell', 'non-classical monocyte', 
        'regulatory T cell', 'effector memory CD8-positive, alpha-beta T cell, terminally differentiated', 
        'CD4-positive, alpha-beta cytotoxic T cell', 'gamma-delta T cell', 
        'CD16-negative, CD56-bright natural killer cell, human', 
        'class switched memory B cell', 'mucosal invariant T cell'
    ])].copy()
    adata_3 = adata_3[adata_3.obs['disease'].isin(['normal'])].copy()
    adata_3.obs['dataset'] = 'dataset_3'
    filtered_adatas.append(adata_3)
    print(f"Filtered dataset 3: {adata_3.n_obs} cells remaining")
    
    # Dataset 4: Thymic
    adata_4 = adatas[4]
    adata_4 = adata_4[adata_4.obs['cell_type'].isin([
        'double-positive, alpha-beta thymocyte', 'double negative thymocyte', 
        'CD4-positive, alpha-beta T cell', 'CD8-positive, alpha-beta T cell', 
        'fibroblast', 'alpha-beta T cell', 'cortical thymic epithelial cell', 
        'regulatory T cell', 'medullary thymic epithelial cell', 
        'CD8-alpha-alpha-positive, alpha-beta intraepithelial T cell', 
        'endothelial cell', 'vascular associated smooth muscle cell', 
        'T cell', 'CD4-positive, alpha-beta memory T cell', 'dendritic cell', 
        'gamma-delta T cell', 'memory B cell', 'naive B cell', 
        'natural killer cell', 'CD8-positive, alpha-beta memory T cell'
    ])].copy()
    adata_4 = adata_4[adata_4.obs['disease'].isin(['normal'])].copy()
    adata_4.obs['dataset'] = 'dataset_4'
    filtered_adatas.append(adata_4)
    print(f"Filtered dataset 4: {adata_4.n_obs} cells remaining")
    
    # Dataset 5: Tumor microenvironment
    adata_5 = adatas[5]
    adata_5 = adata_5[adata_5.obs['cell_type'].isin([
        'T cell', 'epithelial cell', 'mononuclear phagocyte', 'fibroblast', 
        'malignant cell', 'endothelial cell', 'B cell', 'neutrophil', 'mast cell'
    ])].copy()
    adata_5 = adata_5[adata_5.obs['disease'].isin(['normal'])].copy()
    adata_5.obs['dataset'] = 'dataset_5'
    filtered_adatas.append(adata_5)
    print(f"Filtered dataset 5: {adata_5.n_obs} cells remaining")
    
    # Dataset 6: Blood
    adata_6 = adatas[6]
    adata_6 = adata_6[adata_6.obs['cell_type'].isin([
        'central memory CD4-positive, alpha-beta T cell', 
        'CD16-positive, CD56-dim natural killer cell, human', 'classical monocyte', 
        'effector memory CD8-positive, alpha-beta T cell, terminally differentiated', 
        'effector memory CD4-positive, alpha-beta T cell', 
        'effector memory CD8-positive, alpha-beta T cell', 
        'central memory CD8-positive, alpha-beta T cell', 'naive B cell', 
        'gamma-delta T cell', 'non-classical monocyte', 'mucosal invariant T cell', 
        'class switched memory B cell', 'unswitched memory B cell', 
        'conventional dendritic cell', 'memory B cell', 'regulatory T cell', 
        'effector memory CD4-positive, alpha-beta T cell, terminally differentiated', 
        'megakaryocyte', 'plasmacytoid dendritic cell', 
        'CD16-negative, CD56-bright natural killer cell, human'
    ])].copy()
    adata_6 = adata_6[adata_6.obs['disease'].isin(['normal'])].copy()
    adata_6.obs['dataset'] = 'dataset_6'
    filtered_adatas.append(adata_6)
    print(f"Filtered dataset 6: {adata_6.n_obs} cells remaining")
    
    # Combine all filtered datasets
    print("Combining filtered datasets...")
    combined_adata = filtered_adatas[0].concatenate(
        filtered_adatas[1:],
        join='inner',  # Use only genes that are found in all datasets
        index_unique='-',  # Make observation names unique by adding a suffix
        batch_key='batch',  # Add batch key to track dataset origin
        batch_categories=[f'dataset_{i}' for i in range(len(filtered_adatas))]
    )
    
    print(f"Combined object has {combined_adata.n_obs} cells and {combined_adata.n_vars} genes")
    print(f"Cell type categories: {combined_adata.obs['cell_type'].unique()}")
    
    # Save combined data
    combined_adata.write_h5ad(args.combined_h5ad)
    print(f"Saved combined data to {args.combined_h5ad}")
    
    # Create a standardization mapping for cell types with underscore format
    cell_type_mapping = {
        # T cell related mappings
        "T cell": "T_cell",
        "alpha-beta T cell": "alpha_beta_T_cell",  
        "CD4-positive, alpha-beta T cell": "CD4_positive_alpha_beta_T_cell",
        "CD4-positive helper T cell": "CD4_positive_helper_T_cell",
        "CD4-positive, alpha-beta memory T cell": "CD4_positive_alpha_beta_memory_T_cell",
        "CD4-positive, alpha-beta cytotoxic T cell": "CD4_positive_alpha_beta_cytotoxic_T_cell",
        "naive thymus-derived CD4-positive, alpha-beta T cell": "naive_thymus_derived_CD4_positive_alpha_beta_T_cell",
        "central memory CD4-positive, alpha-beta T cell": "central_memory_CD4_positive_alpha_beta_T_cell",
        "effector memory CD4-positive, alpha-beta T cell": "effector_memory_CD4_positive_alpha_beta_T_cell",
        "effector memory CD4-positive, alpha-beta T cell, terminally differentiated": "effector_memory_CD4_positive_alpha_beta_T_cell_terminally_differentiated",
        
        "CD8-positive, alpha-beta T cell": "CD8_positive_alpha_beta_T_cell",
        "CD8-positive, alpha-beta cytotoxic T cell": "CD8_positive_alpha_beta_cytotoxic_T_cell",
        "CD8-positive, alpha-beta memory T cell": "CD8_positive_alpha_beta_memory_T_cell",
        "naive thymus-derived CD8-positive, alpha-beta T cell": "naive_thymus_derived_CD8_positive_alpha_beta_T_cell",
        "central memory CD8-positive, alpha-beta T cell": "central_memory_CD8_positive_alpha_beta_T_cell",
        "effector memory CD8-positive, alpha-beta T cell": "effector_memory_CD8_positive_alpha_beta_T_cell",
        "effector memory CD8-positive, alpha-beta T cell, terminally differentiated": "effector_memory_CD8_positive_alpha_beta_T_cell_terminally_differentiated",
        "CD8-alpha-alpha-positive, alpha-beta intraepithelial T cell": "CD8_alpha_alpha_positive_alpha_beta_intraepithelial_T_cell",
        
        "gamma-delta T cell": "gamma_delta_T_cell",
        "mucosal invariant T cell": "mucosal_invariant_T_cell",
        "regulatory T cell": "regulatory_T_cell",
        "double-positive, alpha-beta thymocyte": "double_positive_alpha_beta_thymocyte",
        "double negative thymocyte": "double_negative_thymocyte",
        
        # B cell related mappings
        "B cell": "B_cell",
        "naive B cell": "naive_B_cell",
        "memory B cell": "memory_B_cell",
        "class switched memory B cell": "class_switched_memory_B_cell",
        "unswitched memory B cell": "unswitched_memory_B_cell",
        
        # NK cell related mappings
        "natural killer cell": "natural_killer_cell",
        "CD16-positive, CD56-dim natural killer cell, human": "CD16_positive_CD56_dim_natural_killer_cell",
        "CD16-negative, CD56-bright natural killer cell, human": "CD16_negative_CD56_bright_natural_killer_cell",
        
        # Monocyte/macrophage related mappings
        "monocyte": "monocyte",
        "classical monocyte": "classical_monocyte",
        "non-classical monocyte": "non_classical_monocyte",
        "macrophage": "macrophage",
        "alveolar macrophage": "alveolar_macrophage",
        "inflammatory macrophage": "inflammatory_macrophage",
        "mononuclear phagocyte": "mononuclear_phagocyte",
        
        # Dendritic cell mappings
        "dendritic cell": "dendritic_cell",
        "conventional dendritic cell": "conventional_dendritic_cell",
        "plasmacytoid dendritic cell": "plasmacytoid_dendritic_cell",
        
        # Progenitor cell mappings
        "hematopoietic precursor cell": "hematopoietic_precursor_cell",
        "hematopoietic multipotent progenitor cell": "hematopoietic_multipotent_progenitor_cell",
        
        # Other immune cell types
        "leukocyte": "leukocyte",
        "neutrophil": "neutrophil",
        "mast cell": "mast_cell",
        "megakaryocyte": "megakaryocyte",
        
        # Epithelial cells
        "epithelial cell": "epithelial_cell",
        "cortical thymic epithelial cell": "cortical_thymic_epithelial_cell",
        "medullary thymic epithelial cell": "medullary_thymic_epithelial_cell",
        "pulmonary alveolar type 1 cell": "pulmonary_alveolar_type_1_cell",
        "pulmonary alveolar type 2 cell": "pulmonary_alveolar_type_2_cell",
        "lung ciliated cell": "lung_ciliated_cell",
        "cholangiocyte": "cholangiocyte",
        
        # Hepatocytes
        "periportal region hepatocyte": "periportal_region_hepatocyte",
        "centrilobular region hepatocyte": "centrilobular_region_hepatocyte",
        "midzonal region hepatocyte": "midzonal_region_hepatocyte",
        
        # Endothelial cells
        "endothelial cell": "endothelial_cell",
        "capillary endothelial cell": "capillary_endothelial_cell",
        "endothelial cell of pericentral hepatic sinusoid": "endothelial_cell_of_pericentral_hepatic_sinusoid",
        "alveolar capillary type 2 endothelial cell": "alveolar_capillary_type_2_endothelial_cell",
        "endothelial cell of lymphatic vessel": "endothelial_cell_of_lymphatic_vessel",
        "endothelial cell of artery": "endothelial_cell_of_artery",
        "vein endothelial cell": "vein_endothelial_cell",
        
        # Fibroblasts and smooth muscle cells
        "fibroblast": "fibroblast",
        "fibroblast of lung": "fibroblast_of_lung",
        "vascular associated smooth muscle cell": "vascular_associated_smooth_muscle_cell",
        "smooth muscle cell of the pulmonary artery": "smooth_muscle_cell_of_the_pulmonary_artery",
        
        # Other
        "malignant cell": "malignant_cell"
    }
    
    # Apply the mapping to standardize cell types
    combined_adata.obs['standardized_cell_type'] = combined_adata.obs['cell_type'].map(cell_type_mapping)
    
    # Check for any cell types that might not have been mapped
    unmapped = combined_adata.obs[combined_adata.obs['standardized_cell_type'].isna()]['cell_type'].unique()
    if len(unmapped) > 0:
        print("Unmapped cell types:", unmapped)
        # Assign unmapped types to their original values with underscore formatting
        for cell_type in unmapped:
            # Replace spaces, commas, hyphens with underscores
            formatted = cell_type.replace(' ', '_').replace(',', '').replace('-', '_')
            # Replace consecutive underscores with single underscore
            while '__' in formatted:
                formatted = formatted.replace('__', '_')
            # Remove 'human' suffix if present
            formatted = formatted.replace('_human', '')
            # Remove trailing underscores
            formatted = formatted.rstrip('_')
            
            combined_adata.obs.loc[combined_adata.obs['cell_type'] == cell_type, 'standardized_cell_type'] = formatted
            cell_type_mapping[cell_type] = formatted
    
    # Verify the standardization
    print("Total number of standardized cell types:", combined_adata.obs['standardized_cell_type'].nunique())
    print("Standardized cell type counts:")
    print(combined_adata.obs['standardized_cell_type'].value_counts())
    
    # Save the standardized combined data
    combined_adata.write_h5ad(args.combined_h5ad.replace('.h5ad', '_standardized.h5ad'))
    print(f"Saved standardized combined data")
    
    # Prepare data for Spoticell model
    prepare_spoticell_data(combined_adata, args.output_dir, args.cell_type_col)
    
    print("All preprocessing steps completed successfully!")


if __name__ == "__main__":
    main()