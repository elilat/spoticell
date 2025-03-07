#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spoticell Prediction Script
---------------------------
Uses a trained Spoticell model to predict cell types from scRNA-seq data.
"""

import os
import argparse
import json
import numpy as np
import torch
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from train_model import SpoticellModel, SpoticellDataset


def load_model(model_path, device):
    """
    Load a trained Spoticell model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    device : torch.device
        Device to load the model on
        
    Returns:
    --------
    model : SpoticellModel
        Loaded model
    metadata : dict
        Model metadata
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get metadata from the same directory
    metadata_path = os.path.join(os.path.dirname(model_path), '..', 'metadata.json')
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(os.path.dirname(model_path), 'metadata.json')
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model with the same parameters
    model = SpoticellModel(
        matrix_size=metadata['matrix_size'],
        num_classes=len(metadata['cell_type_to_idx']),
        cnn_channels=32,
        embed_dim=256,
        transformer_heads=8,
        transformer_layers=2,
        dropout=0.1
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, metadata


def predict_cell_types(model, adata, gene_positions, gene_idx_map, matrix_size, cell_type_mapping, device, threshold=0.5):
    """
    Predict cell types for cells in an AnnData object.
    
    Parameters:
    -----------
    model : SpoticellModel
        Trained model
    adata : AnnData
        AnnData object with cells to predict
    gene_positions : numpy.ndarray
        Positions of genes in the matrix
    gene_idx_map : numpy.ndarray
        Mapping from original gene indices to sorted positions
    matrix_size : int
        Size of the matrix
    cell_type_mapping : dict
        Mapping from cell type indices to names
    device : torch.device
        Device to run prediction on
    threshold : float
        Probability threshold for positive predictions
        
    Returns:
    --------
    pred_df : pandas.DataFrame
        DataFrame with predictions for each cell
    """
    model.eval()
    num_cells = adata.n_obs
    num_classes = len(cell_type_mapping)
    
    # Create batches for prediction
    batch_size = 32
    all_probabilities = []
    
    with torch.no_grad():
        for i in range(0, num_cells, batch_size):
            batch_cells = adata[i:min(i+batch_size, num_cells)]
            
            # Create matrices for this batch
            batch_matrices = []
            for cell_idx in range(batch_cells.n_obs):
                # Get expression values for this cell
                if sparse.issparse(batch_cells.X):
                    cell_expr = batch_cells.X[cell_idx].toarray().flatten()
                else:
                    cell_expr = batch_cells.X[cell_idx]
                
                # Create matrix
                matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
                
                # Only add non-zero genes
                non_zero_indices = np.nonzero(cell_expr)[0]
                for gene_idx in non_zero_indices:
                    # Map original gene index to sorted position
                    matrix_idx = gene_idx_map[gene_idx]
                    row, col = gene_positions[matrix_idx]
                    matrix[row, col] = cell_expr[gene_idx]
                
                # Add channel dimension and convert to tensor
                matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
                batch_matrices.append(matrix)
            
            # Stack matrices and move to device
            batch_matrices = torch.stack(batch_matrices).to(device)
            
            # Get predictions
            logits = model(batch_matrices)
            probabilities = torch.sigmoid(logits)
            
            # Save probabilities
            all_probabilities.append(probabilities.cpu().numpy())
    
    # Combine all batches
    all_probabilities = np.vstack(all_probabilities)
    
    # Get predictions from probabilities
    predictions = (all_probabilities >= threshold).astype(int)
    
    # Create DataFrame with cell names and predictions
    pred_df = pd.DataFrame(index=adata.obs_names)
    
    # Add probabilities for each cell type
    for i, cell_type in enumerate(cell_type_mapping.values()):
        pred_df[f"{cell_type}_prob"] = all_probabilities[:, i]
        pred_df[f"{cell_type}_pred"] = predictions[:, i]
    
    # Add top predicted cell type
    top_indices = np.argmax(all_probabilities, axis=1)
    pred_df['top_predicted_type'] = [list(cell_type_mapping.values())[i] for i in top_indices]
    pred_df['top_probability'] = np.max(all_probabilities, axis=1)
    
    # Add number of predicted cell types (multi-label)
    pred_df['num_predicted_types'] = np.sum(predictions, axis=1)
    
    return pred_df


def plot_prediction_heatmap(pred_df, cell_type_mapping, output_path):
    """
    Plot a heatmap of prediction probabilities.
    
    Parameters:
    -----------
    pred_df : pandas.DataFrame
        DataFrame with predictions
    cell_type_mapping : dict
        Mapping from cell type indices to names
    output_path : str
        Path to save the plot
    """
    # Extract probability columns
    prob_cols = [f"{cell_type}_prob" for cell_type in cell_type_mapping.values()]
    prob_matrix = pred_df[prob_cols].values
    
    # Rename columns for plotting
    cell_types = list(cell_type_mapping.values())
    
    # Cluster cells by their probability profiles
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(10, len(pred_df)), random_state=42)
    clusters = kmeans.fit_predict(prob_matrix)
    
    # Sort cells by cluster and then by top predicted type
    pred_df['cluster'] = clusters
    sorted_indices = pred_df.sort_values(['cluster', 'top_predicted_type']).index
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        prob_matrix[sorted_indices.get_indexer(pred_df.index)],
        cmap='viridis',
        xticklabels=cell_types,
        yticklabels=False,
        cbar_kws={'label': 'Probability'}
    )
    plt.title('Cell Type Prediction Probabilities')
    plt.xlabel('Cell Type')
    plt.ylabel('Cells (clustered)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Predict cell types using Spoticell model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to h5ad file with cells to predict')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for positive predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, metadata = load_model(args.model_path, device)
    
    # Get required data from model directory
    model_dir = os.path.dirname(args.model_path)
    data_dir = os.path.join(os.path.dirname(model_dir), 'data')
    
    gene_positions = np.load(os.path.join(data_dir, 'gene_positions.npy'))
    gene_idx_map = np.load(os.path.join(data_dir, 'gene_idx_map.npy'))
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Get cell type mapping
    cell_type_mapping = metadata['idx_to_cell_type']
    
    # Predict cell types
    print("Predicting cell types...")
    pred_df = predict_cell_types(
        model, adata, gene_positions, gene_idx_map,
        metadata['matrix_size'], cell_type_mapping, device, args.threshold
    )
    
    # Save predictions
    output_path = os.path.join(args.output_dir, 'cell_type_predictions.csv')
    pred_df.to_csv(output_path)
    print(f"Saved predictions to {output_path}")
    
    # Plot prediction heatmap
    heatmap_path = os.path.join(args.output_dir, 'prediction_heatmap.png')
    plot_prediction_heatmap(pred_df, cell_type_mapping, heatmap_path)
    print(f"Saved prediction heatmap to {heatmap_path}")
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Total cells: {len(pred_df)}")
    print("\nTop predicted cell types:")
    top_types = pred_df['top_predicted_type'].value_counts()
    for cell_type, count in top_types.items():
        print(f"  - {cell_type}: {count} cells ({100 * count / len(pred_df):.2f}%)")
    
    # Analyze multi-label predictions
    print("\nNumber of cell types per cell:")
    type_counts = pred_df['num_predicted_types'].value_counts().sort_index()
    for count, frequency in type_counts.items():
        print(f"  - {count} types: {frequency} cells ({100 * frequency / len(pred_df):.2f}%)")
    
    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()