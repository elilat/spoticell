#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spoticell Model Training Script
-------------------------------
Trains the Spoticell model for cell type classification using data
preprocessed by preprocessing.py
"""

import os
import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from datetime import datetime
import sys

# ==================== CONFIGURABLE PARAMETERS ====================

# Model Architecture Parameters
MODEL_PARAMS = {
    'MATRIX_SIZE': None,  # Will be loaded from metadata.json
    'NUM_CLASSES': None,  # Will be loaded from metadata.json
    'CNN_CHANNELS': 16,  # Base number of CNN channels
    'EMBED_DIM': 128,  # Embedding dimension for transformer
    'TRANSFORMER_HEADS': 4,  # Number of attention heads in transformer
    'TRANSFORMER_LAYERS': 1,  # Number of transformer layers
    'DROPOUT': 0.1,  # Dropout rate
}

# Training Parameters
TRAINING_PARAMS = {
    'BATCH_SIZE': 16,  # Batch size for training
    'EPOCHS': 50,  # Number of training epochs
    'LEARNING_RATE': 0.001,  # Initial learning rate
    'WEIGHT_DECAY': 1e-5,  # Weight decay (L2 regularization)
    'PATIENCE': 10,  # Patience for early stopping
    'THRESHOLD': 0.5,  # Probability threshold for positive predictions
    'SCHEDULER_FACTOR': 0.5,  # Factor by which to reduce learning rate
    'SCHEDULER_PATIENCE': 5,  # Patience for learning rate scheduler
    'GRADIENT_ACCUMULATION_STEPS': 4,
}

# Data Parameters
DATA_PARAMS = {
    'DATA_DIR': 'spoticell_data_efficient',  # Directory containing preprocessed data
    'OUTPUT_DIR': 'models/spoticell_v1',  # Directory to save model checkpoints and logs
    'SUBSET_SIZE': None,  # Limit dataset to first N cells (for testing)
    'TRAIN_SPLIT': 0.7,  # Proportion of data for training
    'VAL_SPLIT': 0.15,  # Proportion of data for validation
    'NUM_WORKERS': 4,  # Number of workers for data loading
}

# Device Parameters
DEVICE_PARAMS = {
    'DEVICE': 'cuda',  # Device to use for training (cuda or cpu)
    'GPU_ID': 0,  # ID of GPU to use if multiple GPUs are available
}

# Reproducibility Parameters
SEED = 42  # Seed for random number generators

# Log Parameters
LOG_PARAMS = {
    'LOG_LEVEL': logging.INFO,  # Logging level
    'LOG_INTERVAL': 10,  # Log training progress every N batches
}

# ==================== SETUP FUNCTIONS ====================

def setup_logging(output_dir):
    """Set up logging configuration."""
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=LOG_PARAMS['LOG_LEVEL'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create a logger
    logger = logging.getLogger('spoticell')
    
    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    return logger

def set_seeds(seed=SEED):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Additional reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_name='cuda', gpu_id=0):
    """Get device for training."""
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    return device

def load_model_params(data_dir):
    """Load model parameters from metadata."""
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    MODEL_PARAMS['MATRIX_SIZE'] = metadata['matrix_size']
    
    # Fix for the n_cell_types key that doesn't exist in the metadata
    # Instead, use the length of cell_type_to_idx which contains all cell types
    if 'n_cell_types' in metadata:
        MODEL_PARAMS['NUM_CLASSES'] = metadata['n_cell_types']
    else:
        # Alternative way to get the number of cell types
        MODEL_PARAMS['NUM_CLASSES'] = len(metadata['cell_type_to_idx'])
        # Add this key to metadata for consistency
        metadata['n_cell_types'] = len(metadata['cell_type_to_idx'])
        logging.info(f"Added missing 'n_cell_types' key to metadata: {MODEL_PARAMS['NUM_CLASSES']}")
    
    return metadata


# ==================== DATASET CLASS ====================

class SpoticellDataset(Dataset):
    """
    Dataset for Spoticell that creates matrices on-the-fly from sparse data.
    """
    def __init__(self, data_dir, transform=None, subset_size=None):
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Load gene positions and mapping
        self.gene_positions = np.load(os.path.join(data_dir, 'gene_positions.npy'))
        self.gene_idx_map = np.load(os.path.join(data_dir, 'gene_idx_map.npy'))
        
        # Load expression data
        self.X = sparse.load_npz(os.path.join(data_dir, 'expression_data.npz'))
        
        # Load labels
        self.labels = np.load(os.path.join(data_dir, 'cell_labels.npy'))
        
        # Set matrix size
        self.matrix_size = self.metadata['matrix_size']
        
        # Optional transform
        self.transform = transform
        
        # Optional subset for testing
        if subset_size is not None and subset_size < self.X.shape[0]:
            self.X = self.X[:subset_size]
            self.labels = self.labels[:subset_size]
            logging.info(f"Using subset of {subset_size} cells for testing")
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # Get expression values for this cell
        if sparse.issparse(self.X):
            cell_expr = self.X[idx].toarray().flatten()
        else:
            cell_expr = self.X[idx]
        
        # Create matrix on-the-fly
        matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=np.float32)
        
        # Only add non-zero genes (for efficiency)
        non_zero_indices = np.nonzero(cell_expr)[0]
        for gene_idx in non_zero_indices:
            # Map original gene index to sorted position
            matrix_idx = self.gene_idx_map[gene_idx]
            row, col = self.gene_positions[matrix_idx]
            matrix[row, col] = cell_expr[gene_idx]
        
        # Apply transform if provided
        if self.transform:
            matrix = self.transform(matrix)
        else:
            # Convert to tensor (add channel dimension)
            matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return matrix, label


# ==================== MODEL ARCHITECTURE ====================

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + residual)
        return x


class TransformerBlock(nn.Module):
    """Simple transformer block for processing gene relationships."""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim)
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class SpoticellModel(nn.Module):
    """
    Spoticell model for cell type classification from gene expression matrices.
    
    Uses a dual-path architecture with:
    1. CNN path to process spatial patterns in expression matrices
    2. Transformer path to capture gene relationships independent of position
    """
    
    def __init__(self, matrix_size, num_classes, cnn_channels=32, embed_dim=256, 
                 transformer_heads=8, transformer_layers=2, dropout=0.1):
        super(SpoticellModel, self).__init__()
        
        # Model parameters
        self.matrix_size = matrix_size
        self.num_classes = num_classes
        self.cnn_channels = cnn_channels
        self.embed_dim = embed_dim
        
        # CNN path
        self.cnn_path = nn.Sequential(
            ConvBlock(1, cnn_channels),
            nn.MaxPool2d(2),
            ConvBlock(cnn_channels, cnn_channels * 2),
            nn.MaxPool2d(2),
            ConvBlock(cnn_channels * 2, cnn_channels * 4),
            nn.MaxPool2d(2),
            ConvBlock(cnn_channels * 4, cnn_channels * 8),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Calculate CNN output size
        cnn_output_dim = cnn_channels * 8
        
        # Transformer path
        # First reshape matrix to sequence and embed
        self.flatten_dim = min(1024, matrix_size * matrix_size)  # Limit sequence length for transformer
        self.embed = nn.Linear(1, embed_dim)  # Embed each gene expression value
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(self.flatten_dim, 1, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, transformer_heads, dropout)
            for _ in range(transformer_layers)
        ])
        
        # Transformer output projection
        self.transformer_output = nn.Linear(embed_dim * self.flatten_dim, cnn_output_dim)
        
        # Fusion and classification
        fusion_dim = cnn_output_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head with multi-label capability
        self.classifier = nn.Linear(fusion_dim // 4, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize positional encoding
        nn.init.normal_(self.pos_encoding, mean=0, std=0.01)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN path
        cnn_out = self.cnn_path(x)
        cnn_out = cnn_out.view(batch_size, -1)
        
        # Transformer path
        # Reshape to sequence: (batch_size, channels, height, width) -> (batch_size, seq_len, 1)
        x_flat = x.view(batch_size, 1, -1)
        x_flat = x_flat[:, :, :self.flatten_dim].transpose(1, 2)  # Limit sequence length
        
        # Embed sequence: (batch_size, seq_len, 1) -> (batch_size, seq_len, embed_dim)
        x_embed = self.embed(x_flat)
        
        # Prepare for transformer: (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x_embed = x_embed.transpose(0, 1)
        
        # Add positional encoding
        x_embed = x_embed + self.pos_encoding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x_embed = layer(x_embed)
        
        # Return to batch first: (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
        x_embed = x_embed.transpose(0, 1)
        
        # Flatten and project: (batch_size, seq_len, embed_dim) -> (batch_size, projection_dim)
        x_embed = x_embed.reshape(batch_size, -1)
        transformer_out = self.transformer_output(x_embed)
        
        # Fusion
        fused = torch.cat([cnn_out, transformer_out], dim=1)
        fused = self.fusion(fused)
        
        # Classification (multi-label)
        logits = self.classifier(fused)
        
        return logits

    def predict(self, x, threshold=0.5):
        """Make predictions with threshold."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()
            
        return probabilities, predictions


class SpoticellLoss(nn.Module):
    """
    Custom loss function for multi-label cell type classification.
    """
    
    def __init__(self, hierarchy_matrix=None, weight=None, reduction='mean'):
        super(SpoticellLoss, self).__init__()
        self.hierarchy_matrix = hierarchy_matrix
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        
    def forward(self, logits, targets):
        # Basic binary cross-entropy loss
        bce_loss = self.bce(logits, targets)
        
        # If hierarchy matrix is provided, add hierarchical consistency term
        if self.hierarchy_matrix is not None and self.hierarchy_matrix.device != logits.device:
            self.hierarchy_matrix = self.hierarchy_matrix.to(logits.device)
            
        if self.hierarchy_matrix is not None:
            probs = torch.sigmoid(logits)
            
            # For each child, its probability should be <= its parent's probability
            child_probs = probs.unsqueeze(1)  # (batch, 1, num_classes)
            parent_probs = probs.unsqueeze(2)  # (batch, num_classes, 1)
            
            # Apply hierarchy: get all parent-child pairs
            # hierarchy_matrix[i,j]=1 means i is parent of j
            hierarchy_violations = torch.relu(
                child_probs - parent_probs
            ) * self.hierarchy_matrix.unsqueeze(0)
            
            # Sum all violations
            hierarchy_loss = hierarchy_violations.sum(dim=(1, 2)).mean()
            
            # Combine losses
            total_loss = bce_loss + 0.1 * hierarchy_loss
            return total_loss
        
        return bce_loss


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, dataloader, criterion, optimizer, device, logger, epoch, threshold=0.5):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    # Number of batches
    num_batches = len(dataloader)
    
    # Start time
    start_time = time.time()
    
    for batch_idx, (matrices, targets) in enumerate(dataloader):
        # Move data to device
        matrices = matrices.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(matrices)
        
        # Compute loss
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * matrices.size(0)
        
        # Get predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        
        # Collect predictions and targets for metrics
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(preds.cpu().numpy())
        
        # Log progress at intervals
        if (batch_idx + 1) % LOG_PARAMS['LOG_INTERVAL'] == 0:
            elapsed_time = time.time() - start_time
            batch_loss = loss.item()
            progress = 100. * (batch_idx + 1) / num_batches
            
            logger.info(f"Epoch {epoch} [{batch_idx+1}/{num_batches} ({progress:.0f}%)] - "
                        f"Loss: {batch_loss:.4f} - "
                        f"Time: {elapsed_time:.2f}s")
            
            # Reset timer
            start_time = time.time()
    
    # Compute epoch loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Compute metrics
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    return epoch_loss, precision, recall, f1


def validate(model, dataloader, criterion, device, threshold=0.5):
    """Validate model on validation set."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for matrices, targets in dataloader:
            # Move data to device
            matrices = matrices.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(matrices)
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Update statistics
            running_loss += loss.item() * matrices.size(0)
            
            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            # Collect predictions and targets for metrics
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
    
    # Compute epoch loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Compute metrics
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    return epoch_loss, precision, recall, f1


def test(model, dataloader, criterion, device, threshold=0.5):
    """Test model on test set."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for matrices, targets in dataloader:
            # Move data to device
            matrices = matrices.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(matrices)
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Update statistics
            running_loss += loss.item() * matrices.size(0)
            
            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            # Collect predictions, probabilities, and targets for metrics
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_probabilities.append(probs.cpu().numpy())
    
    # Compute epoch loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Compute metrics
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    # Compute per-class metrics
    per_class_precision = precision_score(all_targets, all_predictions, average=None, zero_division=0)
    per_class_recall = recall_score(all_targets, all_predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    
    return (epoch_loss, precision, recall, f1, 
            per_class_precision, per_class_recall, per_class_f1, all_probabilities)


def plot_training_history(train_losses, val_losses, train_f1s, val_f1s, output_dir):
    """Plot training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def save_model(model, epoch, optimizer, scheduler, train_losses, val_losses, train_f1s, val_f1s, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s
    }, path)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Spoticell model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=DATA_PARAMS['DATA_DIR'],
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default=DATA_PARAMS['OUTPUT_DIR'],
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--subset_size', type=int, default=DATA_PARAMS['SUBSET_SIZE'],
                        help='Limit dataset to first N cells (for testing)')
    
    # Model parameters
    parser.add_argument('--cnn_channels', type=int, default=MODEL_PARAMS['CNN_CHANNELS'],
                        help='Base number of CNN channels')
    parser.add_argument('--embed_dim', type=int, default=MODEL_PARAMS['EMBED_DIM'],
                        help='Embedding dimension for transformer')
    parser.add_argument('--transformer_heads', type=int, default=MODEL_PARAMS['TRANSFORMER_HEADS'],
                        help='Number of attention heads in transformer')
    parser.add_argument('--transformer_layers', type=int, default=MODEL_PARAMS['TRANSFORMER_LAYERS'],
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=MODEL_PARAMS['DROPOUT'],
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=TRAINING_PARAMS['BATCH_SIZE'],
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=TRAINING_PARAMS['EPOCHS'],
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=TRAINING_PARAMS['LEARNING_RATE'],
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=TRAINING_PARAMS['WEIGHT_DECAY'],
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=TRAINING_PARAMS['PATIENCE'],
                        help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=DATA_PARAMS['NUM_WORKERS'],
                        help='Number of workers for data loading')
    parser.add_argument('--threshold', type=float, default=TRAINING_PARAMS['THRESHOLD'],
                        help='Probability threshold for positive predictions')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=DEVICE_PARAMS['DEVICE'],
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--gpu_id', type=int, default=DEVICE_PARAMS['GPU_ID'],
                        help='ID of GPU to use if multiple GPUs are available')
    
    # Data split parameters
    parser.add_argument('--train_split', type=float, default=DATA_PARAMS['TRAIN_SPLIT'],
                        help='Proportion of data for training')
    parser.add_argument('--val_split', type=float, default=DATA_PARAMS['VAL_SPLIT'],
                        help='Proportion of data for validation')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=LOG_PARAMS['LOG_INTERVAL'],
                        help='Log training progress every N batches')
    
    return parser.parse_args()


# ==================== MAIN FUNCTION ====================

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Update parameters from arguments
    DATA_PARAMS['DATA_DIR'] = args.data_dir
    DATA_PARAMS['OUTPUT_DIR'] = args.output_dir
    DATA_PARAMS['SUBSET_SIZE'] = args.subset_size
    DATA_PARAMS['TRAIN_SPLIT'] = args.train_split
    DATA_PARAMS['VAL_SPLIT'] = args.val_split
    DATA_PARAMS['NUM_WORKERS'] = args.num_workers
    
    MODEL_PARAMS['CNN_CHANNELS'] = args.cnn_channels
    MODEL_PARAMS['EMBED_DIM'] = args.embed_dim
    MODEL_PARAMS['TRANSFORMER_HEADS'] = args.transformer_heads
    MODEL_PARAMS['TRANSFORMER_LAYERS'] = args.transformer_layers
    MODEL_PARAMS['DROPOUT'] = args.dropout
    
    TRAINING_PARAMS['BATCH_SIZE'] = args.batch_size
    TRAINING_PARAMS['EPOCHS'] = args.epochs
    TRAINING_PARAMS['LEARNING_RATE'] = args.lr
    TRAINING_PARAMS['WEIGHT_DECAY'] = args.weight_decay
    TRAINING_PARAMS['PATIENCE'] = args.patience
    TRAINING_PARAMS['THRESHOLD'] = args.threshold
    
    DEVICE_PARAMS['DEVICE'] = args.device
    DEVICE_PARAMS['GPU_ID'] = args.gpu_id
    
    LOG_PARAMS['LOG_INTERVAL'] = args.log_interval
    
    # Create output directory and setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    # Set seeds for reproducibility
    set_seeds()
    
    # Set device
    device = get_device(DEVICE_PARAMS['DEVICE'], DEVICE_PARAMS['GPU_ID'])
    logger.info(f"Using device: {device}")
    
    # Save arguments
    params_dict = {
        'MODEL_PARAMS': MODEL_PARAMS,
        'TRAINING_PARAMS': TRAINING_PARAMS,
        'DATA_PARAMS': DATA_PARAMS,
        'DEVICE_PARAMS': DEVICE_PARAMS,
        'LOG_PARAMS': LOG_PARAMS,
        'SEED': SEED
    }
    
    with open(os.path.join(args.output_dir, 'params.json'), 'w') as f:
        json.dump(params_dict, f, indent=4)
    
    # Load model parameters from metadata
    metadata = load_model_params(args.data_dir)
    logger.info(f"Loaded metadata with matrix size {MODEL_PARAMS['MATRIX_SIZE']} and {MODEL_PARAMS['NUM_CLASSES']} cell types")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = SpoticellDataset(args.data_dir, subset_size=args.subset_size)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(args.train_split * dataset_size)
    val_size = int(args.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    logger.info(f"Total dataset size: {dataset_size}")
    logger.info(f"Training set size: {train_size}")
    logger.info(f"Validation set size: {val_size}")
    logger.info(f"Test set size: {test_size}")
    
    # Create training indices and randomly shuffle them
    indices = list(range(dataset_size))
    random.shuffle(indices)  # Shuffle the indices
    
    # Split indices for train, validation, and test sets
    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + val_size)]
    test_indices = indices[(train_size + val_size):]
    
    # Create data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = SpoticellModel(
        matrix_size=MODEL_PARAMS['MATRIX_SIZE'],
        num_classes=MODEL_PARAMS['NUM_CLASSES'],
        cnn_channels=MODEL_PARAMS['CNN_CHANNELS'],
        embed_dim=MODEL_PARAMS['EMBED_DIM'],
        transformer_heads=MODEL_PARAMS['TRANSFORMER_HEADS'],
        transformer_layers=MODEL_PARAMS['TRANSFORMER_LAYERS'],
        dropout=MODEL_PARAMS['DROPOUT']
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create criterion, optimizer, and scheduler
    criterion = SpoticellLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=TRAINING_PARAMS['LEARNING_RATE'], 
        weight_decay=TRAINING_PARAMS['WEIGHT_DECAY']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=TRAINING_PARAMS['SCHEDULER_FACTOR'], 
        patience=TRAINING_PARAMS['SCHEDULER_PATIENCE'], 
        verbose=True
    )
    
    # Initialize training history
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    # Initialize best validation loss and F1 score
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Create a CSV file for detailed tracking
    csv_path = os.path.join(args.output_dir, 'training_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,train_precision,val_precision,train_recall,val_recall,train_f1,val_f1,learning_rate\n")
    
    # Training loop
    logger.info("Starting training...")
    total_start_time = time.time()
    
    for epoch in range(TRAINING_PARAMS['EPOCHS']):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_precision, train_recall, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch + 1, TRAINING_PARAMS['THRESHOLD']
        )
        
        # Validate
        val_loss, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, device, TRAINING_PARAMS['THRESHOLD']
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Update training history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        # Check if this is the best model
        is_best_loss = val_loss < best_val_loss
        is_best_f1 = val_f1 > best_val_f1
        
        if is_best_loss:
            best_val_loss = val_loss
            save_model(
                model, epoch, optimizer, scheduler, train_losses, val_losses, train_f1s, val_f1s,
                os.path.join(args.output_dir, 'best_model_loss.pt')
            )
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        if is_best_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            save_model(
                model, epoch, optimizer, scheduler, train_losses, val_losses, train_f1s, val_f1s,
                os.path.join(args.output_dir, 'best_model_f1.pt')
            )
            logger.info(f"New best validation F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{TRAINING_PARAMS['EPOCHS']} - Time: {epoch_time:.2f}s - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
                   f"LR: {current_lr:.6f}")
        
        # Update CSV file with detailed metrics
        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                   f"{train_precision:.6f},{val_precision:.6f},"
                   f"{train_recall:.6f},{val_recall:.6f},"
                   f"{train_f1:.6f},{val_f1:.6f},{current_lr:.6f}\n")
        
        # Save current model
        save_model(
            model, epoch, optimizer, scheduler, train_losses, val_losses, train_f1s, val_f1s,
            os.path.join(args.output_dir, 'latest_model.pt')
        )
        
        # Plot training progress so far
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plot_training_history(train_losses, val_losses, train_f1s, val_f1s, args.output_dir)
            logger.info(f"Saved training plots at epoch {epoch+1}")
        
        # Check early stopping
        if patience_counter >= TRAINING_PARAMS['PATIENCE']:
            logger.info(f"Early stopping at epoch {epoch+1} as validation F1 did not improve for {TRAINING_PARAMS['PATIENCE']} epochs")
            break
    
    # Plot final training history
    plot_training_history(train_losses, val_losses, train_f1s, val_f1s, args.output_dir)
    
    # Load best model
    logger.info(f"Loading best model from epoch {best_epoch + 1}...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_f1.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test best model
    logger.info(f"Testing best model from epoch {best_epoch + 1}...")
    test_loss, test_precision, test_recall, test_f1, per_class_precision, per_class_recall, per_class_f1, all_probabilities = test(
        model, test_loader, criterion, device, TRAINING_PARAMS['THRESHOLD']
    )
    
    # Print test results
    logger.info(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, "
              f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist()
    }
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # Save per-class metrics with cell type names
    per_class_metrics = []
    for i, cell_type in enumerate(metadata['idx_to_cell_type'].values()):
        per_class_metrics.append({
            'cell_type': cell_type,
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i])
        })
    
    # Sort by F1 score
    per_class_metrics.sort(key=lambda x: x['f1'], reverse=True)
    
    with open(os.path.join(args.output_dir, 'per_class_metrics.json'), 'w') as f:
        json.dump(per_class_metrics, f, indent=4)
    
    # Save class probabilities
    np.save(os.path.join(args.output_dir, 'test_probabilities.npy'), all_probabilities)
    
    # Print total training time
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Best model saved at epoch {best_epoch + 1}")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    logger.info(f"Test F1: {test_f1:.4f}")
    
    # Save final summary
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        f.write(f"Best model saved at epoch {best_epoch + 1}\n")
        f.write(f"Best validation F1: {best_val_f1:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        
        # Add top and bottom 5 cell types by performance
        f.write("\nTop 5 cell types by F1 score:\n")
        for i in range(min(5, len(per_class_metrics))):
            cell = per_class_metrics[i]
            f.write(f"{cell['cell_type']}: F1={cell['f1']:.4f}, Precision={cell['precision']:.4f}, Recall={cell['recall']:.4f}\n")
            
        f.write("\nBottom 5 cell types by F1 score:\n")
        for cell in per_class_metrics[-min(5, len(per_class_metrics)):]:
            f.write(f"{cell['cell_type']}: F1={cell['f1']:.4f}, Precision={cell['precision']:.4f}, Recall={cell['recall']:.4f}\n")
    
    # Plot per-class F1 scores (top 20)
    plt.figure(figsize=(12, 8))
    top_cells = per_class_metrics[:20]
    cell_names = [cell['cell_type'] for cell in top_cells]
    f1_scores = [cell['f1'] for cell in top_cells]
    
    plt.barh(range(len(cell_names)), f1_scores, color='skyblue')
    plt.yticks(range(len(cell_names)), cell_names)
    plt.xlabel('F1 Score')
    plt.title('F1 Scores for Top 20 Cell Types')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top_cell_types_f1.png'))
    plt.close()
    
    logger.info(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()