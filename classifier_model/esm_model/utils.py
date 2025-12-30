"""
Utility functions for ESM-based protein classifier.

This module contains helper functions for data loading, preprocessing,
evaluation metrics, and other common tasks.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import List, Tuple, Dict, Any
from datetime import datetime
import config


def set_random_seeds(seed: int = None):
    """Set random seeds for reproducibility."""
    seed = seed or config.SEED
    
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_protein_data(csv_path: str = None, require_labels: bool = True) -> Tuple[List[str], List[int]]:
    """
    Load protein sequences and labels from CSV file.
    
    Args:
        csv_path: Path to CSV file
        require_labels: Whether to require a label column (default: True)
        
    Returns:
        sequences: List of protein sequences
        labels: List of binary labels (0/1) or None if not found/required
    """
    csv_path = csv_path or config.DATA_CSV_PATH
    
    print(f"[*] Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows")
    
    # Auto-detect sequence column
    sequence_col = None
    for col in config.SEQUENCE_COLUMNS:
        if col in df.columns:
            sequence_col = col
            break
    
    if sequence_col is None:
        raise ValueError(f"No sequence column found. Available columns: {list(df.columns)}")
    
    # Auto-detect label column
    label_col = None
    for col in config.LABEL_COLUMNS:
        if col in df.columns:
            label_col = col
            break
    
    if require_labels and label_col is None:
        raise ValueError(f"No label column found. Available columns: {list(df.columns)}")
    
    print(f"   Sequence column: '{sequence_col}'")
    if label_col:
        print(f"   Label column: '{label_col}'")
    else:
        print(f"   Label column: None (Unlabeled data)")
    
    # Extract sequences and labels
    sequences = df[sequence_col].tolist()
    labels = df[label_col].tolist() if label_col else None
    
    # Clean data
    clean_data = []
    
    if labels:
        for seq, label in zip(sequences, labels):
            # Skip if missing data
            if pd.isna(seq) or pd.isna(label):
                continue
            
            # Clean sequence
            seq = str(seq).strip().upper()
            if len(seq) == 0:
                continue
            
            # Convert label to int
            try:
                label = int(float(label))
                if label not in [0, 1]:
                    continue
            except (ValueError, TypeError):
                continue
            
            clean_data.append((seq, label))
            
        sequences, labels = zip(*clean_data)
        sequences = list(sequences)
        labels = list(labels)
    else:
        # Just clean sequences
        for seq in sequences:
            if pd.isna(seq):
                continue
            seq = str(seq).strip().upper()
            if len(seq) > 0:
                clean_data.append(seq)
        sequences = clean_data
        labels = None
        
    print(f"   Clean data: {len(sequences)} sequences")
    
    # Print label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"   Label distribution: {label_dist}")
    
    # Print sequence length statistics
    seq_lengths = [len(seq) for seq in sequences]
    print(f"   Sequence lengths:")
    print(f"     Mean: {np.mean(seq_lengths):.1f}")
    print(f"     Median: {np.median(seq_lengths):.1f}")
    print(f"     Min: {np.min(seq_lengths)}")
    print(f"     Max: {np.max(seq_lengths)}")
    print(f"     95th percentile: {np.percentile(seq_lengths, 95):.1f}")
    
    return sequences, labels


def create_train_test_split(sequences: List[str],
                           labels: List[int],
                           test_size: float = None,
                           random_state: int = None) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Split data into train and test sets.

    DEPRECATED: Use create_train_val_test_split instead for proper 3-way split.

    Args:
        sequences: List of protein sequences
        labels: List of binary labels
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        train_sequences, test_sequences, train_labels, test_labels
    """
    test_size = test_size or config.TEST_SIZE
    random_state = random_state or config.RANDOM_STATE

    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Maintain class distribution
    )

    print(f"[*] Data split:")
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Test: {len(test_sequences)} sequences")
    print(f"   Train labels: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"   Test labels: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    return train_sequences, test_sequences, train_labels, test_labels


def create_train_val_test_split(sequences: List[str],
                                labels: List[int],
                                test_size: float = None,
                                val_size: float = None,
                                random_state: int = None) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Split data into train, validation, and test sets.

    First splits into train+val (1-test_size) vs test (test_size).
    Then splits train+val into train (1-val_size) vs val (val_size).

    Args:
        sequences: List of protein sequences
        labels: List of binary labels
        test_size: Fraction for test set (default from config)
        val_size: Fraction of remaining data for validation (default from config)
        random_state: Random seed (default from config)

    Returns:
        train_sequences, val_sequences, test_sequences, train_labels, val_labels, test_labels
    """
    test_size = test_size or config.TEST_SIZE
    val_size = val_size or config.VAL_SIZE
    random_state = random_state or config.RANDOM_STATE

    # First split: separate test set
    train_val_sequences, test_sequences, train_val_labels, test_labels = train_test_split(
        sequences, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Second split: separate validation from training
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_val_sequences, train_val_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_labels
    )

    print(f"📊 Data split (3-way):")
    print(f"   Train: {len(train_sequences)} sequences ({len(train_sequences)/len(sequences)*100:.1f}%)")
    print(f"   Val:   {len(val_sequences)} sequences ({len(val_sequences)/len(sequences)*100:.1f}%)")
    print(f"   Test:  {len(test_sequences)} sequences ({len(test_sequences)/len(sequences)*100:.1f}%)")
    print(f"   Train labels: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"   Val labels:   {dict(zip(*np.unique(val_labels, return_counts=True)))}")
    print(f"   Test labels:  {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    return train_sequences, val_sequences, test_sequences, train_labels, val_labels, test_labels


# ====================================================================
# DATA AUGMENTATION
# ====================================================================

# Conservative amino acid substitution groups (biochemically similar)
CONSERVATIVE_SUBSTITUTIONS = {
    'A': ['G', 'S'],  # Small, nonpolar
    'V': ['I', 'L', 'M'],  # Hydrophobic
    'I': ['V', 'L', 'M'],
    'L': ['I', 'V', 'M'],
    'M': ['I', 'V', 'L'],
    'F': ['Y', 'W'],  # Aromatic
    'Y': ['F', 'W'],
    'W': ['F', 'Y'],
    'K': ['R'],  # Positively charged
    'R': ['K'],
    'D': ['E'],  # Negatively charged
    'E': ['D'],
    'S': ['T', 'A'],  # Polar, uncharged
    'T': ['S'],
    'N': ['Q'],
    'Q': ['N'],
    'C': ['S'],  # Cysteine similar to serine
    'G': ['A'],  # Glycine
    'P': ['P'],  # Proline (rigid, no substitutions)
    'H': ['H'],  # Histidine (unique, no good substitutions)
}


def augment_protein_sequence(sequence: str,
                            mutation_rate: float = None,
                            random_state: int = None) -> str:
    """
    Apply conservative mutations to a protein sequence for data augmentation.

    Args:
        sequence: Protein sequence string
        mutation_rate: Probability of mutating each position (default from config)
        random_state: Random seed for reproducibility

    Returns:
        augmented_sequence: Mutated sequence
    """
    mutation_rate = mutation_rate or config.AUGMENTATION_MUTATION_RATE

    if random_state is not None:
        np.random.seed(random_state)

    sequence_list = list(sequence)

    for i in range(len(sequence_list)):
        if np.random.random() < mutation_rate:
            aa = sequence_list[i]
            if aa in CONSERVATIVE_SUBSTITUTIONS:
                possible_subs = CONSERVATIVE_SUBSTITUTIONS[aa]
                if possible_subs and possible_subs != [aa]:
                    sequence_list[i] = np.random.choice(possible_subs)

    return ''.join(sequence_list)


def augment_sequences(sequences: List[str],
                     labels: List[int],
                     augmentation_prob: float = None) -> Tuple[List[str], List[int]]:
    """
    Augment a list of sequences with conservative mutations.

    Args:
        sequences: List of protein sequences
        labels: List of labels
        augmentation_prob: Probability of augmenting each sequence

    Returns:
        augmented_sequences: Original + augmented sequences
        augmented_labels: Original + augmented labels
    """
    augmentation_prob = augmentation_prob or config.AUGMENTATION_PROBABILITY

    augmented_sequences = list(sequences)
    augmented_labels = list(labels)

    for seq, label in zip(sequences, labels):
        if np.random.random() < augmentation_prob:
            aug_seq = augment_protein_sequence(seq)
            augmented_sequences.append(aug_seq)
            augmented_labels.append(label)

    print(f"🔄 Data augmentation:")
    print(f"   Original: {len(sequences)} sequences")
    print(f"   Augmented: {len(augmented_sequences)} sequences (+{len(augmented_sequences) - len(sequences)})")

    return augmented_sequences, augmented_labels


class CachedEmbeddingDataset(Dataset):
    """
    Dataset that loads embeddings on-the-fly from cache.

    Useful for very large datasets where loading all embeddings
    into memory at once is not feasible.
    """

    def __init__(self, sequences: List[str], labels: List[int], embedder):
        """
        Initialize the dataset.

        Args:
            sequences: List of protein sequences
            labels: List of labels
            embedder: ESMEmbedder instance for loading cached embeddings
        """
        self.sequences = sequences
        self.labels = labels
        self.embedder = embedder

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Load embedding from cache for a single sequence."""
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Try to load from cache
        cached = self.embedder._load_from_cache(sequence)

        if cached is not None:
            embedding, mask = cached
        else:
            # If not cached, compute (this shouldn't happen if embeddings were pre-computed)
            embedding, mask = self.embedder.embed_sequences(
                [sequence], max_length=config.MAX_SEQUENCE_LENGTH
            )
            embedding = embedding.squeeze(0)  # Remove batch dimension
            mask = mask.squeeze(0)

        return embedding, mask, torch.tensor(label, dtype=torch.float32)


def create_data_loader(embeddings: torch.Tensor,
                      masks: torch.Tensor,
                      labels: torch.Tensor,
                      batch_size: int = None,
                      shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader from embeddings, masks, and labels.

    Args:
        embeddings: ESM embeddings tensor
        masks: Attention masks tensor
        labels: Labels tensor
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        dataloader: PyTorch DataLoader
    """
    batch_size = batch_size or config.BATCH_SIZE

    dataset = TensorDataset(embeddings, masks, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()
    )

    return dataloader


def create_cached_data_loader(sequences: List[str],
                              labels: List[int],
                              embedder,
                              batch_size: int = None,
                              shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader that loads embeddings from cache on-the-fly.

    This is useful for very large datasets where loading all embeddings
    into memory at once is not feasible.

    Args:
        sequences: List of protein sequences
        labels: List of labels
        embedder: ESMEmbedder instance
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        dataloader: PyTorch DataLoader
    """
    batch_size = batch_size or config.BATCH_SIZE

    dataset = CachedEmbeddingDataset(sequences, labels, embedder)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()
    )

    return dataloader


def compute_metrics(y_true: np.ndarray, 
                   y_pred: np.ndarray, 
                   y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        metrics: Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.0  # If only one class present
    
    return metrics


def print_evaluation_report(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           y_proba: np.ndarray = None,
                           title: str = "Evaluation Results"):
    """
    Print a comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        title: Title for the report
    """
    print(f"\n{'='*60}")
    print(f"🎯 {title.upper()}")
    print('='*60)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    # Print main metrics
    print(f"📊 Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"📊 Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"📊 Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"📊 F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    if 'auc' in metrics:
        print(f"📊 AUC:       {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
    
    print('='*60)
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    # Confusion matrix
    print("\n🔍 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"               Predicted")
    print(f"              Neg   Pos")
    print(f"Actual Neg   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Pos   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Probability analysis if available
    if y_proba is not None:
        pos_probs = y_proba[y_true == 1]
        neg_probs = y_proba[y_true == 0]
        
        print("\n📈 Probability Distribution Analysis:")
        if len(pos_probs) > 0:
            print(f"Positive samples - Mean: {pos_probs.mean():.3f}, Std: {pos_probs.std():.3f}")
        if len(neg_probs) > 0:
            print(f"Negative samples - Mean: {neg_probs.mean():.3f}, Std: {neg_probs.std():.3f}")
    
    return metrics


def save_results(results: Dict[str, Any], 
                filename: str = None, 
                results_dir: str = None) -> str:
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results to save
        filename: Output filename (auto-generated if None)
        results_dir: Output directory
        
    Returns:
        filepath: Path to saved file
    """
    results_dir = results_dir or config.RESULTS_DIR
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"esm_results_{timestamp}.json"
    
    filepath = os.path.join(results_dir, filename)
    
    # Add timestamp to results
    results['timestamp'] = datetime.now().isoformat()
    results['config'] = {
        'esm_model': config.ESM_MODEL_NAME,
        'pooling_strategy': config.POOLING_STRATEGY,
        'loss_type': config.LOSS_TYPE,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {filepath}")
    return filepath


def save_model_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         metrics: Dict[str, float],
                         filename: str = None,
                         checkpoint_dir: str = None) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Performance metrics
        filename: Checkpoint filename (auto-generated if None)
        checkpoint_dir: Checkpoint directory
        
    Returns:
        filepath: Path to saved checkpoint
    """
    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    
    if filename is None:
        f1_score = metrics.get('f1_score', 0.0)
        filename = f"esm_model_epoch_{epoch}_f1_{f1_score:.4f}.pt"
    
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'esm_model': config.ESM_MODEL_NAME,
            'esm_embed_dim': config.get_esm_embedding_dim(),
            'pooling_strategy': config.POOLING_STRATEGY,
            'hidden_dim': config.CLASSIFIER_HIDDEN_DIM,
            'dropout': config.CLASSIFIER_DROPOUT,
        }
    }
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"[*] Checkpoint saved: {filepath}")
    
    return filepath


def load_model_checkpoint(filepath: str, 
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        checkpoint: Checkpoint data
    """
    print(f"[*] Loading checkpoint: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=config.get_device(), weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"[*] Checkpoint loaded successfully!")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


def cleanup_old_checkpoints(checkpoint_dir: str = None, keep_best_n: int = None):
    """
    Clean up old checkpoints, keeping only the best N models.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best_n: Number of best models to keep
    """
    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    keep_best_n = keep_best_n or config.KEEP_BEST_N
    
    if keep_best_n <= 0:
        return
    
    # Find all checkpoint files
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt') and 'f1_' in filename:
            filepath = os.path.join(checkpoint_dir, filename)
            
            # Extract F1 score from filename
            try:
                f1_str = filename.split('f1_')[1].split('.pt')[0]
                f1_score = float(f1_str)
                checkpoint_files.append((filepath, f1_score))
            except (IndexError, ValueError):
                continue
    
    # Sort by F1 score (descending)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # Keep only the best N
    if len(checkpoint_files) > keep_best_n:
        files_to_remove = checkpoint_files[keep_best_n:]
        
        for filepath, f1_score in files_to_remove:
            try:
                os.remove(filepath)
                print(f"[*] Removed old checkpoint: {os.path.basename(filepath)} (F1: {f1_score:.4f})")
            except OSError:
                pass


def compare_with_benchmarks(f1_score: float):
    """
    Compare F1 score with known benchmarks.
    
    Args:
        f1_score: F1 score to compare
    """
    print("\n[*] BENCHMARK COMPARISON:")
    print("-" * 50)
    
    benchmarks = {
        'Random Baseline': 0.500,
        'Simple Feedforward': 0.700,
        'BiGRU-Mamba': 0.850,
        'Transformer (ESM2-inspired)': 0.935,
        'ESM Classifier': f1_score
    }
    
    for model_name, score in benchmarks.items():
        indicator = "[*]" if model_name == 'ESM Classifier' else "  "
        print(f"{indicator} {model_name:<25}: {score:.3f}")
    
    print("-" * 50)
    
    if f1_score >= 0.93:
        print("[*] EXCELLENT! Matches/exceeds transformer performance!")
    elif f1_score >= 0.85:
        print("[*] VERY GOOD! Strong competitive performance!")
    elif f1_score >= 0.70:
        print("[*] GOOD! Above baseline performance!")
    else:
        print("[!] Needs improvement to reach competitive levels")


# Convenience function to get the current timestamp
def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")