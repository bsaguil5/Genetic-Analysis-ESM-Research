"""
Configuration settings for ESM-based protein classifier.

This file contains all the hyperparameters and settings for training and testing.
Modify these values to experiment with different configurations.
"""

import os
from pathlib import Path

# ====================================================================
# DATA CONFIGURATION
# ====================================================================

# Path to the labeled dataset CSV file
# Compute absolute path relative to this config file's location
DATA_CSV_PATH = str(Path(__file__).parent.parent.parent / "labeled_sequences.csv")

# Expected column names in the CSV (will auto-detect)
SEQUENCE_COLUMNS = ["Sequence", " Sequence", "sequence"]
LABEL_COLUMNS = ["Label", " Label", "label"]

# Train/val/test split ratios
# First split: train+val (80%) vs test (20%)
# Second split: train (80% of train+val = 64% total) vs val (20% of train+val = 16% total)
TEST_SIZE = 0.2  # Test set size (held out, not used during training)
VAL_SIZE = 0.2   # Validation set size (from remaining train+val data)
RANDOM_STATE = 42  # For reproducible splits

# ====================================================================
# ESM MODEL CONFIGURATION
# ====================================================================

# ESM model to use from Hugging Face
# Options: 
#   - facebook/esm2_t6_8M_UR50D (8M params, 320 dim, fastest)
#   - facebook/esm2_t12_35M_UR50D (35M params, 480 dim, good balance)
#   - facebook/esm2_t30_150M_UR50D (150M params, 640 dim, better quality)
#   - facebook/esm2_t33_650M_UR50D (650M params, 1280 dim, best quality)
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

# Maximum sequence length for processing
MAX_SEQUENCE_LENGTH = 512

# Batch size for ESM embedding computation
# Reduce if you get CUDA OOM errors
# Batch size for ESM embedding computation
# Reduce if you get CUDA OOM errors
ESM_BATCH_SIZE = 8  # Optimized for Colab T4 GPU

# Cache directory for storing computed embeddings
# DISABLED for testing to avoid corrupted cache files
ESM_CACHE_DIR = None  # Disabled - compute embeddings fresh

# ====================================================================
# CLASSIFIER ARCHITECTURE
# ====================================================================

# Pooling strategy for sequence representations
# Options: "mean", "max", "attention"
POOLING_STRATEGY = "attention"

# Attention pooling configuration
ATTENTION_HIDDEN_DIM = 128  # Hidden dimension for attention mechanism (only used if POOLING_STRATEGY="attention")

# Classification head architecture
# Can be a single int (will create 3-layer network with //2 reduction) or list of ints for custom architecture
# Examples:
#   256 -> Creates: 256 -> 128 -> 1
#   [512, 256, 128] -> Creates: 512 -> 256 -> 128 -> 1
CLASSIFIER_HIDDEN_DIM = 256
CLASSIFIER_DROPOUT = 0.3

# ====================================================================
# TRAINING CONFIGURATION
# ====================================================================

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50

# Data augmentation
USE_DATA_AUGMENTATION = False  # Enable data augmentation during training
AUGMENTATION_PROBABILITY = 0.3  # Probability of applying augmentation to a sequence
AUGMENTATION_MUTATION_RATE = 0.02  # Rate of conservative amino acid substitutions

# Early stopping
PATIENCE = 10  # Stop if no improvement for this many epochs
MIN_DELTA = 0.001  # Minimum improvement to count as progress

# Learning rate scheduling
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# Gradient clipping
MAX_GRAD_NORM = 1.0

# ====================================================================
# LOSS FUNCTION CONFIGURATION
# ====================================================================

# Loss function type
# Options: "bce", "focal", "weighted_bce"
LOSS_TYPE = "focal"

# Focal loss parameters (if using focal loss)
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ====================================================================
# OUTPUT CONFIGURATION
# ====================================================================

# Directory to save model checkpoints
CHECKPOINT_DIR = "checkpoints"

# Directory to save training logs and results
RESULTS_DIR = "results"

# Save model every N epochs (0 to disable)
SAVE_EVERY_N_EPOCHS = 0

# Keep only the best N checkpoints (0 to keep all)
KEEP_BEST_N = 0

# ====================================================================
# PREDICTION CONFIGURATION
# ====================================================================

# Confidence threshold for predictions
PREDICTION_THRESHOLD = 0.5  # Probability threshold for binary classification (0.0 to 1.0)
# Values > 0.5 increase precision (fewer false positives)
# Values < 0.5 increase recall (fewer false negatives)

# ====================================================================
# CACHE CONFIGURATION
# ====================================================================

# Cache management
MAX_CACHE_SIZE_GB = 100.0  # Increased to accommodate full dataset (~75GB needed for 1124 sequences)
CACHE_CLEANUP_ON_START = False  # Clean cache when starting training

# ====================================================================
# DEVICE CONFIGURATION
# ====================================================================

# Device to use for training/inference
# Options: "auto", "cuda", "cpu"
DEVICE = "auto"  # Auto-detect: Will use CUDA if available

# Multi-GPU configuration
USE_MULTI_GPU = False  # Enable DataParallel for multi-GPU training
GPU_IDS = None  # List of GPU IDs to use (None = use all available)

# Mixed precision training (speeds up training on modern GPUs)
USE_MIXED_PRECISION = True  # Enabled for GPU acceleration

# ====================================================================
# LOGGING CONFIGURATION
# ====================================================================

# Logging frequency
LOG_EVERY_N_STEPS = 10
EVALUATE_EVERY_N_EPOCHS = 1

# Verbose output
VERBOSE = True

# Save detailed training logs
SAVE_TRAINING_LOG = True

# ====================================================================
# REPRODUCIBILITY
# ====================================================================

# Random seeds for reproducibility
SEED = 42

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def get_esm_embedding_dim():
    """Get the embedding dimension for the configured ESM model."""
    esm_dims = {
        "facebook/esm2_t6_8M_UR50D": 320,
        "facebook/esm2_t12_35M_UR50D": 480,
        "facebook/esm2_t30_150M_UR50D": 640,
        "facebook/esm2_t33_650M_UR50D": 1280,
    }
    return esm_dims.get(ESM_MODEL_NAME, 480)  # Default to 35M model

def get_device():
    """Get the device to use for training/inference."""
    if DEVICE == "auto":
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch
        return torch.device(DEVICE)

def create_output_dirs():
    """Create output directories if they don't exist."""
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if ESM_CACHE_DIR:
        os.makedirs(ESM_CACHE_DIR, exist_ok=True)

# Auto-create directories when config is imported
create_output_dirs()