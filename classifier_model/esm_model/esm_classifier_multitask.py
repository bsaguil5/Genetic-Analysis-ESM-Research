"""
Multi-Task ESM Protein Classifier

Extends the binary classifier to predict multiple related tasks simultaneously:
1. Binary classification (transporter vs non-transporter) - PRIMARY TASK
2. Subfamily prediction (ABC, SWEET, NRT, MATE, etc.) - AUXILIARY
3. Substrate type (sugar, nitrogen, metal, etc.) - AUXILIARY
4. TM domain count (regression) - AUXILIARY

Multi-task learning improves generalization by:
- Forcing the model to learn more robust features
- Reducing overfitting through shared representations
- Improving performance on rare classes (nitrate, ammonium)

Author: Brandon & Claude
Date: December 24, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import config
from esm_classifier import AttentionPooling, FocalLoss


class ESMClassifier_MultiTask(nn.Module):
    """
    Multi-task protein classifier with shared encoder and multiple prediction heads.

    Architecture:
        ESM Embeddings (batch, seq_len, 480)
            ↓
        [Optional Projection]
            ↓
        Attention Pooling (batch, 480)
            ↓
        Shared Encoder (batch, hidden_dim)
            ↓
        ┌─────────┬──────────┬──────────┬──────────┐
        │ Binary  │Subfamily │Substrate │TM Count  │
        │  Head   │   Head   │   Head   │   Head   │
        └─────────┴──────────┴──────────┴──────────┘
    """

    def __init__(self,
                 esm_embed_dim: int,
                 hidden_dim: int = None,
                 pooling_strategy: str = None,
                 dropout: float = None,
                 num_subfamilies: int = 8,
                 num_substrates: int = 6):
        """
        Args:
            esm_embed_dim: Dimension of ESM embeddings (480 for esm2_t12_35M_UR50D)
            hidden_dim: Hidden dimension for classification heads
            pooling_strategy: "attention", "mean", or "max"
            dropout: Dropout rate for regularization
            num_subfamilies: Number of subfamily classes
            num_substrates: Number of substrate classes
        """
        super().__init__()

        # Use config defaults if not specified
        hidden_dim = hidden_dim or config.CLASSIFIER_HIDDEN_DIM
        pooling_strategy = pooling_strategy or config.POOLING_STRATEGY
        dropout = dropout or config.CLASSIFIER_DROPOUT

        self.esm_embed_dim = esm_embed_dim
        self.hidden_dim = hidden_dim
        self.pooling_strategy = pooling_strategy
        self.num_subfamilies = num_subfamilies
        self.num_substrates = num_substrates

        # Optional projection layer (if ESM dim is very large)
        self.use_projection = esm_embed_dim > 512
        if self.use_projection:
            self.projection = nn.Linear(esm_embed_dim, hidden_dim)
            self.proj_dropout = nn.Dropout(dropout * 0.5)
            classifier_input_dim = hidden_dim
        else:
            self.projection = None
            classifier_input_dim = esm_embed_dim

        # Pooling layer (shared across all tasks)
        if pooling_strategy == "attention":
            self.pooling = AttentionPooling(classifier_input_dim)
        elif pooling_strategy in ["mean", "max"]:
            self.pooling = None  # Use functional pooling
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        # Shared encoder (learns general protein features)
        # All tasks benefit from this shared representation
        if isinstance(hidden_dim, int):
            encoder_dims = [hidden_dim, hidden_dim // 2]
        elif isinstance(hidden_dim, list):
            encoder_dims = hidden_dim
        else:
            raise ValueError(f"hidden_dim must be int or list, got {type(hidden_dim)}")

        encoder_layers = []
        current_dim = classifier_input_dim

        for i, next_dim in enumerate(encoder_dims):
            encoder_layers.append(nn.Linear(current_dim, next_dim))
            encoder_layers.append(nn.ReLU())
            dropout_rate = dropout if i == 0 else dropout * 0.5
            encoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = next_dim

        self.shared_encoder = nn.Sequential(*encoder_layers)
        shared_output_dim = current_dim

        # Task-specific prediction heads
        # Each head is a small MLP that specializes in one task

        # 1. Binary classification head (transporter vs non-transporter)
        self.binary_head = nn.Sequential(
            nn.Linear(shared_output_dim, shared_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(shared_output_dim // 2, 1)  # Binary output
        )

        # 2. Subfamily classification head (ABC, SWEET, NRT, etc.)
        self.subfamily_head = nn.Sequential(
            nn.Linear(shared_output_dim, shared_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(shared_output_dim // 2, num_subfamilies)  # Multi-class output
        )

        # 3. Substrate classification head (sugar, nitrogen, metal, etc.)
        self.substrate_head = nn.Sequential(
            nn.Linear(shared_output_dim, shared_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(shared_output_dim // 2, num_substrates)  # Multi-class output
        )

        # 4. TM domain count regression head
        self.tm_count_head = nn.Sequential(
            nn.Linear(shared_output_dim, shared_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(shared_output_dim // 4, 1),  # Regression output
            nn.ReLU()  # Ensure non-negative TM counts
        )

        # Initialize weights
        self._init_weights()

        # Print architecture summary
        print(f"\n   Multi-Task ESM Classifier initialized:")
        print(f"   ESM embed dim: {esm_embed_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Pooling: {pooling_strategy}")
        print(f"   Dropout: {dropout}")
        print(f"   Use projection: {self.use_projection}")
        print(f"\n   Task outputs:")
        print(f"   - Binary classification: 1 output")
        print(f"   - Subfamily classification: {num_subfamilies} classes")
        print(f"   - Substrate classification: {num_substrates} classes")
        print(f"   - TM domain regression: 1 output")
        print(f"\n   Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                embeddings: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task model.

        Args:
            embeddings: ESM embeddings (batch_size, seq_len, embed_dim)
            mask: Attention mask (batch_size, seq_len) - True for real tokens

        Returns:
            Dictionary with predictions for each task:
            {
                'binary': logits (batch_size, 1),
                'subfamily': logits (batch_size, num_subfamilies),
                'substrate': logits (batch_size, num_substrates),
                'tm_count': predictions (batch_size, 1)
            }
        """
        batch_size, seq_len, embed_dim = embeddings.shape

        # Optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)
            embeddings = self.proj_dropout(embeddings)

        # Sequence pooling (same as single-task model)
        if self.pooling_strategy == "attention":
            pooled = self.pooling(embeddings, mask)
        elif self.pooling_strategy == "mean":
            if mask is not None:
                embeddings_masked = embeddings * mask.unsqueeze(-1).float()
                seq_lengths = mask.sum(dim=1, keepdim=True).float()
                pooled = embeddings_masked.sum(dim=1) / seq_lengths
            else:
                pooled = embeddings.mean(dim=1)
        elif self.pooling_strategy == "max":
            if mask is not None:
                embeddings_masked = embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                pooled = embeddings_masked.max(dim=1)[0]
            else:
                pooled = embeddings.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Shared encoder (all tasks use this)
        shared_features = self.shared_encoder(pooled)

        # Task-specific predictions
        predictions = {
            'binary': self.binary_head(shared_features),
            'subfamily': self.subfamily_head(shared_features),
            'substrate': self.substrate_head(shared_features),
            'tm_count': self.tm_count_head(shared_features)
        }

        return predictions


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning with task weighting.

    Loss components:
    1. Binary classification: Focal loss (handles class imbalance)
    2. Subfamily classification: Cross-entropy
    3. Substrate classification: Cross-entropy
    4. TM domain regression: MSE

    Task weights balance the importance and scale of each loss.
    """

    def __init__(self,
                 binary_weight: float = 1.0,
                 subfamily_weight: float = 0.5,
                 substrate_weight: float = 0.3,
                 tm_weight: float = 0.2,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        Args:
            binary_weight: Weight for binary classification (primary task)
            subfamily_weight: Weight for subfamily classification
            substrate_weight: Weight for substrate classification
            tm_weight: Weight for TM domain regression
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()

        self.binary_weight = binary_weight
        self.subfamily_weight = subfamily_weight
        self.substrate_weight = substrate_weight
        self.tm_weight = tm_weight

        # Loss functions for each task
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            predictions: Dictionary of model predictions for each task
            targets: Dictionary of ground truth labels for each task

        Returns:
            total_loss: Weighted sum of all task losses
            loss_dict: Individual losses for logging
        """
        # 1. Binary classification loss (Focal loss for class imbalance)
        binary_loss = self.focal_loss(
            predictions['binary'].squeeze(),
            targets['binary'].float()
        )

        # 2. Subfamily classification loss (Cross-entropy)
        subfamily_loss = self.ce_loss(
            predictions['subfamily'],
            targets['subfamily']
        )

        # 3. Substrate classification loss (Cross-entropy)
        substrate_loss = self.ce_loss(
            predictions['substrate'],
            targets['substrate']
        )

        # 4. TM domain count regression loss (MSE)
        tm_loss = self.mse_loss(
            predictions['tm_count'].squeeze(),
            targets['tm_count'].float()
        )

        # Weighted combination
        total_loss = (
            self.binary_weight * binary_loss +
            self.subfamily_weight * subfamily_loss +
            self.substrate_weight * substrate_loss +
            self.tm_weight * tm_loss
        )

        # Return individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'binary': binary_loss.item(),
            'subfamily': subfamily_loss.item(),
            'substrate': substrate_loss.item(),
            'tm_count': tm_loss.item()
        }

        return total_loss, loss_dict


def create_multitask_classifier(esm_embed_dim: int = None,
                                 num_subfamilies: int = 8,
                                 num_substrates: int = 6) -> ESMClassifier_MultiTask:
    """
    Create multi-task ESM classifier with default configuration.

    Args:
        esm_embed_dim: ESM embedding dimension (auto-detected if None)
        num_subfamilies: Number of subfamily classes
        num_substrates: Number of substrate classes

    Returns:
        model: ESMClassifier_MultiTask instance
    """
    if esm_embed_dim is None:
        esm_embed_dim = config.get_esm_embedding_dim()

    return ESMClassifier_MultiTask(
        esm_embed_dim=esm_embed_dim,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        pooling_strategy=config.POOLING_STRATEGY,
        dropout=config.CLASSIFIER_DROPOUT,
        num_subfamilies=num_subfamilies,
        num_substrates=num_substrates
    )
