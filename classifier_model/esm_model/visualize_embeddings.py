"""
Visualization script for protein classifier embeddings.

Creates PCA projections to visualize:
1. Raw ESM embeddings (before classifier)
2. Classifier embeddings (after network, before output layer)

Shows how the network learns to separate functional vs non-functional proteins.
"""

import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

import config
import utils
from esm_embedder import ESMEmbedder
from esm_classifier import ESMClassifier


def load_trained_model(checkpoint_path):
    """Load a trained model checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get embedding dimension
    esm_embed_dim = config.get_esm_embedding_dim()

    # Create model
    model = ESMClassifier(esm_embed_dim=esm_embed_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully!")
    return model


def extract_features_before_output(model, embeddings, masks):
    """
    Extract features from the layer right before the output.

    Returns features after attention pooling + hidden layers, before final classification.
    """
    model.eval()
    features_list = []

    with torch.no_grad():
        for i in range(len(embeddings)):
            emb = embeddings[i:i+1]  # Single sample
            mask = masks[i:i+1]

            # Pooling (attention/mean/max)
            if model.pooling_strategy == "attention":
                pooled = model.pooling(emb, mask)
            elif model.pooling_strategy == "mean":
                if mask is not None:
                    embeddings_masked = emb * mask.unsqueeze(-1).float()
                    seq_lengths = mask.sum(dim=1, keepdim=True).float()
                    pooled = embeddings_masked.sum(dim=1) / seq_lengths
                else:
                    pooled = emb.mean(dim=1)
            elif model.pooling_strategy == "max":
                if mask is not None:
                    embeddings_masked = emb.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                    pooled = embeddings_masked.max(dim=1)[0]
                else:
                    pooled = emb.max(dim=1)[0]

            # Through classifier layers (stop before final output)
            # Get intermediate features from classifier
            # The classifier is a Sequential, so we can access layers by index
            for idx, layer in enumerate(model.classifier):
                pooled = layer(pooled)
                if idx == len(model.classifier) - 2:  # Stop before final layer
                    break

            features_list.append(pooled.squeeze().cpu().numpy())

    return np.array(features_list)


def extract_raw_esm_features(embeddings, masks):
    """
    Extract raw ESM embeddings by mean pooling (simple baseline).
    """
    features_list = []

    for i in range(len(embeddings)):
        emb = embeddings[i]
        mask = masks[i]

        # Mean pooling over sequence length (ignoring padding)
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (emb * mask_expanded).sum(dim=0) / mask_expanded.sum(dim=0)

        features_list.append(pooled.cpu().numpy())

    return np.array(features_list)

""" During Training (train.py)

  1. Compute ESM embeddings → cached to disk (cache/esm_embeddings/{hash}.pt)
  2. Train model, update weights
  3. Save only the model weights to checkpoint (not embeddings)

  During Visualization (visualize_embeddings.py)

  1. Load the same validation sequences
  2. Compute ESM embeddings again → loads from cache (super fast!)
  3. Load trained model weights from checkpoint
  4. Run forward pass to extract features at different layers

  ---
  How We Get "Before" and "After" Embeddings

  "Before" (Raw ESM): Lines 301-302
  features_before_network = extract_raw_esm_features(val_embeddings, val_masks)
  - Takes cached ESM embeddings
  - Does simple mean pooling (doesn't use trained model)
  - Result: What ESM gave us originally

  "After" (Classifier): Line 303
  features_after_network = extract_features_before_output(model.cpu(),
  val_embeddings, val_masks)
  - Takes same ESM embeddings
  - Runs through loaded trained model (with learned weights from checkpoint)
  - Stops before final output layer
  - Result: What the network learned to transform them into


   """
def apply_pca(features, n_components=2):
    """Apply PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    return reduced, pca


def plot_embeddings_2d(embeddings_before, embeddings_after, labels, predictions, save_path=None):
    """
    Create side-by-side 2D PCA plots.

    Left: Before network (raw ESM)
    Right: After network (classifier embeddings)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors: functional=blue, non-functional=red
    colors = ['red' if label == 0 else 'blue' for label in labels]

    # Plot 1: Before network
    axes[0].scatter(embeddings_before[:, 0], embeddings_before[:, 1],
                   c=colors, alpha=0.6, s=50)
    axes[0].set_title('Before Network (Raw ESM Embeddings)', fontsize=14)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: After network
    axes[1].scatter(embeddings_after[:, 0], embeddings_after[:, 1],
                   c=colors, alpha=0.6, s=50)
    axes[1].set_title('After Network (Classifier Embeddings)', fontsize=14)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Functional (Class 1)'),
        Patch(facecolor='red', label='Non-functional (Class 0)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_embeddings_with_predictions(embeddings_after, labels, predictions, save_path=None):
    """
    Plot embeddings colored by prediction correctness.

    Green: Correct predictions
    Orange: Incorrect predictions
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Determine correctness
    correct = (predictions == labels)

    # Plot incorrect first (so they're visible)
    incorrect_mask = ~correct
    if incorrect_mask.any():
        ax.scatter(embeddings_after[incorrect_mask, 0],
                  embeddings_after[incorrect_mask, 1],
                  c='orange', marker='x', s=100, alpha=0.8,
                  label='Incorrect Predictions', linewidths=2)

    # Plot correct
    correct_mask = correct
    colors_correct = ['red' if label == 0 else 'blue' for label in labels[correct_mask]]
    ax.scatter(embeddings_after[correct_mask, 0],
              embeddings_after[correct_mask, 1],
              c=colors_correct, alpha=0.6, s=50,
              label='Correct Predictions')

    ax.set_title('Classifier Embeddings with Prediction Correctness', fontsize=14)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def create_metrics_table(labels, predictions, probabilities):
    """Create a table of accuracy metrics."""
    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Score': [
            accuracy_score(labels, predictions),
            precision_score(labels, predictions),
            recall_score(labels, predictions),
            f1_score(labels, predictions)
        ]
    }

    df = pd.DataFrame(metrics)
    print("\n" + "="*40)
    print("CLASSIFICATION METRICS")
    print("="*40)
    print(df.to_string(index=False))
    print("="*40)

    return df


def plot_metrics_table(df, save_path=None):
    """Plot metrics as a table figure."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table_data = [[metric, f"{score:.4f}"] for metric, score in zip(df['Metric'], df['Score'])]
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Score'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.6, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Model Performance Metrics', fontsize=14, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics table to {save_path}")

    plt.show()


def main():
    """Main visualization pipeline."""
    print("="*60)
    print("PROTEIN CLASSIFIER EMBEDDING VISUALIZATION")
    print("="*60)

    # Configuration
    checkpoint_path = "checkpoints/esm_model_BEST_f1_0.9744.pt"  # Best checkpoint (F1: 0.9744)

    # Check if checkpoint exists
    import os
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        print("Please train a model first or update the checkpoint path.")
        return

    print(f"Using checkpoint: {checkpoint_path}\n")

    # Load data
    print("Loading data...")
    sequences, labels = utils.load_protein_data(config.DATA_CSV_PATH)

    # Use test split
    train_seqs, val_seqs, train_labels, val_labels = utils.create_train_test_split(
        sequences, labels, config.TEST_SIZE, config.RANDOM_STATE
    )

    print(f"Using {len(val_seqs)} validation samples for visualization\n")

    # Compute embeddings
    print("Computing ESM embeddings...")
    embedder = ESMEmbedder()
    val_embeddings, val_masks = embedder.embed_sequences(val_seqs, max_length=config.MAX_SEQUENCE_LENGTH)

    # Load model
    model = load_trained_model(checkpoint_path)
    device = config.get_device()
    model = model.to(device)

    # Get predictions
    print("\nComputing predictions...")
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(len(val_embeddings)):
            emb = val_embeddings[i:i+1].to(device)
            mask = val_masks[i:i+1].to(device)

            output = model(emb, mask)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

            all_preds.append(pred)
            all_probs.append(prob)

    predictions = np.array(all_preds)
    probabilities = np.array(all_probs)
    labels_np = np.array(val_labels)

    # Extract features
    print("\nExtracting features before output layer...")
    features_before_network = extract_raw_esm_features(val_embeddings, val_masks)
    features_after_network = extract_features_before_output(model.cpu(), val_embeddings, val_masks)

    # Apply PCA
    print("\nApplying PCA to raw ESM features...")
    pca_before, _ = apply_pca(features_before_network, n_components=2)

    print("\nApplying PCA to classifier features...")
    pca_after, _ = apply_pca(features_after_network, n_components=2)

    # Create visualizations
    print("\nCreating visualizations...")

    # Metrics table
    df_metrics = create_metrics_table(labels_np, predictions, probabilities)
    plot_metrics_table(df_metrics, save_path='results/metrics_table.png')

    # Side-by-side comparison
    plot_embeddings_2d(pca_before, pca_after, labels_np, predictions,
                      save_path='results/embeddings_comparison.png')

    # Prediction correctness plot
    plot_embeddings_with_predictions(pca_after, labels_np, predictions,
                                    save_path='results/embeddings_predictions.png')

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("Figures saved to results/ directory:")
    print("  - metrics_table.png")
    print("  - embeddings_comparison.png")
    print("  - embeddings_predictions.png")


if __name__ == "__main__":
    main()
