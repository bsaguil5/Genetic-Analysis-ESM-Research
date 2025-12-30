"""
K-Fold Cross-Validation Training for ESM-based protein classifier.

This script trains the ESM classifier using k-fold cross-validation
to get more robust performance estimates.

Usage:
    python train_kfold.py --n_splits 5

The script will:
1. Load data from the configured CSV file
2. Compute ESM embeddings (cached for efficiency)
3. Split data into k folds
4. Train k separate models (one per fold)
5. Report average performance across all folds
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import argparse
import json

# Import our modules
import config
import utils
from esm_embedder import ESMEmbedder
from esm_classifier import ESMClassifier, get_loss_function


class KFoldESMTrainer:
    """
    K-Fold Cross-Validation Trainer for ESM-based protein classifier.

    Trains multiple models using stratified k-fold cross-validation
    to provide robust performance estimates.
    """

    def __init__(self, n_splits: int = 5):
        """Initialize the k-fold trainer."""
        # Set random seeds for reproducibility
        utils.set_random_seeds(config.SEED)

        # Get device
        self.device = config.get_device()
        print(f"🖥️ Using device: {self.device}")

        # K-fold parameters
        self.n_splits = n_splits

        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.USE_MIXED_PRECISION else None

        # Results storage
        self.fold_results = []

    def prepare_data(self):
        """Load and prepare data."""
        print("\n" + "="*60)
        print("📊 PREPARING DATA")
        print("="*60)

        # Load data
        sequences, labels = utils.load_protein_data(config.DATA_CSV_PATH)

        # Store for k-fold splitting
        self.sequences = sequences
        self.labels = np.array(labels)

        print(f"✅ Data loaded successfully!")
        print(f"   Total sequences: {len(self.sequences)}")
        print(f"   K-fold splits: {self.n_splits}")

    def compute_embeddings(self):
        """Compute ESM embeddings for all sequences."""
        print(f"\n🧬 Computing ESM embeddings for all sequences...")
        embedder = ESMEmbedder()

        self.embeddings, self.masks = embedder.embed_sequences(
            self.sequences, max_length=config.MAX_SEQUENCE_LENGTH
        )

        self.esm_embed_dim = self.embeddings.shape[-1]
        print(f"✅ Embeddings computed!")

    def train_fold(self, fold: int, train_idx: np.ndarray, val_idx: np.ndarray):
        """Train a single fold."""
        print(f"\n{'='*60}")
        print(f"📂 FOLD {fold + 1}/{self.n_splits}")
        print(f"{'='*60}")

        # Split data
        train_embeddings = self.embeddings[train_idx]
        train_masks = self.masks[train_idx]
        train_labels = torch.tensor(self.labels[train_idx], dtype=torch.float32)

        val_embeddings = self.embeddings[val_idx]
        val_masks = self.masks[val_idx]
        val_labels = torch.tensor(self.labels[val_idx], dtype=torch.float32)

        print(f"   Train: {len(train_idx)} sequences")
        print(f"   Val:   {len(val_idx)} sequences")

        # Create data loaders
        train_loader = utils.create_data_loader(
            train_embeddings, train_masks, train_labels,
            batch_size=config.BATCH_SIZE, shuffle=True
        )

        val_loader = utils.create_data_loader(
            val_embeddings, val_masks, val_labels,
            batch_size=config.BATCH_SIZE, shuffle=False
        )

        # Create model
        model = ESMClassifier(esm_embed_dim=self.esm_embed_dim)
        model = model.to(self.device)

        # Multi-GPU support
        if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
            gpu_ids = config.GPU_IDS if config.GPU_IDS else list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=gpu_ids)

        # Create loss, optimizer, scheduler
        criterion = get_loss_function(config.LOSS_TYPE)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        if config.USE_SCHEDULER:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE
            )
        else:
            scheduler = None

        # Training loop
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(config.NUM_EPOCHS):
            # Train epoch
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

            for embeddings_batch, masks_batch, labels_batch in pbar:
                # Move to device
                embeddings_batch = embeddings_batch.to(self.device)
                masks_batch = masks_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                if self.scaler is not None:
                    with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        outputs = model(embeddings_batch, masks_batch)
                        loss = criterion(outputs.squeeze(), labels_batch)

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(embeddings_batch, masks_batch)
                    loss = criterion(outputs.squeeze(), labels_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    optimizer.step()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / num_batches

            # Validate
            if (epoch + 1) % config.EVALUATE_EVERY_N_EPOCHS == 0:
                metrics = self.validate_fold(model, val_loader)
                f1_score = metrics['f1_score']

                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, F1={f1_score:.4f}")

                # Learning rate scheduler
                if scheduler is not None:
                    scheduler.step(f1_score)

                # Early stopping
                if f1_score > best_f1:
                    best_f1 = f1_score
                    patience_counter = 0
                    best_metrics = metrics
                else:
                    patience_counter += 1

                if patience_counter >= config.PATIENCE:
                    print(f"  Early stopping triggered")
                    break

        print(f"✅ Fold {fold + 1} completed! Best F1: {best_f1:.4f}")
        return best_metrics

    def validate_fold(self, model, val_loader):
        """Validate on a single fold."""
        model.eval()

        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for embeddings, masks, labels in val_loader:
                embeddings = embeddings.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                if self.scaler is not None:
                    with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        outputs = model(embeddings, masks)
                else:
                    outputs = model(embeddings, masks)

                # Get predictions
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = probabilities > config.PREDICTION_THRESHOLD

                # Collect results
                all_predictions.extend(predictions.cpu().float().numpy())
                all_probabilities.extend(probabilities.cpu().float().numpy())
                all_labels.extend(labels.cpu().float().numpy())

        # Compute metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)

        metrics = utils.compute_metrics(y_true, y_pred, y_proba)
        return metrics

    def train(self):
        """Main k-fold training loop."""
        print("\n" + "="*60)
        print("🚀 STARTING K-FOLD CROSS-VALIDATION")
        print("="*60)

        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=config.SEED)

        # Train each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.sequences, self.labels)):
            fold_metrics = self.train_fold(fold, train_idx, val_idx)
            self.fold_results.append({
                'fold': fold + 1,
                'metrics': fold_metrics
            })

        # Compute average metrics
        print("\n" + "="*60)
        print("📊 K-FOLD CROSS-VALIDATION RESULTS")
        print("="*60)

        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'auc' in self.fold_results[0]['metrics']:
            metrics_names.append('auc')

        avg_metrics = {}
        std_metrics = {}

        for metric_name in metrics_names:
            values = [fold['metrics'][metric_name] for fold in self.fold_results]
            avg_metrics[metric_name] = np.mean(values)
            std_metrics[metric_name] = np.std(values)

            print(f"{metric_name.upper():10s}: {avg_metrics[metric_name]:.4f} ± {std_metrics[metric_name]:.4f}")

        # Save results
        results = {
            'n_splits': self.n_splits,
            'fold_results': self.fold_results,
            'average_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'timestamp': datetime.now().isoformat()
        }

        results_file = f"kfold_results_{utils.get_timestamp()}.json"
        utils.save_results(results, results_file)

        print(f"\n🎉 K-fold cross-validation completed!")
        print(f"🎯 Average F1 Score: {avg_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}")

        return avg_metrics, std_metrics


def main():
    """Main function to run k-fold training."""
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation Training')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds (default: 5)')
    args = parser.parse_args()

    print("🧬 ESM-BASED PROTEIN CLASSIFIER - K-FOLD CROSS-VALIDATION")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"K-Folds: {args.n_splits}")
    print(f"ESM Model: {config.ESM_MODEL_NAME}")
    print(f"Data: {config.DATA_CSV_PATH}")
    print("="*60)

    try:
        # Create trainer
        trainer = KFoldESMTrainer(n_splits=args.n_splits)

        # Prepare data
        trainer.prepare_data()

        # Compute embeddings
        trainer.compute_embeddings()

        # Train with k-fold
        avg_metrics, std_metrics = trainer.train()

        print(f"\n🎉 Training completed successfully!")
        print(f"🎯 Average F1 Score: {avg_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
