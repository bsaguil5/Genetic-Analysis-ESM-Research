"""
Multi-Task Training Script for ESM-based Protein Classifier

This script trains the multi-task ESM classifier on protein sequence data.
The model learns to predict:
1. Binary classification (transporter vs non-transporter) - PRIMARY TASK
2. Subfamily classification (ABC, SWEET, NRT, etc.) - AUXILIARY
3. Substrate type (sugar, nitrogen, metal, etc.) - AUXILIARY
4. TM domain count (regression) - AUXILIARY

Usage:
    python train_multitask.py

Author: Brandon & Claude
Date: December 24, 2025
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd

# Import our modules
import config
import utils
from esm_embedder import ESMEmbedder
from esm_classifier_multitask import (
    ESMClassifier_MultiTask,
    MultiTaskLoss,
    create_multitask_classifier
)


class MultiTaskTrainer:
    """
    Trainer class for multi-task ESM protein classifier.

    Trains a single model to predict multiple related tasks simultaneously,
    improving generalization and performance on rare classes.
    """

    def __init__(self):
        """Initialize the multi-task trainer."""
        # Set random seeds for reproducibility
        utils.set_random_seeds(config.SEED)

        # Get device
        self.device = config.get_device()
        print(f"[*] Using device: {self.device}")

        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.USE_MIXED_PRECISION else None

        # Training state
        self.best_f1 = 0.0  # Best binary F1 (primary task)
        self.best_f1_score = 0.0  # Alias for compatibility
        self.patience_counter = 0
        self.training_log = []

        # Load label mappings
        mappings_path = os.path.join(os.getcwd(), '..', '..', 'data', 'multitask_label_mappings.json')
        if not os.path.exists(mappings_path):
            # Try from root directory
            mappings_path = 'data/multitask_label_mappings.json'
        with open(mappings_path, 'r') as f:
            self.label_mappings = json.load(f)

        self.num_subfamilies = len(self.label_mappings['subfamily_to_idx'])
        self.num_substrates = len(self.label_mappings['substrate_to_idx'])

        print(f"[*] Loaded label mappings:")
        print(f"   Subfamilies: {self.num_subfamilies} classes")
        print(f"   Substrates: {self.num_substrates} classes")

    def prepare_data(self):
        """Load and prepare multi-task training/validation data."""
        print("\n" + "="*60)
        print("[*] PREPARING MULTI-TASK DATA")
        print("="*60)

        # Load multi-task dataset
        data_path = os.path.join(os.getcwd(), '..', '..', 'data', 'multitask_labeled_sequences.csv')
        if not os.path.exists(data_path):
            # Try from root directory
            data_path = 'data/multitask_labeled_sequences.csv'
        df = pd.read_csv(data_path)
        print(f"[*] Loaded {len(df)} sequences with multi-task labels")

        # Extract data
        sequences = df['Sequence'].tolist()

        # Primary task: binary labels
        binary_labels = df['Label'].values

        # Auxiliary tasks
        subfamily_labels = df['Subfamily'].map(
            self.label_mappings['subfamily_to_idx']
        ).values

        substrate_labels = df['Substrate'].map(
            self.label_mappings['substrate_to_idx']
        ).values

        tm_domain_labels = df['TM_Domains'].values

        print(f"\n[*] Label distribution:")
        print(f"   Binary: {np.bincount(binary_labels.astype(int))}")
        print(f"   Subfamilies: {np.bincount(subfamily_labels.astype(int))}")
        print(f"   Substrates: {np.bincount(substrate_labels.astype(int))}")
        print(f"   TM domains: min={tm_domain_labels.min()}, max={tm_domain_labels.max()}")

        # Train/test split (stratified on binary label)
        from sklearn.model_selection import train_test_split

        # Create indices for splitting
        indices = np.arange(len(sequences))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=binary_labels
        )

        # Split data
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]

        train_binary = binary_labels[train_idx]
        val_binary = binary_labels[val_idx]

        train_subfamily = subfamily_labels[train_idx]
        val_subfamily = subfamily_labels[val_idx]

        train_substrate = substrate_labels[train_idx]
        val_substrate = substrate_labels[val_idx]

        train_tm = tm_domain_labels[train_idx]
        val_tm = tm_domain_labels[val_idx]

        print(f"\n[*] Split: {len(train_seqs)} train, {len(val_seqs)} val")

        # Compute ESM embeddings
        print(f"\n[*] Computing ESM embeddings...")
        embedder = ESMEmbedder()

        print("   Training embeddings...")
        train_embeddings, train_masks = embedder.embed_sequences(
            train_seqs, max_length=config.MAX_SEQUENCE_LENGTH
        )

        print("   Validation embeddings...")
        val_embeddings, val_masks = embedder.embed_sequences(
            val_seqs, max_length=config.MAX_SEQUENCE_LENGTH
        )

        # Convert labels to tensors
        train_labels = {
            'binary': torch.tensor(train_binary, dtype=torch.float32),
            'subfamily': torch.tensor(train_subfamily, dtype=torch.long),
            'substrate': torch.tensor(train_substrate, dtype=torch.long),
            'tm_count': torch.tensor(train_tm, dtype=torch.float32)
        }

        val_labels = {
            'binary': torch.tensor(val_binary, dtype=torch.float32),
            'subfamily': torch.tensor(val_subfamily, dtype=torch.long),
            'substrate': torch.tensor(val_substrate, dtype=torch.long),
            'tm_count': torch.tensor(val_tm, dtype=torch.float32)
        }

        # Create data loaders for multi-task learning
        self.train_loader = self._create_multitask_loader(
            train_embeddings, train_masks, train_labels,
            batch_size=config.BATCH_SIZE, shuffle=True
        )

        self.val_loader = self._create_multitask_loader(
            val_embeddings, val_masks, val_labels,
            batch_size=config.BATCH_SIZE, shuffle=False
        )

        print(f"[*] Data prepared successfully!")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")

        # Store embedding dimension
        self.esm_embed_dim = train_embeddings.shape[-1]

    def _create_multitask_loader(self, embeddings, masks, labels_dict, batch_size, shuffle):
        """Create data loader for multi-task learning."""
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(
            embeddings, masks,
            labels_dict['binary'],
            labels_dict['subfamily'],
            labels_dict['substrate'],
            labels_dict['tm_count']
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

    def create_model(self):
        """Create and initialize the multi-task model, loss, and optimizer."""
        print("\n" + "="*60)
        print("[*] CREATING MULTI-TASK MODEL")
        print("="*60)

        # Create multi-task model
        self.model = create_multitask_classifier(
            esm_embed_dim=self.esm_embed_dim,
            num_subfamilies=self.num_subfamilies,
            num_substrates=self.num_substrates
        )
        self.model = self.model.to(self.device)

        # Multi-GPU support
        if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
            gpu_ids = config.GPU_IDS if config.GPU_IDS else list(range(torch.cuda.device_count()))
            print(f"   Using {len(gpu_ids)} GPUs: {gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        else:
            print(f"   Using single device: {self.device}")

        # Create multi-task loss function
        self.criterion = MultiTaskLoss(
            binary_weight=1.0,      # Primary task
            subfamily_weight=0.5,   # Auxiliary
            substrate_weight=0.3,   # Auxiliary
            tm_weight=0.2,          # Auxiliary
            focal_alpha=config.FOCAL_ALPHA,
            focal_gamma=config.FOCAL_GAMMA
        )

        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Create learning rate scheduler
        if config.USE_SCHEDULER:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE
            )
        else:
            self.scheduler = None

        print(f"[*] Multi-task model created successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Loss weights: Binary=1.0, Subfamily=0.5, Substrate=0.3, TM=0.2")
        print(f"   Optimizer: AdamW (lr={config.LEARNING_RATE})")
        print(f"   Scheduler: {'ReduceLROnPlateau' if config.USE_SCHEDULER else 'None'}")

    def train_epoch(self, epoch):
        """Train for one epoch with multi-task learning."""
        self.model.train()
        total_loss = 0.0
        total_losses = {'total': 0.0, 'binary': 0.0, 'subfamily': 0.0,
                       'substrate': 0.0, 'tm_count': 0.0}
        num_batches = 0

        print(f"\n[*] Starting Epoch {epoch+1}/{config.NUM_EPOCHS}...", flush=True)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}",
                    ncols=100, file=sys.stdout)

        for batch_data in pbar:
            # Unpack multi-task batch
            embeddings, masks, binary_labels, subfamily_labels, substrate_labels, tm_labels = batch_data

            # Move to device
            embeddings = embeddings.to(self.device)
            masks = masks.to(self.device)

            targets = {
                'binary': binary_labels.to(self.device),
                'subfamily': subfamily_labels.to(self.device),
                'substrate': substrate_labels.to(self.device),
                'tm_count': tm_labels.to(self.device)
            }

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            if self.scaler is not None:
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    predictions = self.model(embeddings, masks)
                    loss, loss_dict = self.criterion(predictions, targets)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_GRAD_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(embeddings, masks)
                loss, loss_dict = self.criterion(predictions, targets)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_GRAD_NORM)
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Binary': f'{loss_dict["binary"]:.4f}',
                'SubF': f'{loss_dict["subfamily"]:.4f}'
            })

            # Log training step
            if config.SAVE_TRAINING_LOG:
                self.training_log.append({
                    'epoch': epoch + 1,
                    'batch': len(self.training_log),
                    'total_loss': loss.item(),
                    'binary_loss': loss_dict['binary'],
                    'subfamily_loss': loss_dict['subfamily'],
                    'substrate_loss': loss_dict['substrate'],
                    'tm_loss': loss_dict['tm_count'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now().isoformat()
                })

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        print(f"[*] Epoch {epoch+1} completed.", flush=True)
        print(f"   Total: {avg_losses['total']:.4f}, Binary: {avg_losses['binary']:.4f}, "
              f"Subfamily: {avg_losses['subfamily']:.4f}, Substrate: {avg_losses['substrate']:.4f}, "
              f"TM: {avg_losses['tm_count']:.4f}", flush=True)

        return avg_losses

    def validate(self):
        """Validate the multi-task model and return metrics for all tasks."""
        self.model.eval()

        # Collect predictions and labels for all tasks
        binary_preds, binary_probs, binary_true = [], [], []
        subfamily_preds, subfamily_true = [], []
        substrate_preds, substrate_true = [], []
        tm_preds, tm_true = [], []

        with torch.no_grad():
            for batch_data in self.val_loader:
                embeddings, masks, binary_labels, subfamily_labels, substrate_labels, tm_labels = batch_data

                # Move to device
                embeddings = embeddings.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                if self.scaler is not None:
                    with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        predictions = self.model(embeddings, masks)
                else:
                    predictions = self.model(embeddings, masks)

                # Binary task
                binary_prob = torch.sigmoid(predictions['binary'].squeeze())
                binary_pred = binary_prob > config.PREDICTION_THRESHOLD
                binary_preds.extend(binary_pred.cpu().float().numpy())
                binary_probs.extend(binary_prob.cpu().float().numpy())
                binary_true.extend(binary_labels.cpu().float().numpy())

                # Subfamily task
                subfamily_pred = torch.argmax(predictions['subfamily'], dim=1)
                subfamily_preds.extend(subfamily_pred.cpu().numpy())
                subfamily_true.extend(subfamily_labels.cpu().numpy())

                # Substrate task
                substrate_pred = torch.argmax(predictions['substrate'], dim=1)
                substrate_preds.extend(substrate_pred.cpu().numpy())
                substrate_true.extend(substrate_labels.cpu().numpy())

                # TM count task
                tm_pred = predictions['tm_count'].squeeze()
                tm_preds.extend(tm_pred.cpu().numpy())
                tm_true.extend(tm_labels.cpu().numpy())

        # Convert to numpy arrays
        binary_true = np.array(binary_true)
        binary_pred = np.array(binary_preds)
        binary_prob = np.array(binary_probs)

        # Compute metrics for binary task (primary)
        binary_metrics = utils.compute_metrics(binary_true, binary_pred, binary_prob)

        # Compute accuracy for auxiliary tasks
        subfamily_acc = np.mean(np.array(subfamily_preds) == np.array(subfamily_true))
        substrate_acc = np.mean(np.array(substrate_preds) == np.array(substrate_true))
        tm_mae = np.mean(np.abs(np.array(tm_preds) - np.array(tm_true)))

        # Combine all metrics
        metrics = {
            **binary_metrics,  # Unpack binary metrics (accuracy, f1, precision, recall, etc.)
            'subfamily_accuracy': subfamily_acc,
            'substrate_accuracy': substrate_acc,
            'tm_mae': tm_mae
        }

        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save multi-task model checkpoint."""
        f1_score = metrics['f1_score']
        if is_best:
            filename = f"esm_multitask_BEST_f1_{f1_score:.4f}.pt"
        else:
            filename = f"esm_multitask_epoch_{epoch+1}_f1_{f1_score:.4f}.pt"

        utils.save_model_checkpoint(
            self.model, self.optimizer, epoch, metrics, filename
        )

        if is_best:
            utils.cleanup_old_checkpoints()  # Remove old checkpoints

    def train(self):
        """Main multi-task training loop."""
        print("\n" + "="*60, flush=True)
        print("[*] STARTING MULTI-TASK TRAINING", flush=True)
        print("="*60, flush=True)
        print(f"Epochs: {config.NUM_EPOCHS}", flush=True)
        print(f"Batch size: {config.BATCH_SIZE}", flush=True)
        print(f"Learning rate: {config.LEARNING_RATE}", flush=True)
        print(f"Early stopping patience: {config.PATIENCE}", flush=True)
        print("="*60, flush=True)

        for epoch in range(config.NUM_EPOCHS):
            # Train epoch
            train_losses = self.train_epoch(epoch)

            # Validate every N epochs
            if (epoch + 1) % config.EVALUATE_EVERY_N_EPOCHS == 0:
                print(f"\n[*] Running validation for Epoch {epoch+1}...", flush=True)
                metrics = self.validate()
                f1_score = metrics['f1_score']

                # Print progress
                print(f"\n{'='*60}", flush=True)
                print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}:", flush=True)
                print(f"  Binary Task:", flush=True)
                print(f"    F1:  {f1_score:.4f}", flush=True)
                print(f"    Acc: {metrics['accuracy']:.4f}", flush=True)
                print(f"  Auxiliary Tasks:", flush=True)
                print(f"    Subfamily Acc: {metrics['subfamily_accuracy']:.4f}", flush=True)
                print(f"    Substrate Acc: {metrics['substrate_accuracy']:.4f}", flush=True)
                print(f"    TM MAE:        {metrics['tm_mae']:.2f}", flush=True)

                # Learning rate scheduler step
                if self.scheduler is not None:
                    self.scheduler.step(f1_score)

                # Check if best model (based on primary task F1)
                is_best = f1_score > self.best_f1
                if is_best:
                    self.best_f1 = f1_score
                    self.best_f1_score = f1_score
                    self.patience_counter = 0
                    print(f"  [*] New best F1: {f1_score:.4f}", flush=True)
                else:
                    self.patience_counter += 1
                    print(f"  Patience: {self.patience_counter}/{config.PATIENCE}", flush=True)

                # Save checkpoint
                if config.SAVE_EVERY_N_EPOCHS > 0 and (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
                    print(f"  [*] Saving checkpoint...", flush=True)
                    self.save_checkpoint(epoch, metrics, is_best=False)

                if is_best:
                    print(f"  [*] Saving BEST checkpoint...", flush=True)
                    self.save_checkpoint(epoch, metrics, is_best=True)
                print(f"{'='*60}\n", flush=True)

                # Early stopping
                if self.patience_counter >= config.PATIENCE:
                    print(f"\n[*] Early stopping triggered after {config.PATIENCE} epochs without improvement")
                    break

        print("\n" + "="*60)
        print("[*] MULTI-TASK TRAINING COMPLETED")
        print("="*60)
        print(f"[*] Best Binary F1 Score: {self.best_f1:.4f}")

        # Save training log
        if config.SAVE_TRAINING_LOG and self.training_log:
            log_filename = f"training_log_multitask_{utils.get_timestamp()}.json"
            utils.save_results({'training_log': self.training_log}, log_filename)

        return self.best_f1


def main():
    """Main function to run multi-task training."""
    print("[*] MULTI-TASK ESM PROTEIN CLASSIFIER TRAINING")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ESM Model: {config.ESM_MODEL_NAME}")
    print(f"Data: data/multitask_labeled_sequences.csv")
    print("="*60)

    try:
        # Create trainer
        trainer = MultiTaskTrainer()

        # Prepare data
        trainer.prepare_data()

        # Create model
        trainer.create_model()

        # Train model
        best_f1 = trainer.train()

        print(f"\n[SUCCESS] Multi-task training completed!")
        print(f"[RESULT] Best Binary F1 Score: {best_f1:.4f}")

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
