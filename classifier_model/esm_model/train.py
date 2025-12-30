"""
Training script for ESM-based protein classifier.

This script trains the ESM classifier on protein sequence data.
Run this script to train a new model from scratch.

Usage:
    python train.py

The script will:
1. Load data from the configured CSV file
2. Compute ESM embeddings (cached for efficiency)
3. Train the classifier with the configured hyperparameters
4. Save the best model checkpoint
5. Display training progress and final results
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

# Import our modules
import config
import utils
from esm_embedder import ESMEmbedder
from esm_classifier import ESMClassifier, get_loss_function


class ESMTrainer:
    """
    Trainer class for ESM-based protein classifier.
    
    Handles the training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self):
        """Initialize the trainer."""
        # Set random seeds for reproducibility
        utils.set_random_seeds(config.SEED)
        
        # Get device
        self.device = config.get_device()
        print(f"[*] Using device: {self.device}")
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.USE_MIXED_PRECISION else None # speeds up training
        # 16 bit might go to underflow
        
        # Training state
        self.best_f1 = 0.0
        self.best_f1_score = 0.0  # Alias for compatibility with clean_and_retrain.py
        self.patience_counter = 0
        self.training_log = []
        
    def prepare_data(self):
        """Load and prepare training/validation data."""
        print("\n" + "="*60)
        print("[*] PREPARING DATA")
        print("="*60)
        
        # Load data
        sequences, labels = utils.load_protein_data(config.DATA_CSV_PATH)
        
        # Train/test split
        train_seqs, val_seqs, train_labels, val_labels = utils.create_train_test_split(
            sequences, labels, config.TEST_SIZE, config.RANDOM_STATE
        )
        
        # Compute ESM embeddings
        print(f"\n[*] Computing ESM embeddings...")
        embedder = ESMEmbedder() # create the ESMEmbedder object
        
        print("   Training embeddings...")
        train_embeddings, train_masks = embedder.embed_sequences(
            train_seqs, max_length=config.MAX_SEQUENCE_LENGTH # gets embeddings and attention masks on training sequences
        )
        
        print("   Validation embeddings...")
        val_embeddings, val_masks = embedder.embed_sequences( # gets embeddings and attention masks on the validation seq
            val_seqs, max_length=config.MAX_SEQUENCE_LENGTH
        )
        
        # Convert labels to tensors
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.float32)
        
        # Create data loaders
        self.train_loader = utils.create_data_loader(
            train_embeddings, train_masks, train_labels, 
            batch_size=config.BATCH_SIZE, shuffle=True
        )
        
        self.val_loader = utils.create_data_loader(
            val_embeddings, val_masks, val_labels,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        #^ bundles the data into batches and shuffles training data into each epoch (randomizes order)
        # handle efficient loading during training
        
        print(f"[*] Data prepared successfully!")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        
        # Store embedding dimension for model creation
        self.esm_embed_dim = train_embeddings.shape[-1] # gets the last parameter, which is how many dimensions there are in the embedding
        
    def create_model(self):
        """Create and initialize the model, loss, and optimizer."""
        print("\n" + "="*60)
        print("[*] CREATING MODEL")
        print("="*60)
        
        # Create model
        self.model = ESMClassifier(esm_embed_dim=self.esm_embed_dim)
        self.model = self.model.to(self.device)

        # Multi-GPU support
        if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
            gpu_ids = config.GPU_IDS if config.GPU_IDS else list(range(torch.cuda.device_count()))
            print(f"   Using {len(gpu_ids)} GPUs: {gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        else:
            print(f"   Using single device: {self.device}")
        
        # Create loss function
        self.criterion = get_loss_function(config.LOSS_TYPE) # use FocalLoss
        
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
        
        print(f"[*] Model created successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Loss function: {config.LOSS_TYPE}")
        print(f"   Optimizer: AdamW (lr={config.LEARNING_RATE})")
        print(f"   Scheduler: {'ReduceLROnPlateau' if config.USE_SCHEDULER else 'None'}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        print(f"\n[*] Starting Epoch {epoch+1}/{config.NUM_EPOCHS}...", flush=True)

        # Progress bar with explicit stdout for Windows
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}",
                    ncols=100, file=sys.stdout)
        
        for batch_idx, (embeddings, masks, labels) in enumerate(pbar):
            # Move to device
            embeddings = embeddings.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad() # reset from the last batch
            
            # Forward pass
            if self.scaler is not None: # mixed precision is self.scaler
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    outputs = self.model(embeddings, masks)
                    loss = self.criterion(outputs.squeeze(), labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward() # prevent underflow so the loss isn't 0 and then calls backward()
                """
                PyTorch's autograd traces
                back through the computation graph:
                focal_loss.mean()
                    ← focal_loss array (still in memory!)
                    ← focal_weight × bce_loss
                    ← p_t, alpha_t
                    ← probs, targets
                    ← model outputs
                    ← linear layers
                    ← attention pooling
                    ← all the way back to weights
                """


                # Gradient clipping
                self.scaler.unscale_(self.optimizer) 
                # clip limits gradients to max norm of 1.0 so no explosions happen
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_GRAD_NORM)
                
                # Optimizer step
                self.scaler.step(self.optimizer) # applies to all weights using the computed gradients (using AdamW algorithm)
                self.scaler.update() # updates the scaler's internal state for next iteration
            else:
                outputs = self.model(embeddings, masks)
                loss = self.criterion(outputs.squeeze(), labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_GRAD_NORM)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })
            


            # save the timestamps of the training data
            # Log training step
            if config.SAVE_TRAINING_LOG:
                self.training_log.append({
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now().isoformat()
                })
        
        avg_loss = total_loss / num_batches
        print(f"[*] Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}", flush=True)
        return avg_loss
        # return average loss across all batches, which is just used by train() to track progress
    
    def validate(self): # need to look at validation!
        """Validate the model and return metrics."""
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, masks, labels in self.val_loader:
                # Move to device
                embeddings = embeddings.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        outputs = self.model(embeddings, masks)
                else:
                    outputs = self.model(embeddings, masks)
                
                # Get predictions
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = probabilities > config.PREDICTION_THRESHOLD
                
                # Collect results
                all_predictions.extend(predictions.cpu().float().numpy())
                all_probabilities.extend(probabilities.cpu().float().numpy())
                all_labels.extend(labels.cpu().float().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Compute metrics
        metrics = utils.compute_metrics(y_true, y_pred, y_proba)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        # Create filename
        f1_score = metrics['f1_score'] # pulls out F1 score from metrix dict
        if is_best:
            filename = f"esm_model_BEST_f1_{f1_score:.4f}.pt" # saves the weights to a file
        else:
            filename = f"esm_model_epoch_{epoch+1}_f1_{f1_score:.4f}.pt"
        

        # Save checkpoint
        utils.save_model_checkpoint(
            self.model, self.optimizer, epoch, metrics, filename
        )
        # saves network weights, AdamW's internal state, training progress, metrics, and validation performance to one file


        # if we have a new is_best, then we delete the old checkpoints to save space, only 
        # saving the top N checkpoints
        if is_best:
            utils.cleanup_old_checkpoints()
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60, flush=True)
        print("[*] STARTING TRAINING", flush=True)
        print("="*60, flush=True)
        print(f"Epochs: {config.NUM_EPOCHS}", flush=True)
        print(f"Batch size: {config.BATCH_SIZE}", flush=True)
        print(f"Learning rate: {config.LEARNING_RATE}", flush=True)
        print(f"Early stopping patience: {config.PATIENCE}", flush=True)
        print("="*60, flush=True)
        
        for epoch in range(config.NUM_EPOCHS):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate every N epochs
            if (epoch + 1) % config.EVALUATE_EVERY_N_EPOCHS == 0:
                print(f"\n[*] Running validation for Epoch {epoch+1}...", flush=True)
                metrics = self.validate()
                f1_score = metrics['f1_score']

                # Print progress with explicit flushing
                print(f"\n{'='*60}", flush=True)
                print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}:", flush=True)
                print(f"  Train Loss: {train_loss:.4f}", flush=True)
                print(f"  Val F1:     {f1_score:.4f}", flush=True)
                print(f"  Val Acc:    {metrics['accuracy']:.4f}", flush=True)
                
                # Learning rate scheduler step
                if self.scheduler is not None:
                    self.scheduler.step(f1_score)
                
                # Check if best model
                is_best = f1_score > self.best_f1
                if is_best:
                    self.best_f1 = f1_score
                    self.best_f1_score = f1_score  # Also store as best_f1_score for compatibility
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
        print("[*] TRAINING COMPLETED")
        print("="*60)
        print(f"[*] Best F1 Score: {self.best_f1:.4f}")
        
        # Compare with benchmarks
        utils.compare_with_benchmarks(self.best_f1)
        
        # Save training log
        if config.SAVE_TRAINING_LOG and self.training_log:
            log_filename = f"training_log_{utils.get_timestamp()}.json"
            utils.save_results({'training_log': self.training_log}, log_filename)
        
        return self.best_f1


def main():
    """Main function to run training."""
    print("[*] ESM-BASED PROTEIN CLASSIFIER TRAINING")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ESM Model: {config.ESM_MODEL_NAME}")
    print(f"Data: {config.DATA_CSV_PATH}")
    print("="*60)
    
    try:
        # Create trainer
        trainer = ESMTrainer()
        
        # Prepare data
        trainer.prepare_data()
        
        # Create model
        trainer.create_model()
        
        # Train model
        best_f1 = trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🎯 Best F1 Score: {best_f1:.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()