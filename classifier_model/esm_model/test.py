"""
Testing script for ESM-based protein classifier.

This script loads a trained ESM model and evaluates it on test data.
It can also run predictions on new sequences.


Examples:
    python test.py  # Test best model on default test set
    python test.py --model_path esm_model_BEST_f1_0.9500.pt  # Test specific model
    python test.py --test_file custom_test.csv  # Use custom test file
    python test.py --predict_file new_sequences.csv  # Run predictions only
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")

# Import our modules
import config
import utils
from esm_embedder import ESMEmbedder
from esm_classifier import ESMClassifier


class ESMTester:
    """
    Testing class for ESM-based protein classifier.
    
    Handles model loading, testing, and prediction functionality.
    """
    
    def __init__(self, model_path=None):
        """Initialize the tester."""
        # Set random seeds for reproducibility
        utils.set_random_seeds(config.SEED)
        
        # Get device
        self.device = config.get_device()
        print(f"[*] Using device: {self.device}")
        
        # Initialize ESM embedder
        self.embedder = ESMEmbedder()
        
        # Load model
        self.model_path = model_path or self._find_best_model()
        self.model = self._load_model()
        
    def _find_best_model(self):
        """Find the best trained model in the current directory or checkpoints folder."""
        # Check current directory first
        model_files = [f for f in os.listdir('.') if f.startswith('esm_model_BEST_f1_') and f.endswith('.pt')]
        
        # Check checkpoints directory if no models in current directory
        if not model_files and os.path.exists('checkpoints'):
            checkpoint_files = [os.path.join('checkpoints', f) for f in os.listdir('checkpoints') 
                              if f.startswith('esm_model_BEST_f1_') and f.endswith('.pt')]
            model_files.extend(checkpoint_files)
        
        if not model_files:
            raise FileNotFoundError(
                "No trained model found! Please train a model first using train.py or specify --model_path"
            )
        
        # Sort by F1 score in filename and get the best one
        def extract_f1(filename):
            basename = os.path.basename(filename)
            return float(basename.split('_f1_')[1].replace('.pt', ''))
        
        model_files.sort(key=extract_f1, reverse=True)
        best_model = model_files[0]
        
        print(f"[*] Auto-selected best model: {best_model}")
        return best_model
        
    def _load_model(self):
        """Load the trained model."""
        print(f"\n[*] Loading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model architecture info
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            esm_embed_dim = model_config.get('esm_embed_dim', 480)  # Default for ESM2-35M
        else:
            # Try to infer from state dict
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Look for attention layer input size to infer embedding dimension
            for key, tensor in state_dict.items():
                if 'attention.query' in key and tensor.dim() == 2:
                    esm_embed_dim = tensor.shape[1]
                    break
            else:
                esm_embed_dim = 480  # Default fallback
        
        # Create model
        model = ESMClassifier(esm_embed_dim=esm_embed_dim)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        # Print model info
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"   Training F1: {metrics.get('f1_score', 'N/A')}")
            print(f"   Training Acc: {metrics.get('accuracy', 'N/A')}")
        
        print(f"[OK] Model loaded successfully!")
        return model
        
    def test_model(self, test_file=None):
        """Test the model on test data and return detailed results."""
        print("\n" + "="*60)
        print("[*] TESTING MODEL")
        print("="*60)
        
        # Load test data
        test_csv = test_file or config.DATA_CSV_PATH
        sequences, labels = utils.load_protein_data(test_csv)
        
        # If using training data, split to get test set
        if test_csv == config.DATA_CSV_PATH:
            print("[*] Using train/test split...")
            _, test_seqs, _, test_labels = utils.create_train_test_split(
                sequences, labels, config.TEST_SIZE, config.RANDOM_STATE
            )
        else:
            print(f"[*] Using provided test file: {test_csv}")
            test_seqs, test_labels = sequences, labels
        
        print(f"   Test samples: {len(test_seqs)}")
        
        # Compute embeddings
        print("\n[*] Computing ESM embeddings...")
        test_embeddings, test_masks = self.embedder.embed_sequences(
            test_seqs, max_length=config.MAX_SEQUENCE_LENGTH
        )
        
        # Create data loader
        test_loader = utils.create_data_loader(
            test_embeddings, test_masks, torch.tensor(test_labels, dtype=torch.float32),
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        
        # Run predictions
        print("[*] Running predictions...")
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, masks, labels in test_loader:
                # Move to device
                embeddings = embeddings.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(embeddings, masks)
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = probabilities > config.PREDICTION_THRESHOLD

                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Compute comprehensive metrics
        metrics = utils.compute_metrics(y_true, y_pred, y_proba)
        
        # Print results
        print("\n" + "="*60)
        print("[*] TEST RESULTS")
        print("="*60)
        print(f"    F1 Score:     {metrics['f1_score']:.4f}")
        print(f"    Accuracy:     {metrics['accuracy']:.4f}")
        print(f"    Precision:    {metrics['precision']:.4f}")
        print(f"    Recall:       {metrics['recall']:.4f}")
        if 'auc' in metrics:
            print(f"    AUC:          {metrics['auc']:.4f}")
        elif 'auc_roc' in metrics:
            print(f"    AUC-ROC:      {metrics['auc_roc']:.4f}")
        if 'auc_pr' in metrics:
            print(f"    AUC-PR:       {metrics['auc_pr']:.4f}")
        print("="*60)

        # Detailed classification report
        print("\n[*] CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n[*] CONFUSION MATRIX:")
        print(f"    Predicted:  0    1")
        print(f"Actual 0:    {cm[0,0]:4d} {cm[0,1]:4d}")
        print(f"Actual 1:    {cm[1,0]:4d} {cm[1,1]:4d}")

        # Compare with benchmarks
        print(f"\n[*] BENCHMARK COMPARISON:")
        utils.compare_with_benchmarks(metrics['f1_score'])
        
        # Save detailed results
        timestamp = utils.get_timestamp()
        results = {
            'model_path': self.model_path,
            'test_file': test_csv,
            'test_samples': len(test_seqs),
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'labels': y_true.tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': cm.tolist(),
            'timestamp': timestamp
        }
        
        results_file = f"test_results_{timestamp}.json"
        utils.save_results(results, results_file)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'sequence_index': range(len(test_seqs)),
            'sequence': test_seqs,
            'true_label': y_true,
            'predicted_label': y_pred,
            'probability': y_proba
        })
        
        csv_file = f"test_predictions_{timestamp}.csv"
        predictions_df.to_csv(csv_file, index=False)
        print(f"💾 Predictions saved to: {csv_file}")
        
        return metrics
        
    def predict_sequences(self, sequences_file, output_file=None):
        """Run predictions on new sequences from a file (streaming mode)."""
        print("\n" + "="*60)
        print("🔮 PREDICTING NEW SEQUENCES (STREAMING)")
        print("="*60)
        
        # Load sequences
        try:
            sequences, labels = utils.load_protein_data(sequences_file, require_labels=False)
            print(f"[*] Loaded {len(sequences)} sequences from {sequences_file}")
            if labels is not None:
                print(f"   Labels found: {len([l for l in labels if l is not None])}")
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            return None
            
        # Determine output file
        if output_file is None:
            timestamp = utils.get_timestamp()
            output_file = f"predictions_{timestamp}.csv"
            
        # Initialize output file with header
        header_df = pd.DataFrame(columns=['sequence_index', 'sequence', 'predicted_label', 'probability', 'confidence'])
        header_df.to_csv(output_file, index=False)
        
        # Processing parameters
        CHUNK_SIZE = 100 # Process 100 sequences at a time (Embedding + Prediction)
        total_sequences = len(sequences)
        
        print(f"🔄 Processing in chunks of {CHUNK_SIZE}...")
        
        total_pred_0 = 0
        total_pred_1 = 0
        high_conf_count = 0
        
        from tqdm import tqdm
        
        # Main processing loop
        for i in tqdm(range(0, total_sequences, CHUNK_SIZE), desc="Predicting"):
            chunk_seqs = sequences[i:i + CHUNK_SIZE]
            chunk_indices = list(range(i, i + len(chunk_seqs)))
            
            # 1. Compute Embeddings for Chunk
            # Note: We access the internal model directly to avoid full storage in 'embed_sequences'
            # But the existing `embed_sequences` returns everything.
            # We should use `embed_sequences` on the small chunk. It's safe now.
            
            embeddings, masks = self.embedder.embed_sequences(
                chunk_seqs, max_length=config.MAX_SEQUENCE_LENGTH, show_progress=False
            )
            
            # 2. Run Prediction on Chunk
            with torch.no_grad():
                embeddings = embeddings.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(embeddings, masks)
                probabilities = torch.sigmoid(outputs.squeeze())
                
                # Handle single-item batch case (0-dim tensor)
                if probabilities.ndim == 0:
                    probabilities = probabilities.unsqueeze(0)
                    
                predictions = probabilities > config.PREDICTION_THRESHOLD
                
                # Move to CPU immediately
                probs_np = probabilities.cpu().numpy()
                preds_np = predictions.cpu().numpy()
                
            # 3. Process Results
            chunk_results = []
            for j, seq in enumerate(chunk_seqs):
                prob = float(probs_np[j])
                pred = int(preds_np[j])
                conf = abs(prob - 0.5) * 2
                
                # Update stats
                if pred == 0: total_pred_0 += 1
                else: total_pred_1 += 1
                if conf > 0.8: high_conf_count += 1
                
                chunk_results.append({
                    'sequence_index': chunk_indices[j],
                    'sequence': seq,
                    'predicted_label': pred,
                    'probability': prob,
                    'confidence': conf
                })
                
            # 4. Append to CSV
            chunk_df = pd.DataFrame(chunk_results)
            chunk_df.to_csv(output_file, mode='a', header=False, index=False)
            
            # Explicit memory cleanup
            del embeddings, masks, outputs, probabilities
            torch.cuda.empty_cache()
            
        # Print summary
        print("\n📊 PREDICTION SUMMARY:")
        print(f"   Total sequences: {total_sequences}")
        print(f"   Predicted Class 0: {total_pred_0}")
        print(f"   Predicted Class 1: {total_pred_1}")
        print(f"   High confidence (>0.8): {high_conf_count}")
        print(f"\n💾 Predictions saved to: {output_file}")
        
        return pd.read_csv(output_file) # return sample
        
    def visualize_results(self, test_file=None):
        """Create visualizations of test results."""
        print("\n🎨 Creating visualizations...")
        
        # Run test to get data
        metrics = self.test_model(test_file)
        
        # Note: This is a basic framework. Full implementation would create
        # plots for ROC curve, PR curve, confusion matrix heatmap, etc.
        print("📊 Visualization framework ready (plots would be generated here)")
        
        return metrics


def main():
    """Main function for testing."""
    parser = argparse.ArgumentParser(description='Test ESM-based protein classifier')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--test_file', type=str, help='Path to test CSV file')
    parser.add_argument('--predict_file', type=str, help='Path to sequences file for prediction')
    parser.add_argument('--output_file', type=str, help='Output file for predictions')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    print("[*] ESM-BASED PROTEIN CLASSIFIER TESTING")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_path or 'Auto-detect best model'}")
    print("="*60)
    
    try:
        # Create tester
        tester = ESMTester(model_path=args.model_path)
        
        if args.predict_file:
            # Run predictions only
            tester.predict_sequences(args.predict_file, args.output_file)
        else:
            # Run testing
            if args.visualize:
                tester.visualize_results(args.test_file)
            else:
                tester.test_model(args.test_file)
        
        print(f"\n[OK] Testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[WARN] Testing interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Testing failed with error: {e}")
        raise


if __name__ == "__main__":
    main()