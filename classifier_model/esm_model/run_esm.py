"""
Quick start script for ESM-based protein classifier.

This script provides an easy way to train and test the ESM model.
It handles the full pipeline from data loading to evaluation.

Usage:
    python run_esm.py [--mode MODE] [--data_file DATA_FILE]

Modes:
    train     - Train a new model (default)
    test      - Test the best existing model
    predict   - Run predictions on new sequences
    full      - Train and then test the model

Examples:
    python run_esm.py                           # Train with default settings
    python run_esm.py --mode test               # Test best model
    python run_esm.py --mode full               # Train then test
    python run_esm.py --mode predict --data_file new_sequences.csv
"""

import argparse
import os
import sys
from datetime import datetime

# Import our modules
import config
from train import main as train_main
from test import ESMTester


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        ('torch', 'torch'), 
        ('transformers', 'transformers'), 
        ('numpy', 'numpy'), 
        ('pandas', 'pandas'), 
        ('sklearn', 'scikit-learn'), 
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_data_file():
    """Check if the data file exists."""
    if not os.path.exists(config.DATA_CSV_PATH):
        print(f"❌ Data file not found: {config.DATA_CSV_PATH}")
        print("\n📊 Make sure your data file exists and update DATA_CSV_PATH in config.py")
        print("   Expected format: CSV with 'sequence' and 'label' columns")
        return False
    
    return True


def run_train():
    """Run training."""
    print("🚀 Starting training...")
    print(f"📊 Data file: {config.DATA_CSV_PATH}")
    print(f"🧬 ESM model: {config.ESM_MODEL_NAME}")
    print(f"⚙️ Epochs: {config.NUM_EPOCHS}")
    print(f"📊 Batch size: {config.BATCH_SIZE}")
    print("-" * 50)
    
    # Import and run training
    from train import main as train_main
    train_main()


def run_test(data_file=None):
    """Run testing."""
    print("🧪 Starting testing...")
    
    tester = ESMTester()
    metrics = tester.test_model(test_file=data_file)
    
    return metrics


def run_predict(data_file):
    """Run predictions."""
    print("🔮 Starting predictions...")
    
    if not data_file:
        print("❌ No data file specified for predictions")
        return None
    
    if not os.path.exists(data_file):
        print(f"❌ Prediction file not found: {data_file}")
        return None
    
    tester = ESMTester()
    results = tester.predict_sequences(data_file)
    
    return results


def run_full():
    """Run full pipeline: train then test."""
    print("🎯 Running full pipeline (train + test)...")
    
    # Train model
    print("\n" + "="*60)
    print("STEP 1: TRAINING")
    print("="*60)
    run_train()
    
    # Test model
    print("\n" + "="*60)
    print("STEP 2: TESTING")
    print("="*60)
    metrics = run_test()
    
    return metrics


def show_config():
    """Display current configuration."""
    print("⚙️ CURRENT CONFIGURATION:")
    print("-" * 30)
    print(f"Data file:        {config.DATA_CSV_PATH}")
    print(f"ESM model:        {config.ESM_MODEL_NAME}")
    print(f"Max seq length:   {config.MAX_SEQUENCE_LENGTH}")
    print(f"Batch size:       {config.BATCH_SIZE}")
    print(f"Learning rate:    {config.LEARNING_RATE}")
    print(f"Epochs:           {config.NUM_EPOCHS}")
    print(f"Early stopping:   {config.PATIENCE} epochs")
    print(f"Device:           {config.get_device()}")
    print(f"Cache directory:  {config.ESM_CACHE_DIR}")
    print("-" * 30)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ESM protein classifier runner')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict', 'full'], # what do we want 'predict' and 'full' to do?
                       default='train', help='Mode to run')
    parser.add_argument('--data_file', type=str, help='Data file for testing or prediction')
    parser.add_argument('--show_config', action='store_true', help='Show configuration and exit')
    
    args = parser.parse_args()
    
    # Show header
    print("🧬 ESM-BASED PROTEIN CLASSIFIER")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print("=" * 50)
    
    # Show config if requested
    if args.show_config:
        show_config()
        return
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        return
    
    # Check data file for non-predict modes
    if args.mode != 'predict':
        if not check_data_file():
            return
    
    print("✅ All checks passed!")
    
    # Show current config
    show_config()
    
    try:
        # Run based on mode
        if args.mode == 'train':
            run_train()
            
        elif args.mode == 'test':
            metrics = run_test(args.data_file)
            print(f"\n🎯 Final F1 Score: {metrics['f1_score']:.4f}")
            
        elif args.mode == 'predict':
            if not args.data_file:
                print("❌ --data_file is required for prediction mode")
                return
            results = run_predict(args.data_file)
            if results is not None:
                print(f"\n🔮 Predictions completed for {len(results)} sequences")
                
        elif args.mode == 'full':
            metrics = run_full()
            print(f"\n🎯 Final F1 Score: {metrics['f1_score']:.4f}")
        
        print(f"\n🎉 {args.mode.title()} completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ {args.mode.title()} interrupted by user")
    except Exception as e:
        print(f"\n❌ {args.mode.title()} failed with error: {e}")
        raise


if __name__ == "__main__":
    main()