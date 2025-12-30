
import os
import sys
import pandas as pd
import torch
from pathlib import Path

# Add the classifier_model/esm_model to path so we can import modules
sys.path.insert(0, os.path.join(os.getcwd(), 'classifier_model', 'esm_model'))

import config
import utils
from train import ESMTrainer
from test import ESMTester

# Configuration for this script
# Configuration for this script
DATA_INPUT = "data/raw_sources/labeled_sequences_original.csv" # The master source
if not os.path.exists(DATA_INPUT):
    DATA_INPUT = "data/labeled_sequences.csv" # Fallback

DATA_OUTPUT = "data/cleaned_labeled_sequences.csv"

KEYWORDS_TO_FLIP = [
    "kinase", 
    "synthase", 
    "ribosomal", 
    "transferase", 
    "reductase", 
    "mutase",
    "transcription",
    "zinc finger",
    "regulator",
    "nuclear",
    "ubiquitin",
    "polymerase",
    "nuclease",
    "receptor",
    "factor"
]


def clean_dataset():
    print("\n" + "="*60)
    print("[*] STEP 1: LOADING SANITIZED DATA (SAFE MODE)")
    print("="*60)
    
    # Use the pre-sanitized file from diagnostic_check.py
    SANITIZED_PATH = "data/READY_TO_TRAIN.csv"
    
    if not os.path.exists(SANITIZED_PATH):
        print(f"[!] Error: Sanitized file {SANITIZED_PATH} not found!")
        raise FileNotFoundError("Run diagnostic_check.py first!")

    print(f"[*] Loading {SANITIZED_PATH}...")
    df = pd.read_csv(SANITIZED_PATH)
    print(f"   Loaded {len(df)} rows.")

    # Skip legacy flipping logic
    flipped_count = 0 
    print(f"   Skipping keyword flipping (assumed clean).")
    
    # ---------------------------------------------------------
    # OVERSAMPLING STEP (The Fix for 'Nitrate Blindness')
    # ---------------------------------------------------------
    print("[*] STEP 1.5: OVERSAMPLING NITRATE TRANSPORTERS")
    
    # Filter for Nitrate/Ammonium Transporters
    # Criteria: Label 0 (Transporter) AND "nitrate" or "ammonium" in description
    nitrate_rows = df[
        (df['Label'] == 0) & 
        (df['Protein Name'].str.lower().str.contains('nitrate|ammonium', na=False))
    ]
    
    print(f"   Found {len(nitrate_rows)} Nitrate/Ammonium sequences.")
    
    if len(nitrate_rows) > 0:
        # Calculate how many copies we need to make a dent
        # We want at least 150 examples total to be significant
        duplication_factor = 10 
        print(f"   Duplicating them {duplication_factor}x to balance the dataset...")
        
        oversampled_data = pd.concat([nitrate_rows] * duplication_factor, ignore_index=True)
        df = pd.concat([df, oversampled_data], ignore_index=True)
        
        print(f"   Added {len(oversampled_data)} synthetic examples.")
    else:
        print("   [!] WARNING: No Nitrate sequences found to oversample!")

    # ---------------------------------------------------------

    # Save
    os.makedirs(os.path.dirname(DATA_OUTPUT), exist_ok=True)
    df.to_csv(DATA_OUTPUT, index=False)
    
    # Final Stats
    final_counts = df['Label'].value_counts().to_dict()
    print(f"   Final Balance: {final_counts}")
    print(f"[*] Saved Clean Dataset to: {DATA_OUTPUT}")
    return True

def retrain_model():
    print("\n" + "="*60)
    print("[*] STEP 2: RETRAINING ON CLEAN DATA (THE VICTORY LAP)")
    print("="*60)
    
    # Override Config
    config.DATA_CSV_PATH = os.path.abspath(DATA_OUTPUT)
    print(f"[*] Overridden Config Data Path: {config.DATA_CSV_PATH}")
    
    # Initialize Trainer
    trainer = ESMTrainer()
    
    try:
        print("[DEBUG] Preparing data & embeddings...", flush=True)
        trainer.prepare_data()
        print("[DEBUG] Creating model...", flush=True)
        trainer.create_model()
        
        # Train
        print("[DEBUG] Starting main training loop...", flush=True)
        best_f1 = trainer.train()
        print(f"[DEBUG] Training finished. Best F1: {best_f1}", flush=True)
    except Exception as e:
        print(f"\n[!!!] CRITICAL TRAINING ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise e
    
    # Rename the best model to desired name
    # The trainer saves as esm_model_BEST_f1_XXXX.pt
    # We want esm_model_FINAL_CLEAN.pt
    
    checkpoint_dir = config.CHECKPOINT_DIR
    best_model_name = f"esm_model_BEST_f1_{best_f1:.4f}.pt"
    source_path = os.path.join(checkpoint_dir, best_model_name)
    dest_path = os.path.join(checkpoint_dir, "esm_model_FINAL_CLEAN.pt")
    
    if os.path.exists(source_path):
        import shutil
        shutil.copy2(source_path, dest_path)
        print(f"\n[*] Saved FINAL CLEAN MODEL to: {dest_path}")
        return dest_path
    else:
        print(f"[!] Warning: Could not find best model file {source_path} to rename.")
        return source_path

def final_report(model_path):
    print("\n" + "="*60)
    print("[*] STEP 3: FINAL CLASSIFICATION REPORT")
    print("="*60)
    
    # We need a test set. The trainer split the data randomly.
    # Ideally, we should use the SAME test set split that the trainer just used.
    # But utils.create_test_split is random.
    # To be precise, we should have saved the split in the trainer.
    
    # For now, we will perform a split on the clean data (using the same seed as trainer)
    # and evaluate on the test portion.
    
    sequences, labels = utils.load_protein_data(DATA_OUTPUT)
    train_seqs, test_seqs, train_labels, test_labels = utils.create_train_test_split(
        sequences, labels, config.TEST_SIZE, config.RANDOM_STATE
    )
    
    # Save temp test file
    test_df = pd.DataFrame({'Sequence': test_seqs, 'Label': test_labels})
    temp_test_file = "data/temp_test_clean.csv"
    test_df.to_csv(temp_test_file, index=False)
    
    # Run Eval
    tester = ESMTester(model_path=model_path)
    metrics = tester.test_model(temp_test_file)
    
    # Print nice report
    print("\n" + "-"*40)
    print("VICTORY LAP RESULTS")
    print("-" * 40)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print("-" * 40)
    
    # Cleanup
    if os.path.exists(temp_test_file):
        os.remove(temp_test_file)

if __name__ == "__main__":
    if clean_dataset():
        final_model_path = retrain_model()
        final_report(final_model_path)
