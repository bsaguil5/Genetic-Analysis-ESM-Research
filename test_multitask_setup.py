"""
Quick test to verify multi-task setup before full training.

Tests:
1. Multi-task dataset loads correctly
2. Label mappings are valid
3. Model can be instantiated
4. Forward pass works
5. Loss computation works

Author: Brandon & Claude
Date: December 24, 2025
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np

# Add classifier path
sys.path.insert(0, os.path.join(os.getcwd(), 'classifier_model', 'esm_model'))

from esm_classifier_multitask import (
    ESMClassifier_MultiTask,
    MultiTaskLoss,
    create_multitask_classifier
)
import config


def test_dataset():
    """Test that multi-task dataset was created correctly."""
    print("\n" + "="*60)
    print("TEST 1: Multi-Task Dataset")
    print("="*60)

    # Load dataset
    df = pd.read_csv('data/multitask_labeled_sequences.csv')
    print(f"[OK] Dataset loaded: {len(df)} sequences")

    # Check columns
    required_cols = ['Sequence', 'Label', 'Subfamily', 'Substrate', 'TM_Domains']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[FAIL] Missing columns: {missing}")
        return False
    print(f"[OK] All required columns present")

    # Check label distributions
    print(f"\n  Binary labels: {df['Label'].value_counts().to_dict()}")
    print(f"  Subfamilies: {len(df['Subfamily'].unique())} unique")
    print(f"  Substrates: {len(df['Substrate'].unique())} unique")
    print(f"  TM domains: {df['TM_Domains'].value_counts().to_dict()}")

    return True


def test_label_mappings():
    """Test that label mappings are valid."""
    print("\n" + "="*60)
    print("TEST 2: Label Mappings")
    print("="*60)

    # Load mappings
    with open('data/multitask_label_mappings.json', 'r') as f:
        mappings = json.load(f)
    print(f"[OK] Mappings loaded")

    # Check structure
    required_keys = ['subfamily_to_idx', 'idx_to_subfamily',
                    'substrate_to_idx', 'idx_to_subfamily']
    for key in required_keys:
        if key not in mappings:
            print(f"[FAIL] Missing key: {key}")
            return False
    print(f"[OK] All mapping keys present")

    # Check counts
    num_subfamilies = len(mappings['subfamily_to_idx'])
    num_substrates = len(mappings['substrate_to_idx'])
    print(f"\n  Subfamilies: {num_subfamilies} classes")
    print(f"  Substrates: {num_substrates} classes")

    return True, num_subfamilies, num_substrates


def test_model_creation(num_subfamilies, num_substrates):
    """Test that multi-task model can be created."""
    print("\n" + "="*60)
    print("TEST 3: Model Creation")
    print("="*60)

    # Create model
    esm_embed_dim = config.get_esm_embedding_dim()
    model = create_multitask_classifier(
        esm_embed_dim=esm_embed_dim,
        num_subfamilies=num_subfamilies,
        num_substrates=num_substrates
    )
    print(f"[OK] Model created successfully")

    # Check architecture
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return model


def test_forward_pass(model):
    """Test that forward pass works."""
    print("\n" + "="*60)
    print("TEST 4: Forward Pass")
    print("="*60)

    # Create dummy input
    batch_size = 4
    seq_len = 100
    embed_dim = config.get_esm_embedding_dim()

    embeddings = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    print(f"  Input shape: {embeddings.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(embeddings, mask)

    # Check outputs
    print(f"\n  Output shapes:")
    for task, pred in predictions.items():
        print(f"    {task}: {pred.shape}")

    # Verify shapes
    assert predictions['binary'].shape == (batch_size, 1), "Binary output shape wrong"
    assert predictions['subfamily'].shape[0] == batch_size, "Subfamily output batch wrong"
    assert predictions['substrate'].shape[0] == batch_size, "Substrate output batch wrong"
    assert predictions['tm_count'].shape == (batch_size, 1), "TM count output shape wrong"

    print(f"\n[OK] Forward pass successful, all shapes correct")

    return predictions


def test_loss_computation(predictions, num_subfamilies, num_substrates):
    """Test that loss computation works."""
    print("\n" + "="*60)
    print("TEST 5: Loss Computation")
    print("="*60)

    batch_size = predictions['binary'].shape[0]

    # Create dummy targets
    targets = {
        'binary': torch.randint(0, 2, (batch_size,), dtype=torch.float32),
        'subfamily': torch.randint(0, num_subfamilies, (batch_size,), dtype=torch.long),
        'substrate': torch.randint(0, num_substrates, (batch_size,), dtype=torch.long),
        'tm_count': torch.randint(0, 13, (batch_size,), dtype=torch.float32)
    }

    print(f"  Target shapes:")
    for task, target in targets.items():
        print(f"    {task}: {target.shape}, dtype={target.dtype}")

    # Create loss function
    criterion = MultiTaskLoss(
        binary_weight=1.0,
        subfamily_weight=0.5,
        substrate_weight=0.3,
        tm_weight=0.2
    )

    # Compute loss
    total_loss, loss_dict = criterion(predictions, targets)

    print(f"\n  Loss values:")
    for task, loss_val in loss_dict.items():
        print(f"    {task}: {loss_val:.4f}")

    print(f"\n[OK] Loss computation successful")

    return total_loss, loss_dict


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MULTI-TASK SETUP VERIFICATION")
    print("="*80)

    try:
        # Test 1: Dataset
        if not test_dataset():
            print("\n[FAIL] Dataset test failed")
            return

        # Test 2: Label mappings
        result = test_label_mappings()
        if isinstance(result, bool) and not result:
            print("\n[FAIL] Label mapping test failed")
            return
        _, num_subfamilies, num_substrates = result

        # Test 3: Model creation
        model = test_model_creation(num_subfamilies, num_substrates)

        # Test 4: Forward pass
        predictions = test_forward_pass(model)

        # Test 5: Loss computation
        total_loss, loss_dict = test_loss_computation(predictions, num_subfamilies, num_substrates)

        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print(f"\n[OK] Dataset: Ready")
        print(f"[OK] Label mappings: {num_subfamilies} subfamilies, {num_substrates} substrates")
        print(f"[OK] Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"[OK] Forward pass: Working")
        print(f"[OK] Loss computation: Working")
        print(f"\n[SUCCESS] Ready to start multi-task training!")
        print(f"   Run: python classifier_model/esm_model/train_multitask.py")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
