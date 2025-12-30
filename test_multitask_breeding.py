"""
Test Multi-Task Model on Breeding Targets

Validates the multi-task model on critical breeding proteins for Sorghum bicolor:
- Sugar transport (SUT1)
- Nitrogen uptake (NRT2.4)
- Aluminum tolerance (SbMATE)
- Control (AKT1 potassium channel)

Author: Brandon & Claude
Date: December 24, 2025
"""

import os
import sys
import torch
import json

# Add classifier path - use relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'classifier_model', 'esm_model'))

from esm_embedder import ESMEmbedder
from esm_classifier_multitask import create_multitask_classifier
import config

# Breeding target proteins (CORRECTED IDs)
BREEDING_TARGETS = [
    {
        "id": "XP_002465781.1",
        "name": "Sucrose Transporter (SUT1)",
        "expected": "TRANSPORTER",
        "subfamily": "SWEET",
        "substrate": "Sugar"
    },
    {
        "id": "XP_002455791.2",
        "name": "High-Affinity Nitrate Transporter 2.4 (NRT2.4)",
        "expected": "TRANSPORTER",
        "subfamily": "NRT",
        "substrate": "Nitrogen"
    },
    {
        "id": "ABS89149.1",
        "name": "SbMATE (Aluminum Tolerance)",
        "expected": "TRANSPORTER",
        "subfamily": "MATE",
        "substrate": "Metal"
    },
    {
        "id": "XP_021312865.1",
        "name": "AKT1 (Potassium Channel)",
        "expected": "NON-TRANSPORTER",
        "subfamily": "Cation",
        "substrate": "Metal"
    }
]

# Load label mappings
label_mappings_path = os.path.join(script_dir, 'data', 'multitask_label_mappings.json')
with open(label_mappings_path, 'r') as f:
    label_mappings = json.load(f)

idx_to_subfamily = label_mappings['idx_to_subfamily']
idx_to_substrate = label_mappings['idx_to_substrate']


def fetch_sequence(accession_id):
    """Fetch protein sequence from NCBI."""
    from Bio import Entrez, SeqIO
    import time

    Entrez.email = "bsaguil@udallas.edu"

    try:
        print(f"  [INFO] Fetching from NCBI...", end='', flush=True)
        handle = Entrez.efetch(db="protein", id=accession_id, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        sequence = str(record.seq).strip().upper()
        print(f" Done!")
        time.sleep(0.5)  # Be nice to NCBI
        return sequence
    except Exception as e:
        print(f" Failed: {e}")
        return None


def test_multitask_model():
    """Test multi-task model on breeding targets."""
    print("\n" + "="*80)
    print("MULTI-TASK MODEL - BREEDING TARGET VALIDATION")
    print("="*80)

    # Load best checkpoint
    checkpoint_path = os.path.join(script_dir, "checkpoints", "esm_multitask_BEST_f1_0.9907.pt")
    print(f"\n[*] Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    esm_embed_dim = config.get_esm_embedding_dim()
    num_subfamilies = len(label_mappings['subfamily_to_idx'])
    num_substrates = len(label_mappings['substrate_to_idx'])

    model = create_multitask_classifier(
        esm_embed_dim=esm_embed_dim,
        num_subfamilies=num_subfamilies,
        num_substrates=num_substrates
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"[OK] Model loaded successfully!")
    print(f"   Checkpoint F1: {checkpoint.get('f1_score', 'N/A')}")
    print(f"   Checkpoint Accuracy: {checkpoint.get('accuracy', 'N/A')}")

    # Create embedder
    embedder = ESMEmbedder()

    # Test each breeding target
    print(f"\n{'='*80}")
    print("TESTING BREEDING TARGETS")
    print(f"{'='*80}\n")

    results = []

    for target in BREEDING_TARGETS:
        print(f"\n[Target] {target['name']}")
        print(f"  ID: {target['id']}")
        print(f"  Expected: {target['expected']}")

        # Fetch sequence
        sequence = fetch_sequence(target['id'])
        if sequence is None:
            print(f"  [SKIP] Could not fetch sequence")
            continue

        print(f"  Sequence length: {len(sequence)} aa")

        # Compute embeddings
        embeddings, mask = embedder.embed_sequences([sequence], use_cache=False)

        # Run inference
        with torch.no_grad():
            predictions = model(embeddings, mask)

        # Binary prediction
        binary_prob = torch.sigmoid(predictions['binary']).item()
        binary_pred = "TRANSPORTER" if binary_prob > 0.5 else "NON-TRANSPORTER"

        # Subfamily prediction
        subfamily_idx = torch.argmax(predictions['subfamily'], dim=1).item()
        subfamily_pred = idx_to_subfamily[str(subfamily_idx)]
        subfamily_conf = torch.softmax(predictions['subfamily'], dim=1)[0, subfamily_idx].item()

        # Substrate prediction
        substrate_idx = torch.argmax(predictions['substrate'], dim=1).item()
        substrate_pred = idx_to_substrate[str(substrate_idx)]
        substrate_conf = torch.softmax(predictions['substrate'], dim=1)[0, substrate_idx].item()

        # TM domain prediction
        tm_count = predictions['tm_count'].item()

        # Display results
        print(f"\n  [PREDICTIONS]")
        print(f"    Binary:     {binary_pred} ({binary_prob*100:.2f}% confidence)")
        print(f"    Subfamily:  {subfamily_pred} ({subfamily_conf*100:.2f}% confidence)")
        print(f"    Substrate:  {substrate_pred} ({substrate_conf*100:.2f}% confidence)")
        print(f"    TM Domains: {tm_count:.1f}")

        # Check correctness
        binary_correct = (binary_pred == target['expected'])

        if binary_correct:
            print(f"\n  [RESULT] PASS - Correctly classified as {target['expected']}")
        else:
            print(f"\n  [RESULT] FAIL - Expected {target['expected']}, got {binary_pred}")

        results.append({
            'name': target['name'],
            'id': target['id'],
            'expected': target['expected'],
            'predicted': binary_pred,
            'confidence': binary_prob,
            'subfamily': subfamily_pred,
            'substrate': substrate_pred,
            'tm_count': tm_count,
            'correct': binary_correct
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    passed = sum(1 for r in results if r['correct'])
    total = len(results)

    if total > 0:
        print(f"Breeding Targets Passed: {passed}/{total} ({passed/total*100:.1f}%)\n")
    else:
        print(f"No breeding targets were tested (sequences not found)\n")
        return results

    for r in results:
        status = "[PASS]" if r['correct'] else "[FAIL]"
        print(f"  {status} {r['name']}")
        print(f"         Binary: {r['predicted']} ({r['confidence']*100:.2f}%)")
        print(f"         Subfamily: {r['subfamily']}, Substrate: {r['substrate']}, TM: {r['tm_count']:.1f}")

    if passed == total:
        print(f"\n[SUCCESS] All {total} breeding targets correctly classified!")
    else:
        print(f"\n[WARNING] {total - passed} breeding target(s) misclassified")

    return results


if __name__ == "__main__":
    test_multitask_model()
