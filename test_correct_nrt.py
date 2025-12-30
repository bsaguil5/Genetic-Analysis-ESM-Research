"""
Quick test of hybrid classifier on CORRECT Sorghum nitrate transporter.

Tests XP_002455791.2 - confirmed high-affinity nitrate transporter 2.4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'classifier_model', 'esm_model'))

from hybrid_classifier import HybridClassifier
from Bio import Entrez, SeqIO

Entrez.email = "bsaguil@udallas.edu"

# Test the CORRECT nitrate transporter
print("="*80)
print("TESTING HYBRID CLASSIFIER ON CORRECT SORGHUM NITRATE TRANSPORTER")
print("="*80)

classifier = HybridClassifier(use_blast=True)

# The CORRECT Sorghum nitrate transporter
target = {
    'id': 'XP_002455791.2',
    'name': 'High-Affinity Nitrate Transporter 2.4 (NRT2.4)',
    'expected': 0  # Should be TRANSPORTER
}

print(f"\nFetching: {target['name']} ({target['id']})")

# Fetch sequence
handle = Entrez.efetch(db="protein", id=target['id'], rettype="fasta", retmode="text")
record = SeqIO.read(handle, "fasta")
sequence = str(record.seq)
description = record.description
handle.close()

print(f"Sequence length: {len(sequence)} amino acids")
print(f"Description: {description}\n")

# Predict
result = classifier.predict(sequence, description, target['id'], use_hybrid=True)

# Evaluate
passed = (result['prediction'] == target['expected'])

print(f"\n{'='*80}")
print(f"FINAL RESULT: {'✅ PASS - Correctly identified as TRANSPORTER!' if passed else '❌ FAIL - Incorrectly classified'}")
print(f"{'='*80}")

if passed:
    print(f"\n🎉 SUCCESS! The hybrid classifier correctly identified this nitrate transporter!")
    print(f"   Method used: {result['final_method']}")
    print(f"   Confidence: {result['confidence']:.2%}")

    # Show which component got it right
    if 'ml_result' in result and result['ml_result']:
        ml_pred = "TRANSPORTER" if result['ml_result']['prediction'] == 0 else "NON-TRANSPORTER"
        print(f"\n   ML prediction: {ml_pred} (confidence: {result['ml_result']['confidence']:.2%})")

    if 'blast_result' in result and result['blast_result']:
        blast_pred = "TRANSPORTER" if result['blast_result']['prediction'] == 0 else "NON-TRANSPORTER"
        print(f"   BLAST prediction: {blast_pred} (family: {result['blast_result']['family']})")
else:
    print(f"\n⚠️ The hybrid classifier failed to identify this nitrate transporter.")
    print(f"   This suggests Option A (BLAST hybrid) may not be sufficient.")
