"""
Hybrid Protein Classifier: ESM-2 ML + BLAST Homology Ensemble

This combines the strengths of two approaches:
1. ESM-2 Deep Learning: Excellent for well-represented transporter families (sugar, ABC, aluminum)
2. BLAST Homology: Catches underrepresented families (nitrate, ammonium) via sequence similarity

Architecture:
- Primary: ESM model (0.9969) for high-confidence predictions
- Fallback: BLAST against known transporter families for edge cases
- Ensemble: Intelligent voting system that knows when to trust each method

Author: Brandon & Claude
Date: December 2025
"""

import os
import sys
import torch
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
import time
from pathlib import Path

# Add model path
sys.path.insert(0, os.path.join(os.getcwd(), 'classifier_model', 'esm_model'))

from esm_classifier import ESMClassifier
from esm_embedder import ESMEmbedder
import config

# Configuration
Entrez.email = "bsaguil@udallas.edu"
MODEL_PATH = r"C:\genetic_Analysis_Research-master\checkpoints\esm_model_BEST_f1_0.9969.pt"

# Ensemble Thresholds
ML_HIGH_CONFIDENCE = 0.85  # Trust ML completely above this
ML_LOW_CONFIDENCE = 0.60   # Below this, consult BLAST
BLAST_E_VALUE_CUTOFF = 1e-20  # Strong homology threshold

# Known transporter families (for BLAST reference)
KNOWN_TRANSPORTER_FAMILIES = {
    'nitrate': ['NRT1', 'NRT2', 'NPF', 'NIA'],  # Nitrate transporters
    'ammonium': ['AMT', 'MEP', 'Rh'],  # Ammonium transporters
    'sugar': ['SUT', 'SWEET', 'STP', 'MST'],  # Sugar transporters
    'aluminum': ['MATE', 'ALS', 'ALMT'],  # Aluminum tolerance
    'aquaporin': ['PIP', 'TIP', 'NIP', 'SIP'],  # Water channels
    'abc': ['ABC', 'MDR', 'ABCB', 'ABCC'],  # ABC transporters
}

class HybridClassifier:
    """Ensemble classifier combining ESM-2 ML with BLAST homology search."""

    def __init__(self, model_path=MODEL_PATH, use_blast=True):
        """
        Initialize the hybrid classifier.

        Args:
            model_path: Path to trained ESM model checkpoint
            use_blast: Enable BLAST fallback (disable for speed testing)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_blast = use_blast

        print(f"[*] Initializing Hybrid Classifier")
        print(f"    Device: {self.device}")
        print(f"    BLAST enabled: {self.use_blast}")

        # Load ML model
        print(f"[*] Loading ESM model from {model_path}")
        self.embedder = ESMEmbedder()

        # Load model architecture
        esm_embed_dim = config.get_esm_embedding_dim()
        self.model = ESMClassifier(
            esm_embed_dim=esm_embed_dim,
            hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
            dropout=config.CLASSIFIER_DROPOUT,
            pooling_strategy=config.POOLING_STRATEGY
        ).to(self.device)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"[*] Model loaded successfully")

    def predict_ml(self, sequence):
        """
        Get ML prediction from ESM model.

        Args:
            sequence: Protein sequence string

        Returns:
            dict with 'prediction' (0=transporter, 1=non-transporter), 'confidence', 'probability'
        """
        with torch.no_grad():
            # Get embedding
            embeddings, masks = self.embedder.embed_sequences([sequence], use_cache=False)
            embeddings = embeddings.to(self.device)
            masks = masks.to(self.device)

            # Predict
            logits = self.model(embeddings, masks)
            prob = torch.sigmoid(logits).item()

            # Convert to binary prediction
            prediction = 1 if prob > config.PREDICTION_THRESHOLD else 0
            confidence = prob if prediction == 1 else (1 - prob)

            return {
                'prediction': prediction,
                'confidence': confidence,
                'probability': prob,
                'method': 'ML'
            }

    def predict_blast(self, sequence, description=""):
        """
        Get BLAST homology prediction.

        Args:
            sequence: Protein sequence string
            description: Optional protein description to help target search

        Returns:
            dict with 'prediction', 'confidence', 'family', 'e_value'
        """
        if not self.use_blast:
            return None

        print(f"[*] Running BLAST search (this may take 30-60 seconds)...")

        try:
            # Run BLAST against nr database
            result_handle = NCBIWWW.qblast(
                program="blastp",
                database="nr",
                sequence=sequence,
                hitlist_size=10,  # Top 10 hits
                expect=1e-10,  # Only strong matches
                entrez_query="Viridiplantae[Organism]"  # Plants only
            )

            blast_records = NCBIXML.parse(result_handle)
            record = next(blast_records)

            # Analyze hits
            best_transporter_hit = None
            best_e_value = 1.0
            detected_family = "unknown"

            for alignment in record.alignments:
                for hsp in alignment.hsps:
                    hit_description = alignment.title.lower()

                    # Check if it matches known transporter families
                    for family_name, keywords in KNOWN_TRANSPORTER_FAMILIES.items():
                        if any(keyword.lower() in hit_description for keyword in keywords):
                            if hsp.expect < best_e_value:
                                best_e_value = hsp.expect
                                detected_family = family_name
                                best_transporter_hit = {
                                    'title': alignment.title,
                                    'e_value': hsp.expect,
                                    'identity': hsp.identities / hsp.align_length,
                                    'family': family_name
                                }

            # Make decision
            if best_transporter_hit and best_e_value < BLAST_E_VALUE_CUTOFF:
                prediction = 0  # Transporter
                confidence = min(0.95, -1 * (best_e_value / BLAST_E_VALUE_CUTOFF) + 1)  # Scale by e-value

                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'family': detected_family,
                    'e_value': best_e_value,
                    'identity': best_transporter_hit['identity'],
                    'method': 'BLAST'
                }
            else:
                # No strong transporter hit found
                return {
                    'prediction': 1,  # Non-transporter
                    'confidence': 0.5,  # Low confidence - absence of evidence isn't evidence of absence
                    'family': 'none',
                    'e_value': best_e_value,
                    'method': 'BLAST'
                }

        except Exception as e:
            print(f"[!] BLAST error: {e}")
            return None

    def predict_hybrid(self, sequence, description="", protein_id=""):
        """
        Ensemble prediction combining ML and BLAST.

        Decision Logic:
        1. If ML is high confidence (>0.85) -> Trust ML
        2. If ML is medium confidence (0.60-0.85) -> Check BLAST for confirmation
        3. If ML is low confidence (<0.60) -> BLAST takes priority

        Args:
            sequence: Protein sequence string
            description: Optional protein description
            protein_id: Optional protein ID for logging

        Returns:
            dict with final 'prediction', 'confidence', 'explanation', and component results
        """
        print(f"\n{'='*60}")
        print(f"[*] Hybrid Classification: {protein_id}")
        if description:
            print(f"    Description: {description[:80]}...")
        print(f"{'='*60}")

        # Step 1: Get ML prediction
        ml_result = self.predict_ml(sequence)
        print(f"\n[1] ML Prediction:")
        print(f"    Result: {'TRANSPORTER' if ml_result['prediction'] == 0 else 'NON-TRANSPORTER'}")
        print(f"    Confidence: {ml_result['confidence']:.4f}")
        print(f"    Raw probability: {ml_result['probability']:.4f}")

        # Step 2: Decide if BLAST is needed
        needs_blast = (
            ml_result['confidence'] < ML_HIGH_CONFIDENCE or  # Low/medium confidence
            'nitrate' in description.lower() or  # Known nitrate blindness
            'ammonium' in description.lower()  # Known ammonium gap
        )

        if not needs_blast:
            print(f"\n[*] ML is high confidence ({ml_result['confidence']:.4f}) - skipping BLAST")
            return {
                'prediction': ml_result['prediction'],
                'confidence': ml_result['confidence'],
                'final_method': 'ML_ONLY',
                'explanation': f"High ML confidence ({ml_result['confidence']:.2%})",
                'ml_result': ml_result,
                'blast_result': None
            }

        # Step 3: Run BLAST
        print(f"\n[2] ML confidence is medium/low or nitrate-related - consulting BLAST...")
        blast_result = self.predict_blast(sequence, description)

        if blast_result is None:
            print(f"[!] BLAST failed - falling back to ML only")
            return {
                'prediction': ml_result['prediction'],
                'confidence': ml_result['confidence'] * 0.8,  # Penalize confidence
                'final_method': 'ML_FALLBACK',
                'explanation': 'BLAST unavailable, using ML only (reduced confidence)',
                'ml_result': ml_result,
                'blast_result': None
            }

        print(f"\n[2] BLAST Prediction:")
        print(f"    Result: {'TRANSPORTER' if blast_result['prediction'] == 0 else 'NON-TRANSPORTER'}")
        print(f"    Family: {blast_result['family']}")
        print(f"    E-value: {blast_result['e_value']:.2e}")

        # Step 4: Ensemble decision
        if ml_result['prediction'] == blast_result['prediction']:
            # Agreement - boost confidence
            final_confidence = min(0.99, (ml_result['confidence'] + blast_result['confidence']) / 2 * 1.1)
            explanation = f"ML and BLAST agree ({ml_result['confidence']:.2%} + {blast_result['confidence']:.2%})"
            final_method = 'ENSEMBLE_AGREE'

        elif ml_result['confidence'] > ML_LOW_CONFIDENCE and blast_result['e_value'] > BLAST_E_VALUE_CUTOFF:
            # ML is medium confidence, BLAST is weak - trust ML
            final_confidence = ml_result['confidence'] * 0.9
            explanation = f"ML medium confidence ({ml_result['confidence']:.2%}), BLAST weak - trusting ML"
            final_method = 'ML_PRIORITY'

        elif blast_result['family'] in ['nitrate', 'ammonium'] and blast_result['e_value'] < BLAST_E_VALUE_CUTOFF:
            # Strong BLAST hit for underrepresented family - trust BLAST
            final_confidence = blast_result['confidence']
            explanation = f"Strong BLAST hit for {blast_result['family']} (E={blast_result['e_value']:.2e}) - overriding ML"
            final_method = 'BLAST_OVERRIDE'
            ml_result['prediction'] = blast_result['prediction']  # Override

        else:
            # Conflict with no clear winner - be conservative
            final_confidence = 0.5
            explanation = f"ML vs BLAST conflict - uncertain (ML: {ml_result['confidence']:.2%}, BLAST: {blast_result['confidence']:.2%})"
            final_method = 'ENSEMBLE_CONFLICT'

        print(f"\n[3] Ensemble Decision:")
        print(f"    Final: {'TRANSPORTER' if ml_result['prediction'] == 0 else 'NON-TRANSPORTER'}")
        print(f"    Confidence: {final_confidence:.4f}")
        print(f"    Method: {final_method}")
        print(f"    Explanation: {explanation}")

        return {
            'prediction': ml_result['prediction'],
            'confidence': final_confidence,
            'final_method': final_method,
            'explanation': explanation,
            'ml_result': ml_result,
            'blast_result': blast_result
        }

    def predict(self, sequence, description="", protein_id="", use_hybrid=True):
        """
        Main prediction interface.

        Args:
            sequence: Protein sequence string
            description: Optional protein description
            protein_id: Optional protein ID
            use_hybrid: If True, use ensemble; if False, ML only

        Returns:
            Prediction results dict
        """
        if use_hybrid:
            return self.predict_hybrid(sequence, description, protein_id)
        else:
            return self.predict_ml(sequence)


def test_on_breeding_targets():
    """Test the hybrid classifier on the 4 critical breeding targets."""
    print("\n" + "="*80)
    print("HYBRID CLASSIFIER TEST: BREEDING TARGETS")
    print("="*80)

    # Initialize classifier
    classifier = HybridClassifier(use_blast=True)

    # Test targets
    targets = [
        {"id": "XP_002465781.1", "name": "Sucrose Transporter (SUT1)", "expected": 0},
        {"id": "XP_002455791.2", "name": "High-Affinity Nitrate Transporter 2.4 (NRT2.4)", "expected": 0},  # CORRECTED - Real nitrate transporter
        {"id": "ABS89149.1", "name": "SbMATE (Aluminum Tolerance)", "expected": 0},
        {"id": "XP_021312865.1", "name": "AKT1 (Potassium Channel)", "expected": 1},
    ]

    results = []

    for target in targets:
        print(f"\n{'='*80}")
        print(f"Testing: {target['name']} ({target['id']})")
        print(f"Expected: {'TRANSPORTER' if target['expected'] == 0 else 'NON-TRANSPORTER'}")
        print(f"{'='*80}")

        # Fetch sequence
        try:
            handle = Entrez.efetch(db="protein", id=target['id'], rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            sequence = str(record.seq)
            description = record.description
            handle.close()

            print(f"[*] Sequence fetched: {len(sequence)} amino acids")

            # Predict
            result = classifier.predict(sequence, description, target['id'], use_hybrid=True)

            # Evaluate
            passed = (result['prediction'] == target['expected'])

            results.append({
                'target': target['name'],
                'id': target['id'],
                'expected': target['expected'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'method': result['final_method'],
                'passed': passed
            })

            print(f"\n{'='*80}")
            print(f"RESULT: {'✅ PASS' if passed else '❌ FAIL'}")
            print(f"{'='*80}")

            time.sleep(2)  # Rate limiting for NCBI

        except Exception as e:
            print(f"[!] Error testing {target['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"{r['target']}: {status}")
        print(f"  Prediction: {'TRANSPORTER' if r['prediction'] == 0 else 'NON-TRANSPORTER'} (Confidence: {r['confidence']:.2%})")
        print(f"  Method: {r['method']}\n")

    pass_count = sum(1 for r in results if r['passed'])
    print(f"\nScore: {pass_count}/{len(results)}")

    # Save report
    with open("hybrid_classifier_results.txt", "w") as f:
        f.write("HYBRID CLASSIFIER BREEDING TARGET VALIDATION\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            status = "PASS" if r['passed'] else "FAIL"
            f.write(f"{r['target']} ({r['id']}): {status}\n")
            f.write(f"  Expected: {r['expected']}, Got: {r['prediction']}\n")
            f.write(f"  Confidence: {r['confidence']:.4f}\n")
            f.write(f"  Method: {r['method']}\n\n")
        f.write(f"\nFinal Score: {pass_count}/{len(results)}\n")

    print(f"\n[*] Results saved to hybrid_classifier_results.txt")

    return results


if __name__ == "__main__":
    # Run the breeding target test
    test_on_breeding_targets()
