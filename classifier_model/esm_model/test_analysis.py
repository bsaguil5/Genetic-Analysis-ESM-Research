
import os
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import utils
from esm_embedder import ESMEmbedder
from esm_classifier import ESMClassifier

# CONFIGURATION
# Point to the BRANDONFIXED dataset
REPO_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = REPO_ROOT / "BRANDONFIXED" / "labeled_sequences.csv"

def find_best_model():
    """Find the best trained model."""
    model_files = [f for f in os.listdir('.') if f.startswith('esm_model_BEST_f1_') and f.endswith('.pt')]
    
    if not model_files and os.path.exists('checkpoints'):
        checkpoint_files = [os.path.join('checkpoints', f) for f in os.listdir('checkpoints') 
                          if f.startswith('esm_model_BEST_f1_') and f.endswith('.pt')]
        model_files.extend(checkpoint_files)
    
    if not model_files:
        raise FileNotFoundError("No trained model found!")
    
    # Sort by F1
    def extract_f1(filename):
        basename = os.path.basename(filename)
        return float(basename.split('_f1_')[1].replace('.pt', ''))
    
    model_files.sort(key=extract_f1, reverse=True)
    return model_files[0]

def load_model(model_path, device):
    print(f"📂 Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer embedding dim
    if 'model_config' in checkpoint:
        esm_embed_dim = checkpoint['model_config'].get('esm_embed_dim', 480)
    else:
        esm_embed_dim = 480 # Fallback
        
    model = ESMClassifier(esm_embed_dim=esm_embed_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

import argparse

def analyze_errors(model_path=None):
    print("="*60)
    print("🔬 DETAILED ERROR ANALYSIS (BRANDONFIXED)")
    print("="*60)
    
    # 1. Load Data with Metadata
    print(f"📂 Loading data from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"   Loaded {len(df)} sequences")
    
    # Clean data similar to utils.load_protein_data but keep metadata
    # We strictly need 'Label' and 'Sequence'
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
    df = df.dropna(subset=['Label', 'Sequence'])
    df['Label'] = df['Label'].astype(int)
    
    sequences = df['Sequence'].tolist()
    labels = df['Label'].tolist()
    
    # 2. Setup Model & Embedder
    device = config.get_device()
    embedder = ESMEmbedder()
    
    # Use provided path or find best
    final_model_path = model_path or find_best_model()
    model = load_model(final_model_path, device)
    
    
    # 3. Compute Embeddings (use cache!)
    print("\n🧬 Computing/Loading Embeddings...")
    embeddings, masks = embedder.embed_sequences(sequences, max_length=config.MAX_SEQUENCE_LENGTH)
    
    # 4. Predict
    print("🔮 Running Predictions...")
    data_loader = utils.create_data_loader(
        embeddings, masks, torch.tensor(labels, dtype=torch.float32),
        batch_size=config.BATCH_SIZE, shuffle=False
    )
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for emb, mask, _ in data_loader:
            emb, mask = emb.to(device), mask.to(device)
            outputs = model(emb, mask)
            probs = torch.sigmoid(outputs.squeeze())
            preds = probs > config.PREDICTION_THRESHOLD
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    df['Predicted'] = [int(p) for p in all_preds]
    df['Probability'] = all_probs
    
    # 5. Analysis
    y_true = df['Label'].values
    y_pred = df['Predicted'].values
    
    # Metrics
    print("\n" + "="*60)
    print("📊 CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=['Transporter (0)', 'Non-Transporter (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n🔢 CONFUSION MATRIX:")
    print(f"                   Predicted 0 (Trans)   Predicted 1 (Non-Trans)")
    print(f"Actual 0 (Trans)        {cm[0,0]:<20}  {cm[0,1]:<20}")
    print(f"Actual 1 (Non-Trans)    {cm[1,0]:<20}  {cm[1,1]:<20}")
    
    # False Positives: Actual 0 (Transporter), Predicted 1 (Non-Transporter)
    fp_df = df[(df['Label'] == 0) & (df['Predicted'] == 1)].sort_values(by='Probability', ascending=False)
    
    # FALSE NEGATIVES (for Label 1): Actual 1 (Non-Transporter) but model said 0 (Transporter)
    fn_df = df[(df['Label'] == 1) & (df['Predicted'] == 0)].sort_values(by='Probability', ascending=True)
    
    print("\n" + "="*60)
    print("🚨 ERROR ANALYSIS")
    print("="*60)
    
    # Print Top 5 False Positives 
    print("\n🔴 TOP 5 FALSE POSITIVES (Transporters misclassified as Non-Transporters)")
    print("(The model thinks these are 'Good Boys' / Non-Transporters, but they are Transporters)")
    print("-" * 80)
    if not fp_df.empty:
        for i, row in fp_df.head(5).iterrows():
            desc = row.get('Protein Name', row.get('Description', 'No Description'))
            print(f"[{row['Probability']:.4f}] {desc}")
    else:
        print("None! (Perfect recall for Transporters?)")

    # Print Top 5 False Negatives
    print("\n🔵 TOP 5 FALSE NEGATIVES (Non-Transporters misclassified as Transporters)")
    print("(The model thinks these are 'Bad Boys' / Transporters, but they are Non-Transporters)")
    print("-" * 80)
    if not fn_df.empty:
        for i, row in fn_df.head(5).iterrows():
            desc = row.get('Protein Name', row.get('Description', 'No Description'))
            print(f"[{row['Probability']:.4f}] {desc}")
    else:
        print("None! (Perfect precision for Transporters?)")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model errors detailed')
    parser.add_argument('--model_path', type=str, help='Path to specific model checkpoint')
    args = parser.parse_args()
    
    analyze_errors(args.model_path)
