
# =============================================================================
# ACTIVE FILTER SCRIPT
# Replaces: filter_candidates.py (Archived)
# Difference: This script specifically filters the 'sorghum_discovery_results.csv'
# output from the Discovery Phase, whereas the old script was for generic candidates.
# =============================================================================

import pandas as pd
import os

INPUT_FILE = "data/sorghum_discovery_results.csv"
OUTPUT_FILE = "data/final_sorghum_candidates.csv"
SORGHUM_SOURCE_FILE = "data/sorghum_hypothetical_unknowns.csv"

def filter_sorghum():
    print("💎 STARTING SORGHUM GOLD MINING")
    print("="*60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Did the prediction step finish?")
        return

    # Load Predictions
    print(f"📂 Loading predictions from: {INPUT_FILE}")
    df_pred = pd.read_csv(INPUT_FILE)
    print(f"   Total predictions: {len(df_pred)}")
    
    # Load Source (for Descriptions)
    # The streaming prediction output usually has 'sequence_index' and 'sequence'. 
    # It does NOT store the Description from the input file.
    # We must merge with the original fetch file to get the Protein Names.
    print(f"📂 Loading original metadata from: {SORGHUM_SOURCE_FILE}")
    if os.path.exists(SORGHUM_SOURCE_FILE):
        df_source = pd.read_csv(SORGHUM_SOURCE_FILE)
        # We assume the order is preserved or we can merge on Sequence
        # Merging on Sequence is safest
        # Standardize column name in predictions to match source
        if 'sequence' in df_pred.columns:
            df_pred.rename(columns={'sequence': 'Sequence'}, inplace=True)

        df_merged = pd.merge(df_pred, df_source[['Sequence', 'Description', 'ID']], on='Sequence', how='left')
    else:
        print("⚠️ Warning: Source file missing. Descriptions might be lost.")
        df_merged = df_pred
        df_merged['Description'] = "Unknown"
        df_merged['ID'] = "Unknown"

    # Filter Logic
    # 1. Label = 0 (Transporter)
    # 2. Confidence > 0.95
    
    print("\n🔍 Filtering for Platinum Candidates (>95% Confidence)...")
    
    # Ensure numeric
    df_merged['predicted_label'] = df_merged['predicted_label'].astype(int)
    
    # Filter
    platinum = df_merged[
        (df_merged['predicted_label'] == 0) & 
        (df_merged['confidence'] > 0.95)
    ].copy()
    
    count = len(platinum)
    print(f"✨ Found {count} Sorghum Platinum Candidates!")
    
    if count > 0:
        # Sort by confidence
        platinum = platinum.sort_values('confidence', ascending=False)
        
        print("\n🏆 TOP 5 SORGHUM CANDIDATES")
        print("="*80)
        for i, row in platinum.head(5).iterrows():
            desc = row.get('Description', 'N/A')
            print(f"ID: {row.get('ID', 'N/A')}")
            print(f"Description: {desc}")
            print(f"Confidence: {row['confidence']:.4f}")
            print(f"Sequence: {row['Sequence'][:30]}...")
            print("-" * 80)
            
        # Save
        platinum.to_csv(OUTPUT_FILE, index=False)
        print(f"\n💾 Saved {count} candidates to: {OUTPUT_FILE}")
    else:
        print("😔 No candidates met the strict criteria.")

if __name__ == "__main__":
    filter_sorghum()
