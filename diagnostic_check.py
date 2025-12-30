"""
Diagnostic Investigation: Why Multi-Task Model Fails on Breeding Targets

Investigates:
1. Are breeding targets in the training data?
2. What are their labels in the dataset?
3. Training/validation split analysis
4. Label distribution analysis

Author: Brandon & Claude
Date: December 24, 2025
"""

import pandas as pd

# Breeding target IDs
BREEDING_IDS = [
    "XP_002465781.1",  # SUT1 (Sugar)
    "XP_002455791.2",  # NRT2.4 (Nitrate)
    "ABS89149.1",      # SbMATE (Aluminum)
    "XP_021312865.1"   # AKT1 (Control)
]

print("\n" + "="*80)
print("DIAGNOSTIC INVESTIGATION: Multi-Task Model Failure Analysis")
print("="*80)

# Investigation 1: Check if breeding targets are in the datasets
print("\n[1] CHECKING IF BREEDING TARGETS ARE IN DATASETS")
print("-"*80)

datasets = {
    'Cleaned': 'data/cleaned_labeled_sequences.csv',
    'Multi-task': 'data/multitask_labeled_sequences.csv'
}

for dataset_name, dataset_path in datasets.items():
    try:
        df = pd.read_csv(dataset_path)
        print(f"\n{dataset_name} Dataset:")
        print(f"  Total sequences: {len(df)}")

        for target_id in BREEDING_IDS:
            base_id = target_id.split('.')[0]
            match = df[df['Accession ID'].str.contains(base_id, na=False, regex=False)]

            if len(match) > 0:
                row = match.iloc[0]
                label_str = 'TRANSPORTER' if row['Label'] == 0 else 'NON-TRANSPORTER'
                print(f"  [FOUND] {target_id}")
                print(f"    Label: {row['Label']} ({label_str})")
                if 'Subfamily' in row:
                    print(f"    Subfamily: {row.get('Subfamily', 'N/A')}")
            else:
                print(f"  [NOT FOUND] {target_id}")
    except Exception as e:
        print(f"\n{dataset_name} Dataset: ERROR - {e}")

# Investigation 2: Label distribution
print("\n\n[2] MULTI-TASK DATASET LABEL DISTRIBUTION")
print("-"*80)

try:
    df = pd.read_csv('data/multitask_labeled_sequences.csv')

    print(f"\nBinary Labels:")
    print(f"  Label 0 (TRANSPORTER): {(df['Label'] == 0).sum()}")
    print(f"  Label 1 (NON-TRANSPORTER): {(df['Label'] == 1).sum()}")

    print(f"\nSubfamily Distribution (top 5):")
    for subfamily, count in df['Subfamily'].value_counts().head(5).items():
        pct = count / len(df) * 100
        print(f"  {subfamily}: {count} ({pct:.1f}%)")

    unknown_pct = (df['Subfamily'] == 'Unknown').sum() / len(df) * 100
    if unknown_pct > 50:
        print(f"\n[CRITICAL] {unknown_pct:.1f}% labeled as 'Unknown' subfamily!")
        print(f"  Model may learn to predict 'Unknown' + NON-TRANSPORTER by default")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
