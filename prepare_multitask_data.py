"""
Prepare multi-task labels from existing dataset.

Auto-extracts:
1. Binary label (transporter vs non-transporter) - already have
2. Subfamily (ABC, SWEET, NRT, MATE, etc.)
3. Substrate type (sugar, nitrate, metal, etc.)
4. Approximate TM domain count (from subfamily)

Author: Brandon & Claude
Date: December 24, 2025
"""

import pandas as pd
import re
from collections import Counter

# Subfamily mapping based on protein name keywords
SUBFAMILY_KEYWORDS = {
    'ABC': ['abc transporter', 'abcb', 'abcc', 'abcd', 'abcg', 'mdr', 'multidrug'],
    'SWEET': ['sweet', 'sug', 'sugar transport', 'sucrose transport'],
    'SUT': ['sut', 'sucrose transporter', 'sucrose-proton'],
    'NRT': ['nrt', 'nitrate transporter', 'nitrate transport', 'nitrogen transport'],
    'AMT': ['amt', 'ammonium transporter', 'ammonium transport'],
    'MATE': ['mate', 'aluminum', 'citrate transporter'],
    'Aquaporin': ['aquaporin', 'pip', 'tip', 'nip', 'sip', 'water channel'],
    'Cation': ['potassium transporter', 'calcium transporter', 'magnesium transporter',
               'cation transporter', 'metal transporter'],
    'Anion': ['sulfate transporter', 'phosphate transporter', 'chloride transporter',
              'anion transporter'],
    'MST': ['mst', 'monosaccharide transporter'],
    'NPF': ['npf', 'peptide transporter', 'oligopeptide'],
}

# Substrate type keywords
SUBSTRATE_KEYWORDS = {
    'Sugar': ['sugar', 'sucrose', 'glucose', 'fructose', 'monosaccharide', 'sweet', 'hexose'],
    'Nitrogen': ['nitrate', 'ammonium', 'nitrogen', 'urea'],
    'Metal': ['aluminum', 'zinc', 'iron', 'copper', 'metal', 'cation'],
    'Anion': ['sulfate', 'phosphate', 'chloride', 'anion'],
    'Water': ['aquaporin', 'water'],
    'Organic': ['citrate', 'malate', 'organic', 'peptide', 'amino acid'],
}

# Approximate TM domain counts by subfamily (from literature)
TM_DOMAIN_COUNTS = {
    'ABC': 12,  # ABC transporters: 6 TM per half, 12 total
    'SWEET': 7,  # SWEET family: 7 TM helices
    'SUT': 12,  # Sucrose transporters: ~12 TM
    'NRT': 12,  # Nitrate transporters: ~12 TM
    'AMT': 11,  # Ammonium transporters: ~11 TM
    'MATE': 12,  # MATE family: ~12 TM
    'Aquaporin': 6,  # Aquaporins: 6 TM
    'Cation': 10,  # Variable, approximate
    'Anion': 12,  # Variable, approximate
    'MST': 12,  # Monosaccharide transporters
    'NPF': 12,  # NPF family
    'Unknown': 0,  # Non-transporters
}


def extract_subfamily(protein_name):
    """Extract subfamily from protein name using keyword matching."""
    name_lower = protein_name.lower()

    for subfamily, keywords in SUBFAMILY_KEYWORDS.items():
        if any(keyword in name_lower for keyword in keywords):
            return subfamily

    return 'Unknown'


def extract_substrate(protein_name):
    """Extract substrate type from protein name."""
    name_lower = protein_name.lower()

    for substrate, keywords in SUBSTRATE_KEYWORDS.items():
        if any(keyword in name_lower for keyword in keywords):
            return substrate

    return 'Unknown'


def get_tm_count(subfamily):
    """Get approximate TM domain count for subfamily."""
    return TM_DOMAIN_COUNTS.get(subfamily, 0)


def prepare_multitask_dataset(input_csv, output_csv):
    """
    Prepare multi-task dataset from existing labeled data.

    Args:
        input_csv: Path to cleaned_labeled_sequences.csv
        output_csv: Path to save multitask dataset
    """
    print("="*80)
    print("PREPARING MULTI-TASK DATASET")
    print("="*80)

    # Load data
    print(f"\n[1] Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"    Loaded {len(df)} sequences")

    # Extract multi-task labels
    print("\n[2] Extracting multi-task labels...")

    df['Subfamily'] = df['Protein Name'].apply(extract_subfamily)
    df['Substrate'] = df['Protein Name'].apply(extract_substrate)
    df['TM_Domains'] = df['Subfamily'].apply(get_tm_count)

    # Analysis
    print("\n[3] Dataset Analysis:")
    print(f"\n    Binary Labels:")
    print(f"    {df['Label'].value_counts().to_dict()}")

    print(f"\n    Subfamily Distribution:")
    subfamily_counts = df['Subfamily'].value_counts()
    for subfamily, count in subfamily_counts.items():
        print(f"      {subfamily:15} {count:4} sequences")

    print(f"\n    Substrate Distribution:")
    substrate_counts = df['Substrate'].value_counts()
    for substrate, count in substrate_counts.items():
        print(f"      {substrate:15} {count:4} sequences")

    print(f"\n    TM Domain Distribution:")
    tm_counts = df['TM_Domains'].value_counts().sort_index()
    for tm, count in tm_counts.items():
        print(f"      {tm:2} TM domains: {count:4} sequences")

    # Check for transporters with labels
    transporters = df[df['Label'] == 0]
    print(f"\n[4] Transporter Subfamily Breakdown:")
    trans_subfamilies = transporters['Subfamily'].value_counts()
    for subfamily, count in trans_subfamilies.items():
        print(f"      {subfamily:15} {count:4} sequences")

    # Save
    print(f"\n[5] Saving to {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"    Saved {len(df)} sequences with multi-task labels")

    # Create label mappings for training
    print("\n[6] Creating label mappings...")

    subfamily_to_idx = {name: idx for idx, name in enumerate(sorted(df['Subfamily'].unique()))}
    substrate_to_idx = {name: idx for idx, name in enumerate(sorted(df['Substrate'].unique()))}

    print(f"\n    Subfamily classes ({len(subfamily_to_idx)}):")
    for name, idx in sorted(subfamily_to_idx.items(), key=lambda x: x[1]):
        print(f"      {idx:2} -> {name}")

    print(f"\n    Substrate classes ({len(substrate_to_idx)}):")
    for name, idx in sorted(substrate_to_idx.items(), key=lambda x: x[1]):
        print(f"      {idx:2} -> {name}")

    # Save mappings
    import json
    mappings = {
        'subfamily_to_idx': subfamily_to_idx,
        'idx_to_subfamily': {v: k for k, v in subfamily_to_idx.items()},
        'substrate_to_idx': substrate_to_idx,
        'idx_to_substrate': {v: k for k, v in substrate_to_idx.items()},
    }

    mapping_file = 'data/multitask_label_mappings.json'
    with open(mapping_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"\n[7] Saved label mappings to {mapping_file}")

    print("\n" + "="*80)
    print("MULTI-TASK DATASET PREPARATION COMPLETE")
    print("="*80)

    return df, mappings


if __name__ == "__main__":
    input_file = "data/cleaned_labeled_sequences.csv"
    output_file = "data/multitask_labeled_sequences.csv"

    df, mappings = prepare_multitask_dataset(input_file, output_file)

    print("\n✅ Ready for multi-task training!")
    print(f"   Dataset: {output_file}")
    print(f"   Mappings: data/multitask_label_mappings.json")
