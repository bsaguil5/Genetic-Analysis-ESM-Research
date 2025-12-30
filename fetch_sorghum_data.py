
import os
import pandas as pd
from Bio import Entrez, SeqIO
from tqdm import tqdm
import time

# Configuration
Entrez.email = "bsaguil@udallas.edu"
TRAINING_FILE = "data/raw_sources/labeled_sequences_original.csv"
OUTPUT_FILE = "data/sorghum_hypothetical_unknowns.csv"
TARGET_COUNT = 2000
SEARCH_TERM = "Sorghum bicolor[Organism] AND hypothetical protein[Title]"

def fetch_sorghum_data():
    print(f"🌾 STARTING SORGHUM DISCOVERY FETCH")
    print("="*60)
    
    # 1. Load Training Data for Exclusion
    print(f"📂 Loading training data to build exclusion set...")
    exclusion_sequences = set()
    if os.path.exists(TRAINING_FILE):
        df_train = pd.read_csv(TRAINING_FILE)
        # Assuming column is 'Sequence' or similar. utils.py logic handled this.
        # Let's check common names
        seq_col = None
        for col in ['Sequence', 'sequence', 'seq']:
            if col in df_train.columns:
                seq_col = col
                break
        
        if seq_col:
            for seq in df_train[seq_col].dropna():
                exclusion_sequences.add(str(seq).strip().upper())
            print(f"   Loaded {len(exclusion_sequences)} training sequences to exclude.")
        else:
            print("⚠️ Warning: Could not find sequence column in training file. Exclusion might fail.")
    else:
        print("⚠️ Warning: Training file not found. No exclusion will be performed.")

    # 2. Search NCBI
    print(f"\n🔍 Searching NCBI for: '{SEARCH_TERM}'")
    try:
        handle = Entrez.esearch(db="protein", term=SEARCH_TERM, retmax=TARGET_COUNT + 500) # Fetch extra to account for duplicates/excluded
        record = Entrez.read(handle)
        id_list = record["IdList"]
        print(f"   Found {len(id_list)} hits.")
        if not id_list:
            print("❌ No IDs found. Try a broader search term.")
            return
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return

    # 3. Fetch Details
    print(f"\n📥 Fetching sequences (Target: {TARGET_COUNT})...")
    
    fetched_data = []
    batch_size = 100
    
    # Process in batches
    for i in tqdm(range(0, len(id_list), batch_size), desc="Fetching"):
        if len(fetched_data) >= TARGET_COUNT:
            break
            
        batch_ids = id_list[i:i+batch_size]
        try:
            handle = Entrez.efetch(db="protein", id=batch_ids, rettype="fasta", retmode="text")
            records = list(SeqIO.parse(handle, "fasta"))
            
            for r in records:
                seq = str(r.seq).strip().upper()
                
                # Exclusion Check
                if seq in exclusion_sequences:
                    continue # Skip training data
                
                # Duplicate Check (internal)
                if any(d['Sequence'] == seq for d in fetched_data):
                    continue
                
                fetched_data.append({
                    'ID': r.id,
                    'Description': r.description,
                    'Sequence': seq,
                    'Length': len(seq)
                })
                
                if len(fetched_data) >= TARGET_COUNT:
                    break
                    
            time.sleep(0.5) # Be nice to NCBI servers
            
        except Exception as e:
            print(f"⚠️ Error fetching batch {i}: {e}")
            continue

    # 4. Save Results
    print(f"\n💾 Saving results...")
    if fetched_data:
        df = pd.DataFrame(fetched_data)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Successfully saved {len(df)} unique Sorghum candidates to: {OUTPUT_FILE}")
        print(f"   (Excluded {len(id_list) - len(df)} duplicates/training samples)")
    else:
        print("❌ No data collected!")

if __name__ == "__main__":
    fetch_sorghum_data()
