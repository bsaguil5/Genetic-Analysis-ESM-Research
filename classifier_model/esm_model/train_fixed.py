
import os
import sys
from pathlib import Path

# Add the current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

# OVERRIDE CONFIGURATION
# Point to the BRANDONFIXED dataset
REPO_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = REPO_ROOT / "BRANDONFIXED" / "labeled_sequences.csv"

config.DATA_CSV_PATH = str(DATASET_PATH)

print("="*60)
print("TRAINING WITH FIXED DATASET (BRANDONFIXED)")
print(f"Dataset: {config.DATA_CSV_PATH}")
print("="*60)

# Import the main training function
from train import main

if __name__ == "__main__":
    main()
