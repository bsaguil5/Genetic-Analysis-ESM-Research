# 🧠 ESM Classifier Model

This directory contains the core logic for the protein classifier.

## 🏗️ Architecture
The model uses **ESM-2 (Evolutionary Scale Modeling)** as a feature extractor, followed by a custom classification head.
*   **Embedder**: `esm_embedder.py` (Handles tokenization and generating embeddings from the pre-trained ESM-2 model).
*   **Classifier**: `esm_classifier.py` (Neural network head: Linear -> ReLU -> Dropout -> Linear).

## ⚙️ Configuration
*   **`config.py`**: Controls all hyperparameters (Learning Rate, Batch Size, Model Size `esm2_t6_8M_UR50D`, Paths).
    *   *Note*: Paths in `config.py` are defaults; `clean_and_retrain.py` often overrides them dynamically.

## 🚀 Key Scripts
*   **`train.py`**: The training loop. Uses `ESMTrainer` class.
*   **`test.py`**: Inference loop. Uses `ESMTester` class. Supports streaming for large datasets (`--predict_file`).
*   **`utils.py`**: Data loading, cleaning, and metric calculation.

## 📦 Checkpoints
Trained models are saved to `../checkpoints/`.
The best current model is always named: `esm_model_FINAL_CLEAN.pt`.
*   **Status**: Trained on Clean Data (Dec 21, 2025).
*   **Performance**:
    - Training F1: 0.9969
    - Blind Test Accuracy: 96.55%
    - Blind Test F1: 0.9558
    - AUC: 0.9934
    - Kinase Rejection: 100% (119/119)
    - Transcription Factor Rejection: 100% (5/5)
*   **Validation**: Tested on 319 fresh sequences with zero false positives on hard negatives.
