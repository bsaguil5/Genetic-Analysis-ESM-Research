# ESM Model

This is the main model used for the protein classifier. It uses transfer learning with Meta's ESM2 (a transformer pre-trained on 250 million protein sequences) instead of training embeddings from scratch.

## What's different from the other models?

The other models in `misc_models/` train their own embeddings from scratch on our small dataset. This ESM model uses embeddings that were already trained on millions of sequences, so it:
- Works way better with small datasets (we have 1,323 sequences in cleaned dataset)
- Trains faster (only the classifier head needs training, not the embeddings)
- Gets better accuracy (96.55% blind validation vs 64-75% for the feedforward baseline)
- Achieves 100% rejection rate on challenging kinase and transcription factor proteins
- 100% success rate on breeding targets (SUT1, NRT2.4, SbMATE, AKT1)

Think of it like using a pre-trained language model for text - the model already "knows" a lot about proteins before we even show it our data.

## How it works

```
Protein Sequence
  → ESM2 Model (frozen, pre-trained)
  → 480-dimensional embeddings
  → Attention Pooling (learns which parts of the sequence matter)
  → Small classifier network (256→128→64→1)
  → Prediction
```

Only the attention pooling and classifier are trained (~250k parameters). ESM2 stays frozen (35M parameters).

## Files

### Production (Single-Task Model)
- `train.py` - Trains the single-task binary classifier
- `test.py` - Tests it and shows metrics
- `esm_classifier.py` - The actual model architecture (single-task)
- `esm_embedder.py` - Handles ESM2 embedding generation
- `config.py` - All the hyperparameters
- `utils.py` - Helper functions

### Experimental (Multi-Task Model - NOT RECOMMENDED)
- `train_multitask.py` - Trains multi-task model (archived)
- `esm_classifier_multitask.py` - Multi-task architecture (archived)
- Note: Multi-task model achieved high validation metrics (F1=0.9907) but failed on breeding targets (1/4) due to spurious correlations. See LOGS/12-24-to-28-2025_MULTITASK_EXPERIMENT.txt for details.

## Key features

**Attention pooling**: Instead of just averaging all positions in the sequence, the model learns to pay attention to important positions (like transmembrane domains for transport proteins).

**Focal loss**: Helps with class imbalance by focusing more on hard-to-classify examples.

**Embedding cache**: ESM2 embeddings are expensive to compute, so they're cached to disk after the first run. Makes subsequent training way faster.

**Mixed precision**: Uses FP16 for speed, only ~2-3 minutes to train on GPU.

## Quick start

```bash
# Train
python train.py

# Test
python test.py

# Visualize embeddings
python visualize_embeddings.py
```

First run will be slow (~2-3 minutes) because it needs to compute ESM2 embeddings for all sequences and download the model. After that it's fast because embeddings are cached.

## Why this works so well

ESM2 was trained on 250 million protein sequences using masked language modeling (like BERT but for proteins). So it already learned:
- Evolutionary patterns
- Structural motifs
- Functional relationships
- Context-dependent representations

We just teach it the specific task of "is this a transport protein or not" which is way easier than learning everything from scratch with ~1,300 sequences.

## Performance Metrics

### Training vs Validation
- **Training F1**: 0.9969 (99.69%) - How well it fits the training data
- **Validation F1**: 0.9558 (95.58%) - How well it generalizes to new data (MORE IMPORTANT)
- **Validation Accuracy**: 96.55%
- Always report validation metrics when evaluating real-world performance

### Real-World Validation
- **Blind Test**: 319 fresh sequences (200 transporters, 119 kinases)
- **Kinase Rejection**: 100% (119/119)
- **TF Rejection**: 100% (5/5)
- **Breeding Targets**: 4/4 (100%) - SUT1, NRT2.4, SbMATE, AKT1

The breeding target validation is critical because these proteins were NOT in the training data, proving the model truly generalizes.

## Config

Main things you might want to tweak in `config.py`:
- `ESM_MODEL_NAME` - Which ESM model to use (default: 35M param version)
- `BATCH_SIZE` - Reduce if you get GPU memory errors
- `LEARNING_RATE` - Default 1e-3 works well
- `POOLING_STRATEGY` - "attention" (default), "mean", or "max"
- `LOSS_TYPE` - "focal" (default) or "bce"
