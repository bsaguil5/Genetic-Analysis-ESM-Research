"""
ESM2 Embedding Utilities

This module provides a clean interface for computing and caching ESM2 protein embeddings.
Handles model loading, embedding computation, and automatic caching for efficiency.
"""

"""
~Notes from Brandon~
1. Let's say you call embed_sequences() with 10 sequences. After the cache check,
4 sequences are cached (indices 0, 2, 5, 8), and 6 need computation (indices 1, 3, 4, 6, 7, 9)
and the ESM batch size is 4.
a) ESM's model forward pass executes 2 times on the 6 sequences that need computation
b) after _compute_embeddings() returns,  sequence_indices should just have (1,3,4,6,7,9)
c) before sorting, the first 3 indices are the first 3 in cached, (0,2,5) since you do cached + computed, not the other way around

2. tracing the shape of embeddings for sequence "MKTAY" for entire pipeline:
a) after tokenization: would be (1, 512). just 1 sequence with 512 tokens
b) after forwad pass you have (1, 512, 480). each position gets a 480 dim vector
c) after removing special tokens: (1, 510, 480). CORRECTION -> cached seq would be (5, 480) original length only
d) after final padding: (1, 512, 480)
e) after stacking with 9 other sequences: (9, 512, 480) corection -> stacking with 9 other would be 10 total
so it would be (10, 512, 480)

3. Let's say the user's GPU has 8GB memory, and processing one ESM batch of 4 seq is 2.5GB.
and you need to embed 20 sequences with batch_size=4.
a) without .cpu() after each batch, you will have used 20/4 = 5*2.5 = 12.5GB!! 
b) however, with .cpu() used after each batch, the maximum GPU memory used is just 2.5GB
c) This matters for large datasets because if you don't offload from the GPU, you are going to overload
it, and you will not be able to train or validate! The GPU will not be able to perform operations since it's out of
memory.

4. Caching logic:
a) If you ran a sequence ("MKTAY" for example), twice without clearing cache, ESM only processes it once!
b) If you changed the model name, but the cache still has embeddings from old model, there would be a compatibility bug!
c) we could fix it by saving the caches with the proper name, and we'd only use them if it is compatible with our model

5. concatenation vs stacking
a) cat does not add new dimensions. we used cat because we're just trying to put the cached and computed lists together without separatio, for example.
b) you would use stack because in the big tensor you want to keep track of all the tensors nicely in another dimension rather than having them on the same dimensional line.
c) i'm not sure how it would break if you swapped them

6. let's say you had sequence "AIVL".
a) after full processing the final shape would be (1,510) or whatever config length is
b) it should contain 510-4 = 506 true values for where we put the mask
c) it would contain 4 false values because 4 values are not masks.
d) we don't include <cls> and <eos> in the true count because those aren't part of the sequence nor mask, they are just special tokens that ESM2 requires

7. deterministic properties
let's say you did this:
  # Run 1:
  embeddings1, _ = embedder.embed_sequences(["MKTAY",
  "AIVL"])

  # Run 2 (cache cleared):
  embedder.clear_cache()
  embeddings2, _ = embedder.embed_sequences(["MKTAY",
  "AIVL"])

  # Run 3 (different order):
  embeddings3, _ = embedder.embed_sequences(["AIVL",
  "MKTAY"])
a) embeddings1 and embeddings2 would be identical. the ESM2 model is deterministic,
that is, the weights are fixed and the output given an identical input would always be the same
b) the only different between embeddings 3 and 1 is the order. but the embeddings themselves are the same across all runs.
c) in other words, the "MKTAY" embedding is identical across all 3 runs

8. edge cases
a) if you call embed_sequences() with a sequence longer than 512 amino acids, it gets cut off
b) an empty sequence would just get padded with 512 length mask.
c) a sequence with only one amino acid still goes through but gets padded with 511 length mask. 

9. sorting mystery
a) after concatenating the results the order is not sorted. it's 0 5 2 1 3
b) after sorting it would be 0 1 2 3 5
c) if you skipped hte sorting step the final output would just be out of order and probably mismatched and
i bet it would make something go wrong but you can tell me what

10. integration challenge
a) if you have 1000 training sequences the train_embeddings shape is (1000, 512, 480)
b) if config.BATCH_SIZE=32, train_loader yields 1000/32 rounded up batches.
c) the shape of embeddings in each batch is (32, 512, 480).
"""


import os
import hashlib
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import config


class ESMEmbedder:
    """
    ESM2 embedding computer with automatic caching.
    
    Features:
    - Downloads and loads ESM2 models from Hugging Face
    - Computes per-residue embeddings for protein sequences
    - Automatically caches embeddings by sequence hash
    - Handles padding/truncation to fixed length
    - GPU acceleration support
    """
    
    def __init__(self, # constructor that sets up the embedder, configures model name, cache directory and device
                 model_name: str = None,
                 cache_dir: str = None,
                 device: torch.device = None):
        """
        Initialize ESM embedder.
        
        Args:
            model_name: HuggingFace model name (e.g., 'facebook/esm2_t12_35M_UR50D')
            cache_dir: Directory to store cached embeddings
            device: Device to run model on (auto-detects if None)
        """
        self.model_name = model_name or config.ESM_MODEL_NAME
        self.cache_dir = cache_dir or config.ESM_CACHE_DIR
        self.device = device or config.get_device()
        
        # Create cache directory
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Lazy loading - model and tokenizer loaded on first use
        self._model = None
        self._tokenizer = None
        self._embedding_dim = None

        print(f"[ESM] Embedder initialized:")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Cache: {self.cache_dir if self.cache_dir else 'Disabled'}")

        # Clean cache if needed
        if self.cache_dir:
            self.cleanup_cache_on_start()
    
    @property
    def embedding_dim(self) -> int: # returns the embedding dimensions
        """Get the embedding dimension of the ESM model."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim # 480 embed_dim
    
    def _load_model(self): # downloads and loads the ESM model from HuggingFace (with lazz loading)
        """Lazy load the ESM model and tokenizer."""
        if self._model is not None:
            return
        
        print(f"[*] Loading ESM model: {self.model_name}")
        
        try:
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            
            # Move to device and set to eval mode
            self._model = self._model.to(self.device)
            self._model.eval()
            
            # Get embedding dimension
            self._embedding_dim = self._model.config.hidden_size
            
            print(f"[*] Model loaded successfully!")
            print(f"   Parameters: {sum(p.numel() for p in self._model.parameters()):,}")
            print(f"   Embedding dim: {self._embedding_dim}")
            
        except Exception as e:
            print(f"[!] Error loading ESM model: {e}")
            raise

            """
            1. a. 30/8 times but round
               b. forward... i'm not sure how to figure this one out
               c. i'm not sure
            2. 7 i think because you have the special tokens added to the 5 tokens
            3. 4. yes the embeddings should be identical. b. it would probably be faster because you 
            already cached it before. if you changed in between runs it would probably be different
            """
    
    def _sequence_hash(self, sequence: str) -> str: # used to cache the filenames which is a lot faster
        """Generate a hash for sequence caching."""
        return hashlib.sha256(sequence.encode()).hexdigest()
    
    def _get_cache_path(self, sequence: str) -> Optional[str]: # generates file path for cached embeddings
        """Get the cache file path for a sequence."""
        if not self.cache_dir:
            return None
        seq_hash = self._sequence_hash(sequence)
        return os.path.join(self.cache_dir, f"{seq_hash}.pt")
    

    # if embeddings are already on the disk
        # tries to load precomputed embeddings from the disk
    def _load_from_cache(self, sequence: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Try to load embeddings from cache."""
        cache_path = self._get_cache_path(sequence)
        
        if cache_path and os.path.exists(cache_path):
            try:
                cached_data = torch.load(cache_path, map_location='cpu', weights_only=True)
                return cached_data['embeddings'], cached_data['mask']
            except Exception as e:
                print(f"[!] Warning: Could not load cache for sequence (will recompute): {e}")
                return None
        
        return None
    

    # saves embeddings and mask to disk for future use
    # includes metadata (model name, sequence length)
    def _save_to_cache(self, sequence: str, embeddings: torch.Tensor, mask: torch.Tensor): # saves computed embeddings to disk
        """Save embeddings to cache."""
        cache_path = self._get_cache_path(sequence)
        if not cache_path:
            return
        
        try:
            torch.save({
                'embeddings': embeddings.cpu(),
                'mask': mask.cpu(),
                'model_name': self.model_name,
                'sequence_length': len(sequence)
            }, cache_path)
        except Exception as e:
            print(f"[!] Warning: Could not save to cache: {e}")
    
    # runs the ESM to get the embeddings
    # tokenizes sequenes, runs forward pass through ESM

    # this is the actual part where the sequences are ran through the ESM model to get
    # the embeddings
    # this is the actual part where the sequences are ran through the ESM model to get
    # the embeddings
    def _compute_embeddings(self, sequences: List[str], max_length: int = 512, show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ESM embeddings for a list of sequences.
        
        Args:
            sequences: List of protein sequences
            max_length: Maximum sequence length
            show_progress: Whether to show tqdm progress bar
            
        Returns:
            embeddings: Tensor of shape (num_seqs, max_length, embed_dim)
            masks: Attention masks
        """
        self._load_model() # make sure the ESM model we want is loaded 
        batch_size = config.ESM_BATCH_SIZE # Use specific ESM batch size
        
        all_embeddings = [] # collect embeddings from each batch
        all_masks = [] # will collect masks from each batch
        
        from tqdm import tqdm
        
        iter_range = range(0, len(sequences), batch_size)
        if show_progress:
            iter_range = tqdm(iter_range, desc="Computing embeddings")
            
        # Process sequences in batches
        for i in iter_range:
            batch_seqs = sequences[i:i + batch_size] # split collection of sequences into batches
            
            # Tokenize sequences
            tokenized = self._tokenizer(
                batch_seqs,
                add_special_tokens=True, # what are <cls> and <eos> tokens?
                padding='max_length',  # padd all to same length
                truncation=True, # cut off excess
                max_length=config.MAX_SEQUENCE_LENGTH, # 512
                return_tensors="pt" # return pytorch tensors
            )
            
            # Move to device
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            
            # Compute embeddings
            with torch.no_grad(): # we're not training, so no gradient needed
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state  # (batch, seq_len, embed_dim)
                # ^ ESM's final layer output... these are the embeddings from ESM
                """
                Step 3: Run ESM Model (lines 176-178)

                with torch.no_grad():  # Don't compute gradients (we're not training)
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state  # (batch, seq_len, embed_dim)

                This is the magic! The ESM model processes the sequences

                What happens:
                Input:  input_ids shape (2, 512)  # 2 sequences, 512 tokens each
                        ↓
                    ESM Model Processing
                        ↓
                Output: embeddings shape (2, 512, 480)
                        # 2 sequences, 512 positions, 480-dimensional embeddings

                last_hidden_state: The final layer's output from the transformer - these are our
                embeddings!
                """

            # Remove special tokens (CLS, SEP, etc.)
            # ESM adds <cls> at start and <eos> at end
            embeddings = embeddings[:, 1:-1, :]  # Remove first and last tokens
            attention_mask = attention_mask[:, 1:-1]  # Remove first and last tokens
            # this removes the <cls> and <eos> tokens from embeddings and masks
            
            all_embeddings.append(embeddings.cpu())
            all_masks.append(attention_mask.cpu())
            # ^ collect results and move them back to the CPU memory from GPU           

        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0) # joins tensors along a dimension
        masks = torch.cat(all_masks, dim=0)
        """
        Example:
        all_embeddings = [
            tensor (3, 510, 480),  # 3 sequences
            tensor (3, 510, 480),  # 3 sequences
            tensor (1, 510, 480)   # 1 sequence
        ]

        embeddings = torch.cat(all_embeddings, dim=0)
        # Concatenate along dimension 0 (batch dimension)

        Result: (7, 510, 480)  # All 7 sequences together!

        Visual:
        torch.cat along dim=0:
        [batch1] ─┐
        [batch2] ─┼─→ (7, 510, 480)
        [batch3] ─┘

        ---
        """
        
        return embeddings, masks
    

    # main method
    # this is what train.py calls
    # checks the cache first
    # pads/trauncates to max_length
    # returns embeddings tensor + masks tensor
    def embed_sequences(self,  
                       sequences: List[str], # all the sequences 
                       max_length: int = None, # 512 from config
                       use_cache: bool = True, # use cached embeddings?
                       batch_size: int = None,
                       show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]: # returns (embeddings, masks)
        """
        Compute embeddings for a list of sequences with caching.
        
        Args:
            sequences: List of protein sequences
            max_length: Maximum sequence length (for padding/truncation)
            use_cache: Whether to use cached embeddings
            batch_size: Batch size for ESM computation
            
        Returns:
            embeddings: Tensor of shape (num_sequences, max_length, embed_dim)
            masks: Boolean tensor of shape (num_sequences, max_length)
        """
        max_length = max_length or config.MAX_SEQUENCE_LENGTH # sets to config value if not provided
        
        print(f"[*] Computing embeddings for {len(sequences)} sequences...")
        print(f"   Max length: {max_length}")
        print(f"   Use cache: {use_cache}")
        
        # Check cache for each sequence
        cached_results = [] # stores (index, (embedding, masked)) for cached ones
        sequences_to_compute = [] # just the ones we need to do
        sequence_indices = [] # origial positions
        
        # for each sequence:
        # if use_cache is True: try to load from disk via _load_from_cache(seq)
        # if found in cache: add to cache_results and skip to next sequence
        # if not found: add to sequences_to_compute
        for i, seq in enumerate(sequences): 
            if use_cache:
                cached = self._load_from_cache(seq) # try to load from disk
                if cached is not None:
                    cached_results.append((i, cached))
                    continue # skip to next sequence
            
            # if we get to this part that means that specific sequence isn't cached
            # CRITICAL FIX: Truncate sequence BEFORE computing to prevent OOM on massive proteins
            if max_length and len(seq) > max_length:
                seq = seq[:max_length]
            
            sequences_to_compute.append(seq)
            sequence_indices.append(i)
        
        print(f"   Found {len(cached_results)} cached embeddings")
        print(f"   Computing {len(sequences_to_compute)} new embeddings")
        
        # compute the sequences thatwe stored in sequences_to_compute
        computed_results = []
        if sequences_to_compute: # only do this part if we have uncached sequences
            embeddings, masks = self._compute_embeddings(sequences_to_compute, batch_size, show_progress=show_progress) # compute the cache
            # ^ might turn into embeddings.shape = (2,100,480); masks.shape = (2, 100)
            # runs ESM model on all uncached sequences at once

            for j, (seq, emb, mask) in enumerate(zip(sequences_to_compute, embeddings, masks)):
                # Save to cache
                """ zip pairs up items from multiple lists:
                   Example:
                    sequences_to_compute = ["AIVLGG...", "PVKLGG..."]
                    embeddings = tensor([embedding_0, embedding_1])  # shape (2, 100, 480)
                    masks = tensor([mask_0, mask_1])                 # shape (2, 100)

                    zip(sequences_to_compute, embeddings, masks)
                    → [
                        ("AIVLGG...", embedding_0, mask_0),  # First tuple
                        ("PVKLGG...", embedding_1, mask_1)   # Second tuple
                        ] 
                      enumerate adds intex counter to each item so we can sort by index:
                        → [
                            (0, ("AIVLGG...", embedding_0, mask_0)),  # j=0
                            (1, ("PVKLGG...", embedding_1, mask_1))   # j=1
                            ]  

                enumerate unpacks everything on one line:
                  On first iteration:
                    j = 0
                    seq = "AIVLGG..."
                    emb = embedding_0  # shape (100, 480)
                    mask = mask_0      # shape (100,)
         
                        """
                


                
                if use_cache:
                    pass # self._save_to_cache(seq, emb, mask) # DISABLED: Too slow to write 1000 files
                
                # Store result
                original_index = sequence_indices[j] # need to keep track of what the original index was
                # has position and the (emb, mask) part.
                computed_results.append((original_index, (emb, mask))) # store in computed_results
        
        # Combine cached and computed results
        all_results = cached_results + computed_results # concatonates the two lists with embeddings
        # ^ currently these are out of order
        all_results.sort(key=lambda x: x[0])  # Sort by original index
        


        # will contain embeddings again after padding/truncation
        final_embeddings = []
        final_masks = []
        
        for _, (emb, mask) in all_results:
            # Pad or truncate to max_length
            curr_len = emb.size(0)
            
            if curr_len > max_length: # if it's too long we'll cut it off rip
                # Truncate
                emb = emb[:max_length]
                mask = mask[:max_length]
            elif curr_len < max_length: # if it's too short we'll add padding with 0s
                # Pad
                pad_len = max_length - curr_len
                emb_pad = torch.zeros(pad_len, emb.size(1))
                mask_pad = torch.zeros(pad_len, dtype=torch.bool)
                
                emb = torch.cat([emb, emb_pad], dim=0)
                mask = torch.cat([mask, mask_pad], dim=0)
            
            final_embeddings.append(emb) 
            final_masks.append(mask)
        
        # Stack into tensors 
        # stack combines the embeddings into one big tensor
        embeddings_tensor = torch.stack(final_embeddings, dim=0)
        """
        embeddings_tensor = torch.stack(final_embeddings, dim=0)
        # Result: (4, 512, 480)
        #          ↑  ↑    ↑
        #          │  │    └─ embedding dimension
        #          │  └─ sequence length (padded)
        #          └─ batch size (number of sequences)

        Visual:
        Stack along dim=0:
        [emb_A] ─┐
        [emb_B] ─┤
        [emb_C] ─┼─→ embeddings_tensor (4, 512, 480)
        [emb_D] ─┘

        """
        masks_tensor = torch.stack(final_masks, dim=0)
        """
        Line 292: masks_tensor = torch.stack(final_masks, dim=0)

        Same thing for masks:
        final_masks = [
            tensor (512,),  # mask_A
            tensor (512,),  # mask_B
            tensor (512,),  # mask_C
            tensor (512,)   # mask_D
        ]

        masks_tensor = torch.stack(final_masks, dim=0)
        # Result: (4, 512)
        #          ↑  ↑
        #          │  └─ sequence length
        #          └─ batch size

        ---
        """
        # same thing happens with the masks
        
        print(f"[OK] Embeddings computed successfully!")
        print(f"   Shape: {embeddings_tensor.shape}")
        print(f"   Embedding dim: {embeddings_tensor.shape[-1]}")
        
        return embeddings_tensor, masks_tensor
    
    def clear_cache(self): # deletes all cached embeddings
        # use this if we want to change ESM models or recompute everything from start
        """Clear all cached embeddings."""
        import shutil
        if self.cache_dir and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"🗑️ Cache cleared: {self.cache_dir}")
    
    def cache_stats(self): # shows cached statistics
        # how many files and total size
        """Print cache statistics."""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            print("[*] Cache is empty (directory doesn't exist)")
            return

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pt')]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f))
                        for f in cache_files)

        print(f"[*] Cache Statistics:")
        print(f"   Files: {len(cache_files)}")
        print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")
        print(f"   Directory: {self.cache_dir}")

    def get_cache_size_gb(self) -> float:
        """Get total cache size in GB."""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return 0.0

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pt')]
        total_bytes = sum(os.path.getsize(os.path.join(self.cache_dir, f))
                         for f in cache_files)
        return total_bytes / (1024 ** 3)  # Convert to GB

    def cleanup_old_cache(self, max_size_gb: float = None):
        """
        Clean up old cache files if cache exceeds maximum size.

        Removes oldest files first (by modification time) until cache is under limit.

        Args:
            max_size_gb: Maximum cache size in GB (default from config)
        """
        max_size_gb = max_size_gb or config.MAX_CACHE_SIZE_GB

        if max_size_gb <= 0:
            return  # No limit

        current_size = self.get_cache_size_gb()

        if current_size <= max_size_gb:
            return  # Under limit

        print(f"⚠️ Cache size ({current_size:.2f} GB) exceeds limit ({max_size_gb:.2f} GB)")
        print(f"   Cleaning up old cache files...")

        # Get all cache files with their modification times
        cache_files = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pt'):
                filepath = os.path.join(self.cache_dir, filename)
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                cache_files.append((filepath, mtime, size))

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])

        # Remove oldest files until under limit
        removed_count = 0
        for filepath, _, size in cache_files:
            if current_size <= max_size_gb:
                break

            try:
                os.remove(filepath)
                current_size -= size / (1024 ** 3)
                removed_count += 1
            except OSError:
                pass

        print(f"[OK] Removed {removed_count} old cache files")
        print(f"   New cache size: {current_size:.2f} GB")

    def cleanup_cache_on_start(self):
        """Clean cache at startup if configured to do so."""
        if config.CACHE_CLEANUP_ON_START:
            print("🧹 Cleaning cache on startup (CACHE_CLEANUP_ON_START=True)")
            self.clear_cache()
        else:
            # Check size limit
            self.cleanup_old_cache()


# Convenience function for quick embedding computation
# calls the main method embed_sequences() in one line
# convenience wrapper function
def compute_esm_embeddings(sequences: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to compute ESM embeddings.
    
    Args:
        sequences: List of protein sequences
        **kwargs: Additional arguments passed to ESMEmbedder.embed_sequences
        
    Returns:
        embeddings: Tensor of shape (num_sequences, max_length, embed_dim)
        masks: Boolean tensor of shape (num_sequences, max_length)
    """
    embedder = ESMEmbedder()
    return embedder.embed_sequences(sequences, **kwargs)