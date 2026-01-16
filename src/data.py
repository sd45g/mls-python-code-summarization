"""
Data loading and preprocessing module for code summarization.

This module provides:
- JsonlCodeSummaryDataset: Loads code-summary pairs from JSONL files
- Collator: Tokenizes and batches data for training
- Batch: Container for batched tensors
"""

import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class JsonlCodeSummaryDataset(Dataset):
    """
    Dataset that loads code-summary pairs from a JSONL file.
    
    Each line in the file should be a JSON object with 'code' and 'summary' keys.
    """
    
    def __init__(self, data_path: str):
        """Load all examples from the JSONL file."""
        self.examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
    
    def __len__(self):
        """Return the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Return a single example as a dictionary."""
        return self.examples[idx]


class Batch:
    """Container for a batch of tokenized sequences."""
    
    def __init__(self, src_ids, src_mask, tgt_ids):
        """
        Args:
            src_ids: Source token IDs [batch_size, src_len]
            src_mask: Attention mask for source [batch_size, src_len]
            tgt_ids: Target token IDs [batch_size, tgt_len]
        """
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.tgt_ids = tgt_ids


class Collator:
    """
    Collator that tokenizes and batches code-summary pairs.
    
    Handles tokenization, adding special tokens (BOS/EOS), and padding.
    """
    
    def __init__(self, tokenizer_path, max_src_len=256, max_tgt_len=64):
        """
        Initialize the collator with a tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer JSON file
            max_src_len: Maximum source (code) sequence length
            max_tgt_len: Maximum target (summary) sequence length
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Get special token IDs
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        
        if self.bos_id is None or self.eos_id is None or self.pad_id is None:
            raise ValueError("Tokenizer must contain [BOS], [EOS], and [PAD] special tokens.")
            
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __call__(self, batch_list):
        """
        Tokenize and batch a list of examples.
        
        Args:
            batch_list: List of dicts with 'code' and 'summary' keys
            
        Returns:
            Batch object with src_ids, src_mask, tgt_ids tensors
        """
        
        src_ids_list = []
        tgt_ids_list = []
        
        for item in batch_list:
            code = item.get("code", "")
            summary = item.get("summary", "")
            
            # Source: [BOS] code [EOS]
            # Since user's code suggests simple encoding, we'll stick to standard adding
            curr_src = self.tokenizer.encode(code).ids
            # Truncate for BOS+EOS
            if len(curr_src) > self.max_src_len - 2:
                curr_src = curr_src[:self.max_src_len - 2]
            src_full = [self.bos_id] + curr_src + [self.eos_id]
            src_ids_list.append(torch.tensor(src_full, dtype=torch.long))
            
            # Target: [BOS] summary [EOS]
            curr_tgt = self.tokenizer.encode(summary).ids
            if len(curr_tgt) > self.max_tgt_len - 2:
                curr_tgt = curr_tgt[:self.max_tgt_len - 2]
            tgt_full = [self.bos_id] + curr_tgt + [self.eos_id]
            tgt_ids_list.append(torch.tensor(tgt_full, dtype=torch.long))

        # Pad sequences
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=self.pad_id)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids_list, batch_first=True, padding_value=self.pad_id)
        
        # Mask: 1 for real, 0 for pad
        src_mask = (src_padded != self.pad_id).long()
        
        return Batch(src_padded, src_mask, tgt_padded)
