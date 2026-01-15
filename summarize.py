#!/usr/bin/env python
"""
Command-line interface for code summarization.

Usage:
    python summarize.py --input "def add(a, b): return a + b"
    python summarize.py --input "def add(a, b): return a + b" --model models/best.pt
"""

import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from tokenizers import Tokenizer
from src.transformer_model import TransformerSeq2Seq


def load_model(model_path: str, tokenizer_path: str, device: str):
    """Load the trained model and tokenizer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise ValueError("Tokenizer missing [PAD] token")

    # Model config must match training
    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        pad_id=pad_id,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()
    return model, tokenizer


def summarize(model, tokenizer, code: str, device: str, max_src_len: int = 256, max_gen_len: int = 64) -> str:
    """Generate a summary for the given code."""
    if not code.strip():
        return ""

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    if bos_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must contain [BOS], [EOS], [PAD]")

    # Encode with BOS/EOS tokens
    enc = tokenizer.encode(code)
    ids = enc.ids[:max_src_len - 2]
    ids = [bos_id] + ids + [eos_id]
    if len(ids) < max_src_len:
        ids = ids + [pad_id] * (max_src_len - len(ids))

    src_ids = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = (src_ids != pad_id).long()

    with torch.no_grad():
        sequences = model.generate(
            src_ids=src_ids,
            src_mask=src_mask,
            max_len=max_gen_len,
            bos_id=bos_id,
            eos_id=eos_id,
        )

    gen_ids = sequences[0]

    # Remove BOS
    if gen_ids and gen_ids[0] == bos_id:
        gen_ids = gen_ids[1:]

    # Cut at EOS
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]

    summary = tokenizer.decode(gen_ids).strip()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate a summary for Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python summarize.py --input "def add(a, b): return a + b"
    python summarize.py --input "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Python code to summarize"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/best.pt",
        help="Path to model checkpoint (default: models/best.pt)"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default="data/tokenizer/tokenizer.json",
        help="Path to tokenizer (default: data/tokenizer/tokenizer.json)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at '{args.model}'")
        print("\nTo use this tool, you need a trained model:")
        print("  1. Download best.pt from Google Drive")
        print("  2. Place it in the 'models/' directory")
        print("  3. Or specify a custom path with --model /path/to/model.pt")
        sys.exit(1)
    
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found at '{args.tokenizer}'")
        print("Run 'python scripts/train_tokenizer.py' first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, args.tokenizer, device)
    
    print(f"\nInput code:\n{args.input}\n")
    
    summary = summarize(model, tokenizer, args.input, device)
    
    print(f"Generated summary:\n{summary}")


if __name__ == "__main__":
    main()
