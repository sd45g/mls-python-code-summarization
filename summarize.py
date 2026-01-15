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

# Add scripts folder to path so we can import inference
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

import torch
from inference import load_inference_model_transformer, summarize_code_transformer


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
    model, tokenizer = load_inference_model_transformer(args.model, args.tokenizer, device)
    
    print(f"\nInput code:\n{args.input}\n")
    
    summary = summarize_code_transformer(model, tokenizer, args.input, device)
    
    print(f"Generated summary:\n{summary}")


if __name__ == "__main__":
    main()
