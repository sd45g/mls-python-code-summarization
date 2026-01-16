#!/usr/bin/env python
"""
Command-line interface for code summarization.

Usage:
    python summarize.py                    # Interactive mode
    python summarize.py --input "def ..."  # Single input mode
"""

import os
import sys
import argparse

# Add scripts folder to path so we can import inference
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

import torch
from inference import load_inference_model_transformer, summarize_code_transformer


# Default paths - Google Drive for Colab, local for PC
COLAB_MODEL_PATH = "/content/drive/MyDrive/mls-python-code-summarization/models/best.pt"
LOCAL_MODEL_PATH = "models/best.pt"
TOKENIZER_PATH = "data/tokenizer/tokenizer.json"


def get_model_path():
    """Auto-detect model path: prefer Colab Drive, fallback to local."""
    if os.path.exists(COLAB_MODEL_PATH):
        return COLAB_MODEL_PATH
    elif os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    else:
        return None


def interactive_mode(model, tokenizer, device):
    """Interactive loop: enter code, get summary, repeat."""
    print("\n" + "="*60)
    print("INTERACTIVE CODE SUMMARIZATION")
    print("="*60)
    print("Enter Python code and press Enter to get a summary.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            code = input("Enter code: ").strip()
            
            if code.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not code:
                print("Please enter some code.\n")
                continue
            
            summary = summarize_code_transformer(model, tokenizer, code, device)
            print(f"Summary: {summary}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a summary for Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python summarize.py                                    # Interactive mode
    python summarize.py --input "def add(a, b): return a + b"  # Single input
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Python code to summarize (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detects Colab Drive or local)"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default=TOKENIZER_PATH,
        help="Path to tokenizer"
    )
    
    args = parser.parse_args()
    
    # Auto-detect model path if not specified
    model_path = args.model if args.model else get_model_path()
    
    if model_path is None:
        print("Error: Model not found!")
        print(f"\nChecked locations:")
        print(f"  - Colab Drive: {COLAB_MODEL_PATH}")
        print(f"  - Local: {LOCAL_MODEL_PATH}")
        print("\nMake sure the model exists in one of these locations.")
        sys.exit(1)
    
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found at '{args.tokenizer}'")
        print("Run 'python scripts/train_tokenizer.py' first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")
    model, tokenizer = load_inference_model_transformer(model_path, args.tokenizer, device)
    print("Model loaded successfully!\n")
    
    if args.input:
        # Single input mode
        print(f"Input code:\n{args.input}\n")
        summary = summarize_code_transformer(model, tokenizer, args.input, device)
        print(f"Generated summary:\n{summary}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
