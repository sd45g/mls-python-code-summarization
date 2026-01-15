"""
Evaluate trained model on test set and compute ROUGE, BLEU, and METEOR scores.
Run this after training to measure model quality.
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference import load_inference_model_transformer, summarize_code_transformer

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Installing rouge_score...")
    os.system("pip install rouge-score")
    from rouge_score import rouge_scorer

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    print("Installing nltk...")
    os.system("pip install nltk")
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


def compute_perplexity(model, tokenizer, examples, device, max_src_len=256, max_tgt_len=64):
    """
    Compute perplexity on a set of examples.
    Perplexity = exp(average cross-entropy loss)
    """
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    total_loss = 0.0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for example in tqdm(examples[:200], desc="Computing perplexity"):  # Limit for speed
            code = example['code']
            summary = example['summary']
            
            # Encode source
            src_enc = tokenizer.encode(code).ids[:max_src_len - 2]
            src_ids = [bos_id] + src_enc + [eos_id]
            src_ids = src_ids + [pad_id] * (max_src_len - len(src_ids))
            
            # Encode target
            tgt_enc = tokenizer.encode(summary).ids[:max_tgt_len - 2]
            tgt_full = [bos_id] + tgt_enc + [eos_id]
            tgt_full = tgt_full + [pad_id] * (max_tgt_len - len(tgt_full))
            
            src_ids = torch.tensor([src_ids], dtype=torch.long, device=device)
            tgt_ids = torch.tensor([tgt_full], dtype=torch.long, device=device)
            src_mask = (src_ids != pad_id).long()
            
            tgt_in = tgt_ids[:, :-1]
            labels = tgt_ids[:, 1:]
            
            logits = model(src_ids, src_mask, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            num_tokens = (labels != pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    test_data_path: str,
    device: str = "cuda",
    max_samples: int = 1000,
):
    """
    Evaluate model on test data and compute ROUGE, BLEU, METEOR, and Perplexity.
    """
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_inference_model_transformer(model_path, tokenizer_path, device)
    
    print(f"Loading test data from {test_data_path}...")
    examples = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    # Limit samples for faster evaluation
    if max_samples and len(examples) > max_samples:
        import random
        random.seed(42)
        examples = random.sample(examples, max_samples)
    
    print(f"Evaluating on {len(examples)} samples...")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1  # Smoothing for short sentences
    
    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'meteor': []}
    predictions = []
    references = []
    
    for example in tqdm(examples, desc="Generating summaries"):
        code = example['code']
        reference = example['summary']
        
        # Generate prediction
        prediction = summarize_code_transformer(
            model, tokenizer, code, device,
            max_src_len=256, max_gen_len=64
        )
        
        predictions.append(prediction)
        references.append(reference)
        
        # Compute ROUGE scores
        scores = scorer.score(reference, prediction)
        for key in ['rouge1', 'rouge2', 'rougeL']:
            all_scores[key].append(scores[key].fmeasure)
        
        # Compute BLEU score
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        if len(pred_tokens) > 0:
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            # Compute METEOR score (handles synonyms and stemming)
            meteor = meteor_score([ref_tokens], pred_tokens)
        else:
            bleu = 0.0
            meteor = 0.0
        all_scores['bleu'].append(bleu)
        all_scores['meteor'].append(meteor)
    
    # Compute averages
    avg_scores = {}
    for key in all_scores:
        avg_scores[key] = sum(all_scores[key]) / len(all_scores[key])
    
    # Compute perplexity
    print("\nComputing perplexity...")
    perplexity = compute_perplexity(model, tokenizer, examples, device)
    avg_scores['perplexity'] = perplexity
    
    print("\n" + "="*50)
    print("EVALUATION SCORES")
    print("="*50)
    print("ROUGE Scores:")
    print(f"  ROUGE-1: {avg_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {avg_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {avg_scores['rougeL']:.4f}")
    print("-"*50)
    print("Other Metrics:")
    print(f"  BLEU:       {avg_scores['bleu']:.4f}")
    print(f"  METEOR:     {avg_scores['meteor']:.4f}")
    print(f"  Perplexity: {avg_scores['perplexity']:.2f}")
    print("="*50)
    
    # Show some examples
    print("\nSample predictions:")
    print("-"*50)
    for i in range(min(5, len(predictions))):
        print(f"\nCode (truncated): {examples[i]['code'][:100]}...")
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-"*50)
    
    return avg_scores


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths (adjust for Colab)
    model_path = "/content/drive/MyDrive/mls-python-code-summarization/models/best.pt"
    tokenizer_path = "data/tokenizer/tokenizer.json"
    test_data_path = "data/processed/codesearchnet_clean/test.jsonl"
    
    # Check if files exist
    if not os.path.exists(model_path):
        # Try local path
        model_path = "models/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Train the model first using: python scripts/train.py")
        return
    
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}")
        print("Run preprocessing first: python scripts/preprocess_data.py")
        return
    
    scores = evaluate_model(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        test_data_path=test_data_path,
        device=device,
        max_samples=1000,
    )
    
    return scores


if __name__ == "__main__":
    main()
