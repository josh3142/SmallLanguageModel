#!/usr/bin/env python3
"""
TinyStories Language Model Inference Script
Load and test a trained TinyStories model with various generation options
"""
import os
import argparse
import torch

from model.transformer import ModelConfig
from generation.inference import Inference

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Suppress tokenizer warnings


model_config = ModelConfig(
    vocab_size=50257,
    max_seq_len=256,
    emb_dim=512,
    n_layers=12,
    n_heads=8,
    dropout=0.1
)

def main():
    parser = argparse.ArgumentParser(description="TinyStories Model Inference")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", "-d", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--prompt", "-p", help="Single prompt for generation")
    parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, help="Top-p sampling (0 to disable)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    inference = Inference.load_from_checkpoint(
        checkpoint_path=args.checkpoint, device=device
    )

    # prompt generation
    if args.prompt:
        print(f"\nGenerating for prompt: '{args.prompt}'")
        print("-" * 50)
        
        generated = inference.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
        )
        print(generated)
        print(f"Perplexity of text: {inference.get_perplexity(generated)}")

if __name__ == "__main__":
    main()
