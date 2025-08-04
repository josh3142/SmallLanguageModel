from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

from model.transformer import GPT2LanguageModel
from model.utils import GPT2Tokenizer


class Inference:
    """
    Inference class for text generation and evaluation.
    
    Provides methods for text generation with various sampling strategies,
    perplexity calculation, and model introspection. Supports loading from
    checkpoints and batch processing.
    """
    
    def __init__(
            self,
            model: nn.Module,
            tokenizer: GPT2Tokenizer,
            device: str = "cpu"
        ) -> None:
        """        
        Args:
            model: Pytorch model instance
            tokenizer: Tokenizer for encoding/decoding text
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device= device
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text with multiple sampling strategies.
        
        Args:
            text: Input text
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0 for greedy, higher for more random)
            top_k: Sample k most likely tokens (None to disable)
            top_p: Sample all tokens s.t. cumulative probability is greater than
                top_p (None to disable)
            stop_tokens: List of strings to stop generation when encountered
            
        Returns:
            Generated text including the original text
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) == 0:
            tokens = [self.tokenizer.eos_token]
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        generated_tokens = []

        for _ in range(max_new_tokens):
            input_tokens = tokens[:, -self.model.config.max_seq_len:]
            logits = self.model(input_tokens)
            # model generates for each token the next one, but we only need the
            # last token, because it represents the word to add.
            logits = logits[0, -1, :] / temperature

            # Select top-k tokens and give all others "zero probability"
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('inf')

            # Select top-p tokens that have a last probability p
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_idcs = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_idcs_to_remove = cumulative_probs > top_p
                sorted_idcs_to_remove[1:] = sorted_idcs_to_remove[:-1].clone()
                sorted_idcs_to_remove[0] = 0
                idcs_to_remove = sorted_idcs[sorted_idcs_to_remove]
                logits[idcs_to_remove] = -float('inf')

            # Sample next token
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())

            # Stop conditions
            if next_token.item() == self.tokenizer.eos_token:
                break
            if stop_tokens:
                current_text = self.tokenizer.decode(generated_tokens)
                if any(stop in current_text for stop in stop_tokens):
                    break

        full_tokens = tokens[0].cpu().numpy().tolist()
        return self.tokenizer.decode(full_tokens)

    @torch.no_grad()
    def get_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of a text.
        
        Perplexity measures how well the model predicts the text. Lower values
        indicate better prediction (less "perplexed" by the text).
        
        Args:
            text: Input text to calculate perplexity for
            
        Returns:
            Perplexity score (float('inf') if text is too short)
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2:
            return float('inf')

        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        total_loss, total_tokens = 0, 0
        max_seq_len = self.model.config.max_seq_len

        for i in range(0, tokens.size(1) - 1, max_seq_len - 1):
            chunk = tokens[:, i:i + max_seq_len]
            if chunk.size(1) < 2:
                continue
            inputs, targets = chunk[:, :-1], chunk[:, 1:]
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

        return float('inf') if total_tokens == 0 else math.exp(total_loss / total_tokens)

    def generate_multiple(self, texts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple texts with the same parameters.
        
        Args:
            texts: List of input texts
            **kwargs: Any parameters accepted by generate() method
            
        Returns:
            List of generated texts corresponding to input texts
        """
        return [self.generate(text, **kwargs) for text in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded model.
        
        Returns:
            Dictionary containing model configuration and statistics including
            parameter count, architecture details, and device information
        """
        return {
            'num_parameters': self.model.get_num_params(),
            'vocab_size': self.model.config.vocab_size,
            'max_seq_len': self.model.config.max_seq_len,
            'emb_dim': self.model.config.emb_dim,
            'n_layers': self.model.config.n_layers,
            'n_heads': self.model.config.n_heads,
            'device': self.device
        }

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str, 
        tokenizer: Optional[GPT2Tokenizer] = None,
        device: str = "cpu"
    ) ->  Inference:
        """
        Load model from checkpoint file and create inference instance.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
            tokenizer: Tokenizer instance (creates GPT2Tokenizer if None)
            device: Device to load model on ('cpu', 'cuda', etc.)
            
        Returns:
            Inference instance with loaded model ready for inference
        """
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = checkpoint['config']
        model = GPT2LanguageModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        if tokenizer is None:
            tokenizer = GPT2Tokenizer()

        print(f"Model loaded with {model.get_num_params():,} parameters")
        print(f"Checkpoint at epoch {checkpoint.get('epoch', 'unknown')}, " +
              f"step {checkpoint.get('step', 'unknown')}")
        return cls(model=model, tokenizer=tokenizer, device=device)
