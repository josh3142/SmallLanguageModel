""" 
Transformer similar to GPT-2 model 
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
https://github.com/openai/gpt-2

For a pytorch implementation have a look at 
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    max_seq_len: int = 256
    emb_dim: int = 384  # embedding dimension
    n_layers: int = 6
    n_heads: int = 6
    dropout: float = 0.1
    bias: bool = True
    
    def __post_init__(self):
        assert self.emb_dim % self.n_heads == 0, "emb_dim must be divisible by n_heads"

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention used in transformer decoder blocks.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.emb_dim % config.n_heads == 0
        
        # Query, key, value projections combined in one layer
        self.c_attn = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_heads = config.n_heads
        self.emb_dim = config.emb_dim
        self.dropout = config.dropout
        
        # Causal mask to prohibit access to future tokens
        self.register_buffer(
            "causal_mask", 
            torch.tril(
                torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
            ).view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    @staticmethod
    def _reshape_projection(
            p: torch.Tensor, b: int, seq_len: int, n_heads: int, emb_dim: int
        ) -> torch.Tensor:
        """
        Reshape projections `p` for multi-head attention to 
        (b, n_heads, seq_len, emb_dim / n_heads) shape
        """
        return p.view(b, seq_len, n_heads, emb_dim // n_heads).transpose(1,2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.size()  # batch, sequence length, embedding dimension
        
        # Calculate QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.emb_dim, dim=2)
        
        k = self._reshape_projection(k, b, seq_len, self.n_heads, self.emb_dim)
        q = self._reshape_projection(q, b, seq_len, self.n_heads, self.emb_dim)
        v = self._reshape_projection(v, b, seq_len, self.n_heads, self.emb_dim)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # (b, n_heads, seq_len, seq_len)
        att = self.attn_dropout(att)
        y = att @ v 
        assert y.shape == (b, self.n_heads, seq_len, self.emb_dim // self.n_heads)

        # Re-assemble all head outputs side by side and return the value
        y = y.transpose(1, 2).contiguous().view(b, seq_len, self.emb_dim)
        assert y.shape == (b, seq_len, self.emb_dim)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    Two layer Multi-layer perceptron with GELU and dropout.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * 4, bias=config.bias),
            nn.GELU(),  # GPT-2 uses GELU
            nn.Linear(config.emb_dim * 4, config.emb_dim, bias=config.bias),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2LanguageModel(nn.Module):
    """
    GPT-2 Language Model with decoder-only transformer architecture (like GPT-2).

    Features of the model:
        - Causal self-attention: attends only to previous tokens, enabling 
            autoregressive generation.
        - decoder-only transformer (only causal attention mechanism)
        - Multi-head attention: splits embedding dimension across multiple 
            attention heads.
        - Positional embeddings: learned positional embeddings are added to 
            token embeddings to incorporate token order information.
        - weights of embeddings and weights of the final layer are shared due to 
            symmetry: M: tokens -> vectors, N: vectors -> tokens with M=N.T
        - Pre-layer normalization: LayerNorm is applied before each sub-layer 
            (attention and MLP).
        - Initialization: weights initialized with normal distribution 
            (mean=0, std=0.02)
        - Dropout regularization applied after embeddings, attention weights, 
            and MLP layers.
        - Uses GELU activation in MLP layers.

    References:
        - Original GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position_emb = nn.Embedding(config.max_seq_len, config.emb_dim)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final layer norm and language model head
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        
        # Weight sharing (tie embeddings and output weights - GPT-2 convention)
        self.token_emb.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    
    def get_num_params(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        b, seq_len = x.size() # (batch size, sequence length)
        assert seq_len <= self.config.max_seq_len, (
            f"Sequence length {seq_len} exceeds "  
            f"maximum {self.config.max_seq_len}"
        )
        
        idcs_pos = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)
        pos_emb = self.position_emb(idcs_pos)  # position embeddings (1, seq_len, emb_dim)
        tok_emb = self.token_emb(x)  # token embeddings (b, seq_len, emb_dim)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
