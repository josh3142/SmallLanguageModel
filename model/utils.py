import tiktoken
from transformers import AutoTokenizer
from typing import List
  
class GPT2Tokenizer:
    """
    GPT-2 tokenizer wrapper for consistency with TinyStories implementations
    Most repositories use GPT-2 tokenizer for fair comparison
    """
    def __init__(self):
        # self.tokenizer = tiktoken.get_encoding("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories')
        # self.vocab_size = self.tokenizer.n_vocab
        
        # Special tokens
        self.pad_token = 50256  # Using a high number as PAD
        self.eos_token = 50256  # GPT-2 uses same token for EOS
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text"""
        # Filter out pad tokens
        token_ids = [t for t in token_ids if t != self.pad_token]
        return self.tokenizer.decode(token_ids)