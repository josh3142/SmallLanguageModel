#!/usr/bin/env python3
"""
Tiny Language Model Training on TinyStories Dataset
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Any, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset

from model.transformer import GPT2LanguageModel, ModelConfig
from model.utils import GPT2Tokenizer
from utils import set_seed, ExperimentManager, plot_loss
from training.trainer import Trainer, TrainingConfig


def setup_logging(name: str):
    """
    Setup logging configuration.
    
    The logged information will be printed to the terminal and saved `name`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(name, mode="a"), # save logging message in file
            logging.StreamHandler() # print logging meassage to terminal
        ]
    )
    return logging.getLogger(__name__) # name logger after current module

# Get a module-level logger; this logger won't output anything yet until logging
# is configured in the main script using setup_logging().
# This allows logging calls in this file (e.g., TinyStoriesDataset)
# to work properly without needing to configure logging here.
logger = logging.getLogger(__name__)


class TinyStoriesDataset(Dataset):
    """
    TinyStories dataset with proper preprocessing.
    """
    def __init__(self,
            split: str = "train",
            max_length: int = 256,
            max_examples: Optional[int] = None,
            tokenizer: Any = GPT2Tokenizer(),
            combine_text: bool = False
        ):
        """Initializes the TinyStoriesDataset object.

        Args:
            split (str, optional):  Which data split to use, either 'train' or 
                'val'. Defaults to "train".
            max_length (int, optional): Maximum length of tokenized sequences. 
                Sequences longer than this are truncated. Default is 256.
            max_examples (Optional[int], optional): Maximum number of examples 
                to load from the dataset. If None, all examples are loaded.
            tokenizer (Any, optional): Tokenizer used to convert text into token 
                IDs. Default is GPT2Tokenizer.
            combine_text (bool, optional):  Whether to combine multiple texts 
                into a single sequence (batch) or not. Defaults to False.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.combine_text = combine_text
        
        logger.info(f"Loading TinyStories dataset ({split} split)...")
        try:
            dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
            if max_examples:
                dataset = dataset.select(range(min(max_examples, len(dataset))))
            
            self.texts = [example['text'] for example in dataset]
            logger.info(f"Loaded {len(self.texts)} examples from TinyStories")
            
        except Exception as e:
            logger.warning(f"Could not load TinyStories dataset: {e}")
            logger.info("Using sample data instead...")
            self.texts = self._get_sample_data() * 100 
        
        self.data = self._preprocess_and_tokenize_data()
        logger.info(f"Preprocessed {len(self.data)} training examples")
    
    def _get_sample_data(self) -> List[str]:
        """Fallback sample data if TinyStories can't be loaded.
        
        (Only useful for debugging.)
        """
        return [
            "Once upon a time, there was a little girl named Lucy. She loved to play in the garden with her colorful flowers.",
            "Tom the cat was very curious about everything. He liked to explore the big house and find new places to sleep.",
            "Sarah and her grandmother loved to bake cookies together. They would mix flour and sugar in a big bowl.",
            "The little bird wanted to learn how to fly. His mother showed him how to flap his wings up and down.",
            "Max had a bright red bicycle that he rode everywhere. He loved to feel the wind blowing through his hair.",
            "Emma found a magic book in the old library. When she opened it, golden letters appeared on the pages.",
            "The friendly dog named Charlie loved to play fetch with his favorite tennis ball in the park.",
            "Lily discovered a tiny fairy living in her garden. The fairy had sparkly wings and a sweet voice."
        ]
    
    def _preprocess_and_tokenize_data(self) -> List[torch.Tensor]:
        """Preprocess and tokenize all texts."""

        if self.combine_text:
            processed_data = self._concatenate_and_slice_data()
        else:
            processed_data = self._split_and_pad_data()
        return processed_data
    
    def _split_and_pad_data(self) -> List[torch.Tensor]:
        """Each batch contains a single text distributed over several patches. 
        
        A single text is split into several batches and to last batch is filled
        to `max_length` such that all batches have the same length. 
        """
        processed_data = []
        for text in tqdm(self.texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_token]
            
            # Split into chunks and pad until the batch is full
            for i in range(0, len(tokens), self.max_length):
                chunk = tokens[i:i + self.max_length]
                
                if len(chunk) < self.max_length:
                    chunk.extend([self.tokenizer.pad_token] * (self.max_length - len(chunk)))
                
                processed_data.append(torch.tensor(chunk, dtype=torch.long))
        return processed_data
    
    def _concatenate_and_slice_data(self) -> List[torch.Tensor]:
        """Entire text is concatenated and finally splitted in batches.

        The entire text is concatenated and splitted in batches. Each batch
        can contain more than one original text. The last batch is filled
        to `max_length` such that all batches have the same length. 
        (This can only happen to one batch)
        """
        processed_data = []
        concatenated_data = []
        for text in tqdm(self.texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_token]
            concatenated_data += tokens

        for i in range(0, len(concatenated_data), self.max_length):
            chunk = concatenated_data[i:i + self.max_length]
            
            if len(chunk) < self.max_length:
                chunk.extend([self.tokenizer.pad_token] * (self.max_length - len(chunk)))

            processed_data.append(torch.tensor(chunk, dtype=torch.long))
        return processed_data
    
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Get item for next token prediction. 
        Input is all tokens except last, target is all tokens except first. 
        """
        tokens = self.data[idx]
        input, output = tokens[:-1], tokens[1:]
        return input, output


@torch.no_grad()
def generate_text(
    model: GPT2LanguageModel, 
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    device: str = "cpu"
) -> str:
    """
    Generates text from a given prompt.

    This function uses temperature scaling and top-k sampling to produce 
    diverser. It stops generating when it reaches the specified number of new 
    tokens, encounters an end-of-sequence (EOS) token, or exceeds the model's 
    maximum sequence length.

    Args:
        model (GPT2LanguageModel): The language model used for generation.
        tokenizer (GPT2Tokenizer): The tokenizer used to encode the prompt and 
            decode the output.
        prompt (str): The input text prompt to begin generation from.
        max_new_tokens (int, optional): Maximum number of new tokens to 
            generate. Defaults to 100.
        temperature (float, optional): Sampling temperature. Lower values make 
            the output more deterministic. Defaults to 0.8.
        top_k (Optional[int], optional): If set, only the top_k most probable 
            tokens are considered for sampling. Defaults to 50.
        device (str, optional): The device to run the model on. Defaults to "cpu".

    Returns:
        str: The generated text including the prompt.
    """
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type="cuda"):
            logits = model(tokens)
        
        # only last token needed for prediciton
        logits = logits[0, -1, :] / temperature

        # Top-k filtering by setting all small logits to -inf
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('inf')
        
        # select one of the top_k tokens randomly according to their prob
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        # Stop if we hit EOS or exceed max length
        if next_token.item() == tokenizer.eos_token:
            break
        if tokens.size(1) >= model.config.max_seq_len:
            break
    
    generated_tokens = tokens[0].cpu().numpy().tolist()
    return tokenizer.decode(generated_tokens)


def main():
    model_config = ModelConfig(
        vocab_size=50257,  # GPT-2 tokenizer
        max_seq_len=256,
        emb_dim=512,
        n_layers=12,
        n_heads=8,
        dropout=0.1
    )
    
    train_config = TrainingConfig(
        batch_size=32,
        learning_rate=3e-4,
        epochs=10,
        max_examples=None,
        experiment_description="baseline",
        resume_from_ckpt=None
    )
    set_seed(train_config.seed)

    # Setup experiment
    exp_manager = ExperimentManager()
    exp_manager.setup_experiment(
        description=train_config.experiment_description,
        resume_from=train_config.resume_from_ckpt
    )
    
    logger = setup_logging(
        os.path.join(exp_manager.experiment_dir, "training.log")
    )
    logger.info(f"Experiment: {exp_manager.experiment_name}")
    logger.info(f"Experiment directory: {exp_manager.experiment_dir}")

    logger.info(f"Training on device: {train_config.device}")
    logger.info(f"Model config: {model_config}")
    
    # Create datasets model and trainer
    logger.info("Loading datasets...")
    train_dataset = TinyStoriesDataset(
        "train", model_config.max_seq_len, max_examples=train_config.max_examples
    )
    val_dataset = TinyStoriesDataset(
        "validation", model_config.max_seq_len, max_examples=train_config.max_examples
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size, shuffle=False
    )
    
    # initialize model and trainer. optionally load previous checkpoint
    model = GPT2LanguageModel(model_config)
    trainer = Trainer(model, train_config) 
    if train_config.resume_from_ckpt:
        trainer.load_checkpoint(
            exp_manager.get_checkpoint(train_config.resume_from_ckpt)
        )

    # testing prompts
    test_prompts = [
        "Once upon a time",
        "The little girl"
    ]
    tokenizer = GPT2Tokenizer()
 
    # Training loop
    logger.info("Starting training...")
    start_epoch = trainer.epoch
    for epoch in range(start_epoch, start_epoch + train_config.epochs):
        train_loss = trainer.train_epoch(train_loader, val_loader)
        logger.info(f"Epoch {epoch+1}/{train_config.epochs} - Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % 1 == 0: # Save checkpoint
            trainer.save_checkpoint(
                exp_manager.get_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            )                

        for prompt in test_prompts:
            generated = generate_text(  
            model, tokenizer, prompt, 
            max_new_tokens=50, 
            temperature=0.8, 
            device=train_config.device
            )
            logger.info(f"\nPrompt: '{prompt}'")
            logger.info(f"Generated: {generated}")

        plot_loss(trainer, exp_manager.get_plot_path())
        
    final_val_loss = trainer.evaluate(val_loader)
    logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
    logger.info("Testing text generation...")
    tokenizer = GPT2Tokenizer()
    
    trainer.save_checkpoint(exp_manager.get_checkpoint("final_model.pth"))
    logger.info("Training completed successfully!\n")

# __name__ holds the module name of the current file or "__main__" if the
#  file is directly run
if __name__ == "__main__":
    main()