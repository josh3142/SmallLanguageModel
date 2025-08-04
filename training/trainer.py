import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from typing import  Optional
from dataclasses import dataclass
from tqdm import tqdm

from model.transformer import GPT2LanguageModel

logger = logging.getLogger(__name__)

@dataclass 
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 3e-4  # Standard for small models
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    epochs: int = 10
    warmup_steps: int = 1000
    eval_interval: int = 500
    save_interval: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = True  # PyTorch 2.0 compilation
    mixed_precision: bool = True  # Use AMP for faster and more accurate training
    max_examples: Optional[int] = None # Maximal samples used from the dataset to train
    seed: int = 42
    resume_from_ckpt: Optional[str] = None
    experiment_description: str = "baseline"

class Trainer:
    """
    Training class with
    - Mixed precision training
    - Gradient clipping
    - Proper evaluation
    """
    def __init__(self, model: GPT2LanguageModel, train_config: TrainingConfig):
        self.config = train_config
        self.device = torch.device(train_config.device)
        self.model = model.to(self.device)
        # Compile model for faster training
        if train_config.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with PyTorch 2.0...")
            self.model = torch.compile(self.model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            weight_decay=train_config.weight_decay
        )
        # Mixed precision scaler to prevent gradient underflow (scaling loss)
        self.mixed_precision = train_config.mixed_precision
        self.scaler = torch.amp.GradScaler() if train_config.mixed_precision else None
        # Training state
        self.step = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.epoch+1}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            loss = self._training_step(
                inputs, targets, is_mixed_precision=self.mixed_precision
            )
            total_loss += loss.item()
            self.step += 1
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
        if val_loader:
            val_loss = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            logger.info(f"Epoch {self.epoch}: Validation loss = {val_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.epoch += 1
        
        return avg_loss
    
    def _training_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        is_mixed_precision: bool = True
    ) -> torch.Tensor:
        """
        Performs a single training step.

        If `is_mixed_precision` is True, uses automatic mixed precision (AMP)
        with gradient scaling. Otherwise, runs in full FP32.

        Args:
            inputs (torch.Tensor): The input tensors.
            targets (torch.Tensor): The target tensors.
            is_mixed_precision (bool): Whether to use AMP.

        Returns:
            torch.Tensor: The computed loss.
        """
        
        with torch.amp.autocast(device_type="cuda", enabled=is_mixed_precision):
            logits = self.model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        self.optimizer.zero_grad()

        if is_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        return loss
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model w.r.t. cross entropy loss if target is predicted."""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            with torch.amp.autocast(device_type="cuda", enabled=self.mixed_precision):
                logits = self.model(inputs)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
        
        self.model.train()
        return total_loss / num_batches
    
    def save_checkpoint(self, filepath: str):
        """
        Saves the current training state to a file.

        This method creates a checkpoint dictionary. If the model is wrapped 
        (e.g., with `torch.nn.DataParallel`), it extracts the original model 
        before saving.
        """
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') \
            else self.model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': model_to_save.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")


    def load_checkpoint(self, filepath: str):
        """
        Loads a checkpoint and resumes training state.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        logger.info(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(
            filepath, map_location=self.device, weights_only=False
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        logger.info(f"Resumed from epoch {self.epoch}, step {self.step}")
