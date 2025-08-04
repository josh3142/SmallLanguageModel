import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from training.trainer import Trainer

def set_seed(seed: int | None=42) -> None:
    if seed is None:
        print("No seed was chosen.")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)             # CPU seed
        torch.cuda.manual_seed(seed)        # CUDA seed for current GPU
        torch.cuda.manual_seed_all(seed)    # CUDA seed for all GPUs
        
        # PyTorch deterministic options
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExperimentManager:
    """Handles experiment directory creation and file management."""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.experiment_name = None
        self.experiment_dir = None
    
    def setup_experiment(self, description: str = None, resume_from: str = None) -> str:
        """
        Setup experiment directory. Returns experiment directory path.
        """
        if resume_from is not None:
            self.experiment_dir = os.path.join(self.base_dir, description)
            self.experiment_name = description  # Add this line
            if not os.path.exists(os.path.join(self.experiment_dir, resume_from)):  # Fix the check
                print("Checkpoint not found.")
                print("Start script with `resume_from_ckpt=None` or provide proper checkpoint")
                sys.exit()
        else:
            self.experiment_name = self._generate_new_experiment_name(description)
            self.experiment_dir = os.path.join(self.base_dir, self.experiment_name)
            os.makedirs(self.experiment_dir, exist_ok=True)
           
        return self.experiment_dir
    
    def get_checkpoint(self, ckpt_name: str) -> str:
        """Get path for checkpoint file."""
        return os.path.join(self.experiment_dir, ckpt_name)
    
    def get_plot_path(self, filename: str = "training_curves.png") -> str:
        """Get path for plot file."""
        return os.path.join(self.experiment_dir, filename)
    
    def _generate_new_experiment_name(self, description: str = None) -> str:
        """Generate new experiment name with timestamp."""
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        if description:
            clean_desc = "".join(c if c.isalnum() else "_" for c in description).strip("_")
            return f"{timestamp}_{clean_desc}"
        return timestamp
    

def plot_loss(trainer: Trainer, path: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if trainer.val_losses:
        plt.subplot(1, 2, 2)
        plt.plot(trainer.val_losses, label='Validation Loss')
        plt.title('Validation Loss')
        plt.xlabel('Evaluation Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.show()