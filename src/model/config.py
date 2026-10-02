"""
Configuration settings for the Deep Learning Grammatical Error Detection (GED) System.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    """Configuration for Deep Learning Model inference and training."""
    
    # Pretrained Hugging Face model for English Grammatical Error Correction/Detection
    model_name: str = "prithivida/grammar_error_correcter_v1"
    fallback_model_name: str = "google/flan-t5-base"
    
    # Inference Device: auto-detects Apple Silicon MPS, NVIDIA CUDA, or CPU
    device: str = "auto"
    
    # Generation parameters
    max_length: int = 128
    num_beams: int = 4
    early_stopping: bool = True
    temperature: float = 1.0
    
    # Confidence & Detection Thresholds
    min_confidence_threshold: float = 0.65
    
    # Standard Grammatical Error Categories
    error_categories: List[str] = field(default_factory=lambda: [
        "Subject-Verb Agreement",
        "Verb Tense & Form",
        "Article / Determiner",
        "Preposition Error",
        "Spelling / Typo",
        "Punctuation",
        "Word Order",
        "Redundancy",
        "General Grammatical Error"
    ])
    
    def resolve_device(self) -> str:
        """Resolve device to 'mps', 'cuda', or 'cpu' dynamically."""
        if self.device != "auto":
            return self.device
            
        try:
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning on custom datasets."""
    base_model: str = "t5-small"
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_total_limit: int = 2
    logging_steps: int = 50
    eval_steps: int = 200
