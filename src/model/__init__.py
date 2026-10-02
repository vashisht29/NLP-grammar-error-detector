"""
Model package exports.
"""
from .config import ModelConfig, TrainingConfig
from .classifier import ErrorClassifier
from .detector import DeepGrammarDetector, GrammarError, DetectionResult

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "ErrorClassifier",
    "DeepGrammarDetector",
    "GrammarError",
    "DetectionResult",
]
