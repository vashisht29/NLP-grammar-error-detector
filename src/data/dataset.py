"""
Dataset utilities for Grammatical Error Detection (GED) and Correction (GEC).
Provides PyTorch Dataset implementations for parallel noisy/clean sentence pairs.
"""
import json
from typing import List, Tuple, Dict, Any, Optional

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Graceful stub if torch is not installed
    class Dataset:
        pass


class GrammarDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning Sequence-to-Sequence models on GEC tasks.
    Pairs (erroneous_sentence, corrected_sentence).
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: Any = None,
        max_length: int = 128,
        prefix: str = "gec: "
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src_text, tgt_text = self.pairs[idx]
        
        if self.tokenizer is None:
            return {"source": src_text, "target": tgt_text}

        input_text = f"{self.prefix}{src_text}"
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        label_ids = labels["input_ids"].squeeze(0)
        # Replace padding token id with -100 to ignore during loss computation
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label_ids
        }

    @classmethod
    def from_json(cls, filepath: str, **kwargs) -> "GrammarDataset":
        """Loads dataset from JSON file containing [{'erroneous': '...', 'corrected': '...'}, ...]."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = [(item["erroneous"], item["corrected"]) for item in data]
        return cls(pairs=pairs, **kwargs)
