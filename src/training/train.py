"""
Fine-tuning Pipeline for Transformer-based Grammatical Error Detection & Correction (GEC/GED).
Supports training T5/BART architectures using HuggingFace Seq2SeqTrainer or PyTorch.
"""
import os
import argparse
import json
from typing import List, Tuple

from ..model.config import TrainingConfig
from ..data.synthetic_noise import SyntheticErrorGenerator
from ..data.dataset import GrammarDataset


SAMPLE_SEED_SENTENCES = [
    "She goes to school every morning.",
    "Yesterday I went to the supermarket and bought some fresh apples.",
    "He does not have any interest in playing football.",
    "They were very happy with their exam results.",
    "An honest person always tells the truth.",
    "She has lived in London for five years.",
    "The students are studying hard for their final examinations.",
    "He walked into the room and closed the door behind him.",
    "We are planning to visit our grandparents next weekend.",
    "The book on the table belongs to my sister."
]


def generate_synthetic_corpus(num_samples: int = 100) -> List[Tuple[str, str]]:
    """Generates synthetic parallel (erroneous, clean) sentence pairs."""
    generator = SyntheticErrorGenerator()
    pairs = []
    
    for i in range(num_samples):
        clean = SAMPLE_SEED_SENTENCES[i % len(SAMPLE_SEED_SENTENCES)]
        noisy, _ = generator.perturb_sentence(clean)
        pairs.append((noisy, clean))
        
    return pairs


def train(args):
    """Executes the model training/fine-tuning loop."""
    print("=" * 60)
    print("  Deep Learning Grammatical Error Detection (GED) Training")
    print("=" * 60)

    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
            DataCollatorForSeq2Seq
        )
    except ImportError:
        print("[Error] PyTorch or HuggingFace Transformers is not installed.")
        print("Please install via: pip install torch transformers")
        return

    config = TrainingConfig(
        base_model=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print(f"Loading base model & tokenizer: {config.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)

    # Prepare dataset
    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading custom dataset from: {args.dataset}")
        with open(args.dataset, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = [(d["erroneous"], d["corrected"]) for d in data]
    else:
        print(f"Generating synthetic training corpus ({args.synthetic_samples} samples)...")
        pairs = generate_synthetic_corpus(num_samples=args.synthetic_samples)

    split_idx = int(len(pairs) * 0.85)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Training samples: {len(train_pairs)} | Validation samples: {len(val_pairs)}")

    train_dataset = GrammarDataset(train_pairs, tokenizer=tokenizer)
    val_dataset = GrammarDataset(val_pairs, tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        save_total_limit=config.save_total_limit,
        num_train_epochs=config.num_epochs,
        predict_with_generate=True,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting model fine-tuning...")
    trainer.train()

    print(f"\nTraining complete! Saving final model weights to: {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Model ready for inference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Deep Learning model for Grammatical Error Detection")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Pretrained base model (e.g., t5-small, google/flan-t5-base)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON dataset with [{'erroneous': '...', 'corrected': '...'}, ...]")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--synthetic_samples", type=int, default=60, help="Number of synthetic pairs if no dataset given")

    args = parser.parse_args()
    train(args)
