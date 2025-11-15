# src/preprocess.py
"""Dataset loading & tokenisation pipeline."""

from __future__ import annotations

from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, default_data_collator


class Preprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._build()

    def _build(self) -> None:
        name = self.cfg.dataset.name
        config_name = self.cfg.dataset.get("config", None)
        cache_dir = ".cache/"
        if config_name:
            ds_train = load_dataset(name, config_name, split=self.cfg.dataset.split, cache_dir=cache_dir)
            ds_eval = load_dataset(name, config_name, split=self.cfg.dataset.get("eval_split", "validation"), cache_dir=cache_dir)
        else:
            ds_train = load_dataset(name, split=self.cfg.dataset.split, cache_dir=cache_dir)
            ds_eval = load_dataset(name, split=self.cfg.dataset.get("eval_split", "validation"), cache_dir=cache_dir)

        if self.cfg.mode == "trial":  # tiny subset
            ds_train = ds_train.shuffle(seed=42).select(range(8))
            ds_eval = ds_eval.shuffle(seed=42).select(range(8))

        def fmt(ex):
            if name.lower().startswith("gsm"):
                ex["text"] = f"Question: {ex['question'].strip()}\n\nAnswer: {ex['answer'].strip()}"
            else:
                ex["text"] = ex.get("text") or ex.get("content") or str(ex)
            return ex

        ds_train = ds_train.map(fmt, num_proc=4)
        ds_eval = ds_eval.map(fmt, num_proc=4)

        def tok(example):
            out = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.cfg.dataset.preprocessing.max_length,
            )
            # Copy input_ids to labels, but replace padding tokens with -100
            labels = out["input_ids"].copy()
            # Replace padding token IDs with -100 so they're ignored in loss calculation
            labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]
            out["labels"] = labels
            return out

        remove_cols = ds_train.column_names
        ds_train = ds_train.map(tok, remove_columns=remove_cols, num_proc=4)
        ds_eval = ds_eval.map(tok, remove_columns=remove_cols, num_proc=4)

        self.ds_train = ds_train
        self.ds_eval = ds_eval

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        bs = self.cfg.training.per_device_batch_size
        # Since we're padding to max_length during tokenization, use default_data_collator
        # which simply converts lists to tensors without additional padding
        collator = default_data_collator
        dl_train = DataLoader(self.ds_train, batch_size=bs, shuffle=True, drop_last=True, collate_fn=collator)
        dl_eval = DataLoader(self.ds_eval, batch_size=bs, shuffle=False, collate_fn=collator)
        return dl_train, dl_eval