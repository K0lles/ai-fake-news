# датасет + токенізація
from typing import List, Dict, Any
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class NewsDataset(Dataset):
    def __init__(self, rows, plm_name: str, max_len: int = 384, author_priors: dict | None = None):
        self.rows = rows
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name, use_fast=True)
        self.max_len = max_len
        self.author_priors = author_priors or {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]

        enc = self.tokenizer(
            r["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        author = r["author"]
        prior = self.author_priors.get(author, 0.5)  # якщо автора ще не бачили → 0.5

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(r["label"], dtype=torch.float32),
            "rating": torch.tensor(r["rating"], dtype=torch.float32),
            "author_prior": torch.tensor(prior, dtype=torch.float32),
            "author": author,
            "position": r["position"],
        }

