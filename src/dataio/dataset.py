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

        row = self.rows[i]
        text = row["text"]
        rating = row["rating"]
        style_score = row.get("style_score", 0.5)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        author_prior = self.author_priors.get(row["author"], 0.5)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "binary_label": torch.tensor(row["binary_label"], dtype=torch.float32),
            "soft_fake": torch.tensor(row["soft_fake"], dtype=torch.float32),
            "rating": torch.tensor(row["rating"], dtype=torch.float32),
            "author_prior": torch.tensor(author_prior, dtype=torch.float32),
            "style_score": torch.tensor(row["style_score"], dtype=torch.float32),
            "author": row["author"],
        }

        # return {
        #     "input_ids": enc["input_ids"].squeeze(0),
        #     "attention_mask": enc["attention_mask"].squeeze(0),
        #     "label": torch.tensor(row["label"], dtype=torch.float32),
        #     "rating": torch.tensor(rating, dtype=torch.float32),
        #     "author_prior": torch.tensor(row["rating"], dtype=torch.float32),  # якщо так було
        #     "author": row["author"],
        #     "style_score": torch.tensor(style_score, dtype=torch.float32),
        #     "soft_fake": torch.tensor(row["soft_fake"], dtype=torch.float32),
        #     "binary_label": torch.tensor(row["binary_label"], dtype=torch.float32),
        # }

        # enc = self.tokenizer(
        #     r["text"],
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_len,
        #     return_tensors="pt"
        # )
        #
        # author = r["author"]
        # prior = self.author_priors.get(author, 0.5)  # якщо автора ще не бачили → 0.5
        #
        # return {
        #     "input_ids": enc["input_ids"].squeeze(0),
        #     "attention_mask": enc["attention_mask"].squeeze(0),
        #     "label": torch.tensor(r["label"], dtype=torch.float32),
        #     "rating": torch.tensor(r["rating"], dtype=torch.float32),
        #     "author_prior": torch.tensor(prior, dtype=torch.float32),
        #     "author": author,
        #     "position": r["position"],
        # }

