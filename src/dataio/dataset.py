# датасет + токенізація
from typing import List, Dict, Any
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class NewsDataset(Dataset):
    """
    Очікує список словників:
    {"text": str, "label": int (0=real, 1=fake)}
    Далі додамо pub_dt/domain для ViL, поки не треба.
    """
    def __init__(self, rows: List[Dict[str, Any]], plm_name: str, max_len: int = 384):
        self.rows = rows
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name, use_fast=True)
        self.max_len = max_len

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

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(r["label"], dtype=torch.float32),
            # додаткові поля:
            "rating": torch.tensor(r["rating"], dtype=torch.float32),
            "author": r["author"],
            "position": r["position"],
        }
        return item
