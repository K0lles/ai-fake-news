from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from src.models.encoders import NewsEncoder  # твій encoder


ROOT = Path(__file__).resolve().parents[1]
CSV_IN  = ROOT / "data" / "news_with_authors_liar.csv"
CSV_OUT = ROOT / "data" / "news_with_authors_liar_with_style.csv"
PLM_NAME = "xlm-roberta-base"
BATCH_SIZE = 32


class TextOnlyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def main():
    df = pd.read_csv(CSV_IN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(PLM_NAME)
    encoder = NewsEncoder(plm_name=PLM_NAME).to(device)
    encoder.eval()

    max_len = 128  # або візьми з конфіга

    ds = TextOnlyDataset(df, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_embs = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            # NewsEncoder повертає CLS-вектор: [B, d]
            emb = encoder(ids, attn)    # [B, d]
            all_embs.append(emb.cpu().numpy())

    all_embs = np.vstack(all_embs)  # [N, d]

    # додаємо до df
    df["__idx"] = np.arange(len(df))

    # 1) прототипи c_a для кожного автора за РЕАЛЬНИМИ висловлюваннями
    # is_fake == 0 -> real
    real_mask = df["is_fake"].values == 0
    authors = df["author"].values

    author_vec_sum = {}
    author_vec_cnt = {}

    for i in range(len(df)):
        if not real_mask[i]:
            continue
        a = authors[i]
        v = all_embs[i]
        author_vec_sum[a] = author_vec_sum.get(a, 0) + v
        author_vec_cnt[a] = author_vec_cnt.get(a, 0) + 1

    author_proto = {}
    for a, s in author_vec_sum.items():
        c = s / author_vec_cnt[a]
        # нормалізуємо прототип
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        author_proto[a] = c

    # 2) для кожного прикладу рахуємо cos-sim до прототипу автора
    style_scores = []
    for i in range(len(df)):
        a = authors[i]
        v = all_embs[i]
        if a in author_proto:
            c = author_proto[a]
            nv = np.linalg.norm(v)
            if nv > 0:
                v_norm = v / nv
                cos_sim = float(np.dot(v_norm, c))  # [-1,1]
                # нормалізуємо в [0,1]
                style = (cos_sim + 1.0) / 2.0
            else:
                style = 0.5
        else:
            # немає прототипу → нейтральна оцінка
            style = 0.5
        style_scores.append(style)

    df["style_score"] = style_scores
    df.drop(columns=["__idx"], inplace=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved style-score CSV to {CSV_OUT}")


if __name__ == "__main__":
    main()
