import os, json, math
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.models.baseline_classifier import BaselineClassifier
from src.dataio.dataset import NewsDataset
from src.models.bayes_classifier import AuthorBayesClassifier
from src.utils.seed import set_seed

import pandas as pd

def load_rows_from_csv(path):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "author": row["author"],
            "text": row["text"],
            "label": int(row["is_fake"]),
            "rating": float(row["rating"]),
            "position": row["position"],
            "style_score": float(row.get("style_score", 0.5)),  # нове поле
            "soft_fake": float(row["soft_fake"]),
            "binary_label": 1 if row["soft_fake"] >= 0.5 else 0,
        })
    return rows

# def load_rows_from_csv(csv_path: str):
#     df = pd.read_csv(csv_path)
#
#     rows = []
#     for _, row in df.iterrows():
#         author = str(row["author"])
#         text   = str(row["text"])
#         rating = float(row["rating"])
#         position = str(row["position"])
#         is_fake = int(row["is_fake"])   # 1=fake, 0=real
#
#         # формуємо "збагачений" текст,
#         # щоб модель одразу бачила автора/посаду/рейтинг:
#         full_text = (
#             f"Автор: {author}. Посада: {position}. "
#             f"Рейтинг автора: {rating}. Текст новини: {text}"
#         )
#
#         rows.append({
#             "text": full_text,
#             "author": author,
#             "position": position,
#             "rating": rating,
#             "label": is_fake      # 0=real, 1=fake — як ми й хочемо
#         })
#     return rows


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    beta_author_reg: float = 0.1,   # сила регуляризатора
):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        # y   = batch["label"].to(device)          # [B]
        y = batch["soft_fake"].to(device)
        rating = batch["rating"].to(device)      # [B]
        author_prior = batch["author_prior"].to(device)
        authors = batch["author"]               # список рядків довжини B
        # якщо вже буде style_score -> додамо трохи нижче
        style_score = batch["style_score"].to(device)

        logits = model(
            ids,
            attn,
            rating=rating,
            author_prior=author_prior,
            style_score=style_score,
        )

        # logits = model(ids, attn, rating=rating, author_prior=author_prior)  # [B]
        y_soft = batch["soft_fake"].to(device)
        bce_loss = crit(logits, y_soft)
        # bce_loss = crit(logits, y)

        # ---- author-level consistency regularizer ----
        with torch.no_grad():
            probs_fake = torch.sigmoid(logits)   # [B]

        author_to_indices: dict[str, list[int]] = {}
        for i, a in enumerate(authors):
            author_to_indices.setdefault(a, []).append(i)

        reg_terms = []
        for a, idxs in author_to_indices.items():
            idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)

            # середня ймовірність "fake" для цього автора в батчі
            mean_fake = probs_fake.index_select(0, idxs_t).mean()

            # prior на fake = 1 - rating (беремо rating з будь-якого прикладу автора)
            prior_fake = (1.0 - rating[idxs_t[0]]).clamp(0.0, 1.0)

            reg_terms.append((mean_fake - prior_fake) ** 2)

        if reg_terms:
            author_reg = torch.stack(reg_terms).mean()
        else:
            author_reg = torch.tensor(0.0, device=device)

        loss = bce_loss + beta_author_reg * author_reg
        # ----------------------------------------------

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * ids.size(0)

    return total_loss / len(loader.dataset)


# def train_one_epoch(model, loader, optimizer, scheduler, device):
#     model.train()
#     crit = nn.BCEWithLogitsLoss()
#     total_loss = 0.0
#
#     for batch in tqdm(loader, desc="train", leave=False):
#         ids = batch["input_ids"].to(device)
#         attn = batch["attention_mask"].to(device)
#         y   = batch["label"].to(device)
#         rating = batch["rating"].to(device)
#         author_prior = batch["author_prior"].to(device)
#
#         logits = model(ids, attn, rating=rating, author_prior=author_prior)
#         loss = crit(logits, y)
#
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         if scheduler:
#             scheduler.step()
#
#         total_loss += loss.item() * ids.size(0)
#
#     return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    probs_list = []
    y_bin_list = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        rating = batch["rating"].to(device)
        author_prior = batch["author_prior"].to(device)
        style_score = batch["style_score"].to(device)

        y_soft = batch["soft_fake"].to(device)
        y_bin = batch["binary_label"].cpu().numpy()

        logits = model(ids, attn, rating=rating, author_prior=author_prior, style_score=style_score)
        loss = crit(logits, y_soft)

        total_loss += loss.item() * ids.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        probs_list.append(probs)
        y_bin_list.append(y_bin)

    probs = np.concatenate(probs_list)
    y_bin = np.concatenate(y_bin_list)

    return total_loss / len(loader.dataset), probs, y_bin


def compute_author_priors(rows, alpha: float = 1.0, beta: float = 1.0):
    """
    rows: список словників з полями "author" і "label" (0=real, 1=fake)
    Повертає dict: author -> prior (ймовірність, що автор пише правду).
    """
    stats = {}  # author -> [n_real, n_fake]
    for r in rows:
        a = r["author"]
        if a not in stats:
            stats[a] = [0, 0]
        if r["label"] == 0:   # 0 = real
            stats[a][0] += 1
        else:                 # 1 = fake
            stats[a][1] += 1

    priors = {}
    for a, (n_real, n_fake) in stats.items():
        prior = (n_real + alpha) / (n_real + n_fake + alpha + beta)
        priors[a] = float(prior)
    return priors



def main2():
    try:
        with open("configs/baseline.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        set_seed(cfg["random_seed"])
        device = cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"

        # --- дані
        csv_path = "data/news_with_authors_liar_with_style.csv"

        rows = load_rows_from_csv(csv_path)
        y = np.array([r["label"] for r in rows])
        x = np.arange(len(rows))  # просто індекси, бо тексти в rows

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["random_seed"])
        fold = 0
        scores = []

        # author_priors = compute_author_priors(rows)

        try:
            for train_idx, val_idx in skf.split(x, y):
                fold += 1
                print(f"\n===== Fold {fold} =====")

                train_rows = [rows[i] for i in train_idx]
                val_rows = [rows[i] for i in val_idx]

                # рахуємо пріори ТІЛЬКИ по train_rows
                author_priors = compute_author_priors(train_rows)

                train_ds = NewsDataset(
                    train_rows,
                    plm_name=cfg["plm_name"],
                    max_len=cfg["max_len"],
                    author_priors=author_priors
                )
                val_ds = NewsDataset(
                    val_rows,
                    plm_name=cfg["plm_name"],
                    max_len=cfg["max_len"],
                    author_priors=author_priors  # на валідації використовуємо ті ж пріори
                )

                train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

                # model = BaselineClassifier(
                #     plm_name=cfg["plm_name"],
                #     use_rating=True,
                #     use_author_prior=True
                # ).to(device)
                model = AuthorBayesClassifier(
                    plm_name=cfg["plm_name"],
                    d_text=768,
                    lambda_prior=1.0,
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                total_steps = len(train_loader) * cfg["epochs"]
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

                # --- цикл епох (усередині фолду)
                best_f1 = -1.0
                for epoch in range(cfg["epochs"]):
                    tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
                    va_loss, va_probs, va_labels = evaluate(model, val_loader, device)
                    from src.utils.metrics import metrics_report

                    mr = metrics_report(va_labels, va_probs, thr=0.5)
                    print(f"Fold {fold} Epoch {epoch + 1}: AUC={mr['auc']:.4f}  F1={mr['f1_macro']:.4f} tr_loss: {tr_loss}")
                    if mr["f1_macro"] > best_f1:
                        best_f1 = mr["f1_macro"]
                scores.append(best_f1)
        finally:
            torch.cuda.empty_cache()
            del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        print("\n==== Mean F1 across folds:", np.mean(scores), "====")
    finally:
        # del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        pass


if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        main2()
    finally:
        torch.cuda.empty_cache()
