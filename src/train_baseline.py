import os, json, math
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.models.baseline import BaselineClassifier
from src.dataio.dataset import NewsDataset
from src.utils.seed import set_seed

import pandas as pd

def load_rows_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        author = str(row["author"])
        text   = str(row["text"])
        rating = float(row["rating"])
        position = str(row["position"])
        is_fake = int(row["is_fake"])   # 1=fake, 0=real

        # формуємо "збагачений" текст,
        # щоб модель одразу бачила автора/посаду/рейтинг:
        full_text = (
            f"Автор: {author}. Посада: {position}. "
            f"Рейтинг автора: {rating}. Текст новини: {text}"
        )

        rows.append({
            "text": full_text,
            "author": author,
            "position": position,
            "rating": rating,
            "label": is_fake      # 0=real, 1=fake — як ми й хочемо
        })
    return rows


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y   = batch["label"].to(device)
        rating = batch["rating"].to(device)
        author_prior = batch["author_prior"].to(device)

        logits = model(ids, attn, rating=rating, author_prior=author_prior)
        loss = crit(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * ids.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    probs, labels = [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y   = batch["label"].to(device)
        rating = batch["rating"].to(device)
        author_prior = batch["author_prior"].to(device)

        logits = model(ids, attn, rating=rating, author_prior=author_prior)
        loss = crit(logits, y)
        total_loss += loss.item() * ids.size(0)

        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(y.cpu().numpy())

    probs = torch.tensor([x for b in probs for x in b]).numpy()
    labels = torch.tensor([x for b in labels for x in b]).numpy()
    return total_loss / len(loader.dataset), probs, labels


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
        csv_path = "data/news_with_authors.csv"

        rows = load_rows_from_csv(csv_path)
        y = np.array([r["label"] for r in rows])
        x = np.arange(len(rows))  # просто індекси, бо тексти в rows

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["random_seed"])
        fold = 0
        scores = []

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

                model = BaselineClassifier(
                    plm_name=cfg["plm_name"],
                    use_rating=True,
                    use_author_prior=True
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

#
# def main():
#     # --- конфіг
#     with open("configs/baseline.json", "r", encoding="utf-8") as f:
#         cfg = json.load(f)
#
#     set_seed(cfg["random_seed"])
#     device = cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
#
#     # --- дані
#     csv_path = "data/samples.csv"
#     rows = load_rows_from_csv(csv_path)
#     dataset = NewsDataset(rows, plm_name=cfg["plm_name"], max_len=cfg["max_len"])
#
#     val_size = int(len(dataset) * cfg["train_val_split"])
#     # train_size = len(dataset) - val_size
#
#     # ...
#     rows = load_rows_from_csv(csv_path)
#
#     # labels як список для стратифікації
#     y = [r["label"] for r in rows]
#
#     train_rows, val_rows = train_test_split(
#         rows,
#         test_size=cfg["train_val_split"],  # 0.2 або 0.3
#         random_state=cfg["random_seed"],
#         stratify=y  # <<< головне: стратифікуємо за міткою
#     )
#
#     train_ds = NewsDataset(train_rows, plm_name=cfg["plm_name"], max_len=cfg["max_len"])
#     val_ds = NewsDataset(val_rows, plm_name=cfg["plm_name"], max_len=cfg["max_len"])
#     # train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg["random_seed"]))
#
#     train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
#     val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
#
#     # --- модель
#     model = BaselineClassifier(plm_name=cfg["plm_name"]).to(device)
#
#     # --- оптимізація
#     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
#
#     total_steps = len(train_loader) * cfg["epochs"]
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=cfg["warmup_steps"],
#         num_training_steps=total_steps
#     )
#
#     # --- тренування
#     best_f1 = -1.0
#     Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
#     from sklearn.model_selection import StratifiedKFold
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["random_seed"])
#     for epoch in range(cfg["epochs"]):
#         tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
#         # va_loss, va_probs, va_labels = evaluate(model, val_loader, device)
#
#         va_loss, va_probs, va_labels = evaluate(model, val_loader, device)
#
#         from src.utils.metrics import metrics_report
#
#         mr = metrics_report(va_labels, va_probs, thr=0.5)
#
#         # якщо у валідації один клас — просто повідомляємо і не оновлюємо best_f1
#         if len(set(va_labels)) < 2:
#             print("⚠️ Val split має лише один клас — AUC не визначений, F1 може бути неінформативним.")
#         else:
#             if mr["f1_macro"] > best_f1:
#                 best_f1 = mr["f1_macro"]
#                 ckpt = os.path.join(cfg["save_dir"], "baseline_best.pt")
#                 torch.save(model.state_dict(), ckpt)
#                 print(f"  saved: {ckpt}")
#
#         from src.utils.metrics import metrics_report
#         mr = metrics_report(va_labels, va_probs, thr=0.5)
#
#         print(f"\nEpoch {epoch+1}/{cfg['epochs']}:")
#         print(f"  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
#         print(f"  AUC={mr['auc']:.4f}  F1_macro={mr['f1_macro']:.4f}")
#         print(mr["report"])
#
#         # збереження чекпойнта за F1
#         if mr["f1_macro"] > best_f1:
#             best_f1 = mr["f1_macro"]
#             ckpt = os.path.join(cfg["save_dir"], "baseline_best.pt")
#             torch.save(model.state_dict(), ckpt)
#             print(f"  saved: {ckpt}")

if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        main2()
    finally:
        torch.cuda.empty_cache()
