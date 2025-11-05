import os, json, math
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

from src.models.baseline import BaselineClassifier
from src.dataio.dataset import NewsDataset
from src.utils.seed import set_seed
from src.utils.metrics import metrics_report

import pandas as pd

def load_rows_from_csv(csv_path: str):
    """
    Очікуваний формат CSV: колонки 'text','label'
    label: 0=real, 1=fake
    """
    df = pd.read_csv(csv_path)
    rows = [{"text": t, "label": int(l)} for t, l in zip(df["text"], df["label"])]
    return rows

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        logits = model(ids, attn)
        loss = crit(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler: scheduler.step()

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

        logits = model(ids, attn)
        loss = crit(logits, y)
        total_loss += loss.item() * ids.size(0)

        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(y.cpu().numpy())

    probs = torch.tensor([x for b in probs for x in b]).numpy()
    labels = torch.tensor([x for b in labels for x in b]).numpy()
    return total_loss / len(loader.dataset), probs, labels

def main():
    # --- конфіг
    with open("configs/baseline.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(cfg["random_seed"])
    device = cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"

    # --- дані
    csv_path = "data/samples.csv"
    rows = load_rows_from_csv(csv_path)
    dataset = NewsDataset(rows, plm_name=cfg["plm_name"], max_len=cfg["max_len"])

    val_size = int(len(dataset) * cfg["train_val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg["random_seed"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # --- модель
    model = BaselineClassifier(plm_name=cfg["plm_name"]).to(device)

    # --- оптимізація
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=total_steps
    )

    # --- тренування
    best_f1 = -1.0
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["epochs"]):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        va_loss, va_probs, va_labels = evaluate(model, val_loader, device)
        from src.utils.metrics import metrics_report
        mr = metrics_report(va_labels, va_probs, thr=0.5)

        print(f"\nEpoch {epoch+1}/{cfg['epochs']}:")
        print(f"  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(f"  AUC={mr['auc']:.4f}  F1_macro={mr['f1_macro']:.4f}")
        print(mr["report"])

        # збереження чекпойнта за F1
        if mr["f1_macro"] > best_f1:
            best_f1 = mr["f1_macro"]
            ckpt = os.path.join(cfg["save_dir"], "baseline_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  saved: {ckpt}")

if __name__ == "__main__":
    main()
