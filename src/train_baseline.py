import json

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

from src.models.baseline_classifier import BaselineClassifier
from src.dataio.dataset import NewsDataset
from src.models.bayes_classifier import AuthorBayesClassifier
from src.utils.seed import set_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

def load_rows_from_csv(path):
    df = pd.read_csv(path)
    rows = []
    for _, row in df[:5000].iterrows():
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
    beta_author_reg: float = 0.2,   # сила регуляризатора
):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    lambda_contrast = 0.1

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

        y_bin = batch["label"].to(device)
        # 2) Контрастивна частина (на ембеддингах тексту)
        emb = model.encode(ids, attn)  # [B, d]
        loss_con = supervised_contrastive_loss(emb, y_bin)

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

        loss = bce_loss + beta_author_reg * author_reg + loss_con * lambda_contrast
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


def supervised_contrastive_loss(embeddings, labels, temperature: float = 0.07):
    """
    Supervised contrastive loss (Khosla et al., 2020) для батчу.

    embeddings: [B, d]  - ембеддинги новин (після encoder)
    labels:     [B]     - 0 / 1

    Ідея: для кожного елемента i
      - позитиви: j з тією ж міткою
      - негативи: j з іншою міткою
    """
    device = embeddings.device
    z = F.normalize(embeddings, p=2, dim=1)  # нормалізація
    labels = labels.view(-1)

    batch_size = z.size(0)
    if batch_size < 2:
        return embeddings.new_tensor(0.0)

    # Матриця подібності cos(z_i, z_j) / T
    logits = torch.matmul(z, z.T) / temperature  # [B, B]

    # Маска позитивів: однакові лейбли
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
    mask_pos = labels_eq.float()

    # Вимикаємо self-contrast
    logits_mask = torch.ones_like(mask_pos, device=device) - torch.eye(batch_size, device=device)
    mask_pos = mask_pos * logits_mask

    # Nominator / denominator для InfoNCE
    exp_logits = torch.exp(logits) * logits_mask  # [B,B]
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # Середнє log_prob по позитивних парах
    num_pos = mask_pos.sum(dim=1)  # [B]
    valid = num_pos > 0

    if not valid.any():
        # У батчі всі з різним класом або всі з одним класом
        return embeddings.new_tensor(0.0)

    loss = -(mask_pos[valid] * log_prob[valid]).sum(dim=1) / num_pos[valid]
    return loss.mean()


def main2():
    try:
        with open("configs/baseline.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        xpoints = np.array([1, 8])
        ypoints = np.array([3, 10])

        plt.plot(xpoints, ypoints)
        plt.savefig("test.png", dpi=300)
        # plt.show()

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
                    # d_text=1024,
                    lambda_prior=1.0,
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                total_steps = len(train_loader) * cfg["epochs"]
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

                # --- цикл епох (усередині фолду)
                best_f1 = -1.0
                first_fold_curves = None  # сюди складемо train/val loss для 1-го фолду
                tr_history = []
                va_history = []
                for epoch in range(cfg["epochs"]):
                    tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
                    va_loss, va_probs, va_labels = evaluate(model, val_loader, device)
                    from src.utils.metrics import metrics_report, error_stats_by_class

                    mr = metrics_report(va_labels, va_probs, thr=0.5)

                    # зберігаємо втрати для графіка
                    tr_history.append(float(tr_loss))
                    va_history.append(float(va_loss))

                    # "reconstruction-подібні" метрики
                    err_stats = error_stats_by_class(va_labels, va_probs)

                    print(f"Fold {fold} Epoch {epoch + 1}: AUC={mr['auc']:.4f}  F1={mr['f1_macro']:.4f} tr_loss: {tr_loss} va_loss: {va_loss}")

                    if err_stats["real"] is not None and err_stats["fake"] is not None:
                        r = err_stats["real"]
                        f = err_stats["fake"]
                        ratio_mean = f["mean"] / max(r["mean"], 1e-8)

                        print(
                            "  Error stats (real): "
                            f"mean={r['mean']:.4f}, median={r['median']:.4f}, "
                            f"std={r['std']:.4f}, min={r['min']:.4f}, max={r['max']:.4f}"
                        )
                        print(
                            "  Error stats (fake): "
                            f"mean={f['mean']:.4f}, median={f['median']:.4f}, "
                            f"std={f['std']:.4f}, min={f['min']:.4f}, max={f['max']:.4f}"
                        )
                        # print(
                        #     f"  Інтерпретація: середня похибка для фейкових новин "
                        #     f"({f['mean']:.4f}) у ~{ratio_mean:.1f} раз(и) більша, "
                        #     f"ніж для реальних ({r['mean']:.4f}), що свідчить про "
                        #     f"краще розрізнення класів моделлю."
                        # )
                    else:
                        pass
                        # print("  Інтерпретація: у валідаційному наборі один з класів відсутній.")

                    if mr["f1_macro"] > best_f1:
                        best_f1 = mr["f1_macro"]
                # збережемо криву для першого фолду, щоб намалювати графік
                # if fold == 1:
                first_fold_curves = {
                    "train_loss": tr_history,
                    "val_loss": va_history,
                }
                scores.append(best_f1)
                # Графік ефективності навчання (для 1-го фолду)
                if first_fold_curves is not None:
                    epochs = range(1, len(first_fold_curves["train_loss"]) + 1)
                    plt.figure()
                    plt.plot(epochs, first_fold_curves["train_loss"], label="Training Loss")
                    plt.plot(epochs, first_fold_curves["val_loss"], label="Validation Loss")
                    plt.xlabel("Epochs")
                    plt.ylabel("Loss")
                    plt.title("Графік ефективності навчання моделі (Fold 1)")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"fold_{fold}.png", dpi=300)
                    # plt.show()
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
