from sklearn.metrics import classification_report, roc_auc_score, f1_score
import numpy as np

def binarize_probs(probs, thr=0.5):
    return (probs >= thr).astype(int)

def metrics_report(y_true, y_prob, thr=0.5):
    y_pred = binarize_probs(y_prob, thr)
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    target_names = ["real", "fake"][:len(unique_classes)]

    report = classification_report(
        y_true, y_pred,
        labels=unique_classes,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    # AUC може впасти, якщо тільки один клас → ловимо виняток
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    f1 = f1_score(y_true, y_pred, average="macro")
    return {"report": report, "auc": auc, "f1_macro": f1}
