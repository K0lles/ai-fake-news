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


def error_stats_by_class(y_true: np.ndarray, y_prob: np.ndarray):
    """
    "Reconstruction-подібна" метрика:
    беремо абсолютну похибку |y_true - y_prob|
    і рахуємо статистику окремо для real (0) і fake (1).
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    err = np.abs(y_true - y_prob)

    stats = {}
    for cls_val, cls_name in [(0.0, "real"), (1.0, "fake")]:
        mask = (y_true == cls_val)
        if mask.sum() == 0:
            stats[cls_name] = None
            continue

        e = err[mask]
        stats[cls_name] = {
            "count": int(mask.sum()),
            "mean": float(e.mean()),
            "median": float(np.median(e)),
            "std": float(e.std(ddof=0)),
            "min": float(e.min()),
            "max": float(e.max()),
        }
    return stats
