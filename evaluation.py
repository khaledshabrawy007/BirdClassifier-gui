import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: list = None) -> np.ndarray:

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    n = len(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[label_to_idx[true]][label_to_idx[pred]] += 1
    return cm


def overall_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    return cm.diagonal().sum() / total
