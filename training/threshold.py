import numpy as np # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix # type: ignore

probs = np.load("val_probs.npy")
labels = np.load("val_labels.npy")
best_threshold = 0.5
best_recall = 0.0
best_metrics = None
print("\nThreshold Sweep (Minimize FN Strategy)")
print("="*60)

candidates = []
for t in np.arange(0.05, 0.96, 0.01):
    preds = (probs > t).astype(int)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    if recall >= 0.90 and precision >= 0.80:
        candidates.append((t, precision, recall, f1, fn))
print("\nCandidate Thresholds (Recall ≥ 0.90 & Precision ≥ 0.80):\n")
for c in candidates:
    print(f"T={c[0]:.2f} | Precision={c[1]:.4f} | Recall={c[2]:.4f} | F1={c[3]:.4f} | FN={c[4]}")