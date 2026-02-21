import torch # type: ignore
import numpy as np # type: ignore
from torch.utils.data import DataLoader # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from training.dataset import FFPPDataset
from models.hybrid_model import HybridModel


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = HybridModel(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = None
):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            fft = batch["fft"].to(device)
            labels = batch["label"].to(device)

            outputs = model(rgb, fft).squeeze(1)
            probabilities = torch.sigmoid(outputs)

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probabilities)

    if threshold is not None:
        y_pred = (y_prob > threshold).astype(float)
        return y_true, y_pred, y_prob
    return y_true, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    return metrics


def find_best_threshold(model, val_loader, device):
    y_true, y_prob = evaluate_model(model, val_loader, device, threshold=None)

    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.arange(0.1, 0.9, 0.01)

    for t in thresholds:
        y_pred = (y_prob > t).astype(float)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = "E:/Datasets/FFPP_Processed"
    checkpoint_path = "checkpoints/best_model.pth"

    val_dataset = FFPPDataset(root_dir=dataset_root, split="val")
    test_dataset = FFPPDataset(root_dir=dataset_root, split="test")

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = load_model(checkpoint_path, device)

    best_threshold, best_val_f1 = find_best_threshold(model, val_loader, device)

    print(f"Optimal threshold (validation): {best_threshold:.2f}")
    print(f"Validation F1 at optimal threshold: {best_val_f1:.4f}")

    y_true, y_pred, y_prob = evaluate_model(
        model,
        test_loader,
        device,
        threshold=best_threshold
    )

    metrics = compute_metrics(y_true, y_pred, y_prob)

    print("\nTest Set Performance")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC AUC  : {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_roc_curve(y_true, y_prob)


if __name__ == "__main__":
    main()