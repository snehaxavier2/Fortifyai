import os
import json
import random
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torch.cuda.amp import GradScaler, autocast # type: ignore
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve) # type: ignore
from models.hybrid_model import HybridModel
from training.dataset import MultiDomainDataset, BalancedDomainSampler
from preprocessing.config import SEED

# Reproducibility 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = True

# Configuration 
CHECKPOINT_DIR   = r"E:\Fortifyai\checkpoints"
BATCH_SIZE       = 24           
GRAD_ACCUM_STEPS = 2            
EPOCHS_STAGE1    = 15           
EPOCHS_STAGE2    = 85           
TOTAL_EPOCHS     = EPOCHS_STAGE1 + EPOCHS_STAGE2   
LR_STAGE1        = 3e-4
LR_STAGE2        = 5e-5
WEIGHT_DECAY     = 1e-4         
LABEL_SMOOTHING  = 0.05
NUM_WORKERS      = 2
UNFREEZE_BLOCKS  = 3


# Evaluation metrics

def compute_eer(labels: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def compute_all_metrics(labels: np.ndarray, probs: np.ndarray,
                        threshold: float = 0.5, domain_name: str = "") -> dict:
    preds = (probs > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    f1        = f1_score(labels, preds, average="macro", zero_division=0)
    auc       = roc_auc_score(labels, probs)
    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    spec      = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    eer       = compute_eer(labels, probs)

    metrics = {
        "domain":     domain_name,
        "f1":         round(float(f1),        4),
        "auc":        round(float(auc),       4),
        "accuracy":   round(float(acc),       4),
        "precision":  round(float(precision), 4),
        "recall":     round(float(recall),    4),
        "fpr":        round(float(fpr),       4),
        "fnr":        round(float(fnr),       4),
        "specificity":round(float(spec),      4),
        "eer":        round(float(eer),       4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "total": int(tp + tn + fp + fn),
    }

    if domain_name:
        _print_metrics(metrics)

    return metrics


def _print_metrics(m: dict) -> None:
    name = m["domain"] or "Overall"
    print(f"\n  ── {name} ──────────────────────────────────────")
    print(f"  Total      : {m['total']:>6}  "
          f"(TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']})")
    print(f"  F1 (macro) : {m['f1']:.4f}")
    print(f"  AUC-ROC    : {m['auc']:.4f}")
    print(f"  Accuracy   : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Precision  : {m['precision']:.4f}  "
          f"(of flagged fakes, {m['precision']*100:.1f}% were real fakes)")
    print(f"  Recall     : {m['recall']:.4f}  "
          f"(caught {m['recall']*100:.1f}% of all deepfakes)")
    print(f"  FPR        : {m['fpr']:.4f}  "
          f"({m['fpr']*100:.1f}% of real images wrongly flagged)")
    print(f"  FNR        : {m['fnr']:.4f}  "
          f"({m['fnr']*100:.1f}% of deepfakes missed  ← most critical)")
    print(f"  Specificity: {m['specificity']:.4f}")
    print(f"  EER        : {m['eer']:.4f}  "
          f"(lower is better, 0=perfect, 0.5=random)")


def _evaluate(model, loader, device, threshold=0.5) -> dict:
    """
    Evaluates model on a DataLoader.
    Returns full metrics dict including probs and labels arrays.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    metrics = compute_all_metrics(labels, probs, threshold)
    metrics["probs"]  = probs
    metrics["labels"] = labels
    return metrics


# Data loaders 

def _make_loader(split: str, batch_size: int) -> DataLoader:
    dataset = MultiDomainDataset(split)
    if split == "train":
        sampler = BalancedDomainSampler(dataset, batch_size=batch_size)
        return DataLoader(
            dataset,
            batch_sampler       = sampler,
            num_workers         = NUM_WORKERS,
            pin_memory          = True,
            persistent_workers  = True
        )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = True
    )


# Label smoothing loss 

class LabelSmoothBCE(nn.Module):
    """BCE with label smoothing"""
    def __init__(self, smoothing: float = 0.05, pos_weight: float = 1.2):
        super().__init__()
        self.smoothing   = smoothing
        self.pos_weight  = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        pw = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=pw
        )


# Single epoch 

def _run_epoch(epoch, model, train_loader, val_loader,
               optimizer, scheduler, criterion, scaler,
               device, history, best_f1, best_auc,
               checkpoint_dir, grad_accum_steps) -> tuple:

    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss   = criterion(logits, labels) / grad_accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    val = _evaluate(model, val_loader, device)
    f1  = val["f1"]
    auc = val["auc"]

    print(f"\n  Epoch {epoch:>3}/{TOTAL_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"F1: {f1:.4f} | AUC: {auc:.4f} | "
          f"Prec: {val['precision']:.4f} | Rec: {val['recall']:.4f} | "
          f"FNR: {val['fnr']:.4f} | EER: {val['eer']:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    history.append({
        "epoch":      epoch,
        "loss":       round(avg_loss, 6),
        "val_f1":     val["f1"],
        "val_auc":    val["auc"],
        "val_acc":    val["accuracy"],
        "val_prec":   val["precision"],
        "val_recall": val["recall"],
        "val_fpr":    val["fpr"],
        "val_fnr":    val["fnr"],
        "val_eer":    val["eer"],
        "val_tp":     val["tp"],
        "val_tn":     val["tn"],
        "val_fp":     val["fp"],
        "val_fn":     val["fn"],
    })

    # Save best checkpoint 
    if f1 > best_f1 or (f1 == best_f1 and auc > best_auc):
        best_f1  = max(f1, best_f1)
        best_auc = max(auc, best_auc)
        path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "best_f1":     best_f1,
            "best_auc":    best_auc,
            "val_metrics": {k: v for k, v in val.items()
                            if k not in ("probs", "labels")},
        }, path)
        print(f"    ✓ Saved best → F1={best_f1:.4f} | "
              f"AUC={best_auc:.4f} | FNR={val['fnr']:.4f} | "
              f"EER={val['eer']:.4f}")

    model.train()
    return best_f1, best_auc


# Main training loop 

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f" FortifyAI v5 — Training")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Batch      : {BATCH_SIZE} × accum {GRAD_ACCUM_STEPS} "
          f"= {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print(f"  Resolution : 224×224")
    print(f"  LR Stage1  : {LR_STAGE1}")
    print(f"  LR Stage2  : {LR_STAGE2}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Label Smooth: {LABEL_SMOOTHING}")

    train_loader = _make_loader("train", BATCH_SIZE)
    val_loader   = _make_loader("val",   BATCH_SIZE * 2)

    model     = HybridModel(pretrained=True).to(device)
    model.enable_gradient_checkpointing()
    criterion = LabelSmoothBCE(smoothing=LABEL_SMOOTHING, pos_weight=1.2)
    scaler    = GradScaler()
    history   = []
    best_f1   = 0.0
    best_auc  = 0.0
    print(f"\n{'='*60}")
    print(f" STAGE 1 — Frozen backbone ({EPOCHS_STAGE1} epochs)")
    print(f"{'='*60}")

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6
    )

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        best_f1, best_auc = _run_epoch(
            epoch, model, train_loader, val_loader,
            optimizer, scheduler, criterion, scaler,
            device, history, best_f1, best_auc,
            CHECKPOINT_DIR, GRAD_ACCUM_STEPS
        )

    # STAGE 2 — Unfreeze last 3 blocks + head 
    print(f"\n{'='*60}")
    print(f" STAGE 2 — Unfreeze last {UNFREEZE_BLOCKS} blocks "
          f"({EPOCHS_STAGE2} epochs)")
    print(f"{'='*60}")

    model.unfreeze_last_blocks(n=UNFREEZE_BLOCKS)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE2,
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(EPOCHS_STAGE1 + 1, TOTAL_EPOCHS + 1):
        best_f1, best_auc = _run_epoch(
            epoch, model, train_loader, val_loader,
            optimizer, scheduler, criterion, scaler,
            device, history, best_f1, best_auc,
            CHECKPOINT_DIR, GRAD_ACCUM_STEPS
        )

    # Save full training history 
    history_path = os.path.join(CHECKPOINT_DIR, "training_history_v5.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f" Training Complete — FortifyAI v5")
    print(f"{'='*60}")
    print(f"  Best Val F1   : {best_f1:.4f}")
    print(f"  Best Val AUC  : {best_auc:.4f}")
    print(f"  History saved : {history_path}")
    print(f"\n  Next steps:")
    print(f"  1. Run: python -m training.tune_threshold")
    print(f"  2. Update THRESHOLD in predictor/gradcam.py")
    print(f"  3. Run: python -m training.evaluate")


if __name__ == "__main__":
    train()