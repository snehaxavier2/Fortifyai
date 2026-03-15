import os
import sys
import argparse
import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torch.amp import autocast # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix # type: ignore


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.hybrid_model import HybridModel
from training.dataset import SingleDomainDataset
from preprocessing.config import DEVICE, OUTPUT_FFPP, OUTPUT_CELEB, OUTPUT_GAN

DOMAIN_CONFIG = [
    ("FF++",     OUTPUT_FFPP),
    ("Celeb-DF", OUTPUT_CELEB),
    ("GAN",      OUTPUT_GAN),
]


def evaluate_domain(model, domain_name, root_dir, device, split="test", batch_size=24):
    dataset = SingleDomainDataset(root_dir, split, domain_name)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits_orig = model(images)
                logits_flip = model(torch.flip(images, dims=[3]))
                probs_orig = torch.sigmoid(logits_orig)
                probs_flip = torch.sigmoid(logits_flip)
                probs = ((probs_orig + probs_flip) / 2).squeeze(1)
            threshold = 0.5
            preds = (probs > threshold).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "domain":    domain_name,
        "total":     len(all_labels),
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "auc":       auc,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "probs":    all_probs,
        "labels":   all_labels
    }


def print_domain_result(r):
    print(f"\n  {'─'*50}")
    print(f"  Domain     : {r['domain']}")
    print(f"  Total      : {r['total']} (Real: {r['tn']+r['fp']} | Fake: {r['fn']+r['tp']})")
    print(f"  Accuracy   : {r['accuracy']:.4f}  ({r['accuracy']*100:.1f}%)")
    print(f"  Precision  : {r['precision']:.4f}")
    print(f"  Recall     : {r['recall']:.4f}")
    print(f"  F1 (macro) : {r['f1']:.4f}")
    print(f"  AUC-ROC    : {r['auc']:.4f}")
    print(f"  Confusion  : TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}")


def print_summary(results):
    f1s   = [r["f1"]       for r in results]
    aucs  = [r["auc"]      for r in results]
    accs  = [r["accuracy"] for r in results]
    print(f"\n{'='*60}")
    print(f" Aggregate Results")
    print(f"{'='*60}")
    print(f"  Mean Accuracy : {np.mean(accs):.4f} ({np.mean(accs)*100:.1f}%)")
    print(f"  Mean F1       : {np.mean(f1s):.4f}")
    print(f"  Mean AUC      : {np.mean(aucs):.4f}")
    print(f"\n{'='*60}")
    print(f" Cross-Domain Gap Analysis")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["f1"] * 40) + "░" * (40 - int(r["f1"] * 40))
        print(f"  {r['domain']:<12}: {r['f1']:.4f} |{bar}|")
    gap = max(f1s) - min(f1s)
    print(f"\n  F1 Range : {gap:.4f} (max - min)")
    if gap < 0.05:
        verdict = "EXCELLENT — model generalizes well across domains"
    elif gap < 0.10:
        verdict = "GOOD — minor domain variation, acceptable"
    elif gap < 0.20:
        verdict = "WARNING — moderate domain gap, possible shortcut learning"
    else:
        verdict = "CRITICAL — severe domain gap, model not generalizing"
    print(f"  Verdict  : {verdict}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
    )
    parser.add_argument("--split", default="test", choices=[ "test","val"])
    args = parser.parse_args()
    print("\n" + "=" * 60)
    print(" FortifyAI v4-Clean — Cross-Domain Evaluation")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {DEVICE}")

    if not os.path.exists(args.checkpoint):
        print(f"\n  ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    print("\n  Loading checkpoint...")
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE)
    model = HybridModel(pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Checkpoint epoch : {ckpt.get('epoch', 'unknown')}")
    print(f"  Training best F1 : {ckpt.get('best_f1', 0.0):.4f}")
    print(f"  Training best AUC: {ckpt.get('best_auc', 0.0):.4f}")
    print(f"\n{'='*60}")
    print(f" Per-Domain Evaluation [test split]")
    print(f"{'='*60}")
    results = []
    for domain_name, root_dir in DOMAIN_CONFIG:
        print(f"\n  Evaluating: {domain_name}...")
        r = evaluate_domain(model, domain_name, root_dir, DEVICE, split=args.split)
        print_domain_result(r)
        results.append(r)
    print_summary(results)
    if args.split == "val":
        if "probs" in results[0] and "labels" in results[0]:
            all_probs = np.concatenate([r["probs"] for r in results])
            all_labels = np.concatenate([r["labels"] for r in results])
        np.save("val_probs.npy", all_probs)
        np.save("val_labels.npy", all_labels)
        print("\nSaved validation probabilities:")
        print("  val_probs.npy")
        print("  val_labels.npy")
        print(f"  Total samples saved: {len(all_probs)}")
    else:
        print("\nERROR: probabilities not found in results dictionary.")


if __name__ == "__main__":
    main()

