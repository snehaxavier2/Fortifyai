import os
import torch # type: ignore
import torch.nn as nn # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score # type: ignore


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        stage1_epochs=3,
        total_epochs=15,
        checkpoint_dir="checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.stage1_epochs = stage1_epochs
        self.total_epochs = total_epochs

        self.criterion = nn.BCEWithLogitsLoss()

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_f1 = 0.0

        # Freeze backbone initially (Stage 1)
        for param in self.model.spatial_features.parameters():
            param.requires_grad = False

    # TRAIN LOOP
    def train(self):

        print("Stage 1: Backbone Frozen")

        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.classifier.parameters(), "lr": 1e-3},
                {"params": self.model.fft_branch.parameters(), "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )

        for epoch in range(1, self.total_epochs + 1):

            if epoch == self.stage1_epochs + 1:
                print("Stage 2: Unfreezing Last Two Backbone Blocks")

                # Unfreeze last 2 blocks
                backbone_blocks = list(self.model.spatial_features.children())
                for block in backbone_blocks[-2:]:
                    for param in block.parameters():
                        param.requires_grad = True

                optimizer = torch.optim.AdamW(
                    [
                        {"params": self.model.classifier.parameters(), "lr": 1e-3},
                        {"params": self.model.fft_branch.parameters(), "lr": 1e-3},
                        {
                            "params": backbone_blocks[-2:].parameters()
                            if hasattr(backbone_blocks[-2:], "parameters")
                            else [p for block in backbone_blocks[-2:] for p in block.parameters()],
                            "lr": 1e-4,
                        },
                    ],
                    weight_decay=1e-4,
                )

            train_loss, train_acc = self._train_one_epoch(optimizer)
            val_loss, val_acc, precision, recall, f1 = self._validate()

            print(f"\nEpoch {epoch}/{self.total_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            # Save best model
            if f1 > self.best_f1:
                self.best_f1 = f1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, "best_model.pth"),
                )
                print(f"Best model saved (F1: {f1:.4f})")

    # TRAIN ONE EPOCH
    def _train_one_epoch(self, optimizer):

        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:

            rgb = batch["rgb"].to(self.device)
            fft = batch["fft"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()

            outputs = self.model(rgb, fft).squeeze(1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.float() == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    # VALIDATION
    def _validate(self):

        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:

                rgb = batch["rgb"].to(self.device)
                fft = batch["fft"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(rgb, fft).squeeze(1)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        return avg_loss, accuracy, precision, recall, f1