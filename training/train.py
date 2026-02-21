import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from training.dataset import FFPPDataset
from models.hybrid_model import HybridModel
from training.trainer import Trainer


def main():

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset Paths
    dataset_root = "E:/Datasets/FFPP_Processed"

    # Create Datasets
    train_dataset = FFPPDataset(root_dir=dataset_root, split="train")
    val_dataset = FFPPDataset(root_dir=dataset_root, split="val")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize Model
    model = HybridModel(pretrained=True).to(device)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        stage1_epochs=5,
        total_epochs=20,
        checkpoint_dir="checkpoints"
    )

    # Start Training
    trainer.train()


if __name__ == "__main__":
    main()
