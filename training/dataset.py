import io
import os
import math
import random
import numpy as np # type: ignore
import torch # type: ignore
from PIL import Image # type: ignore
from torch.utils.data import Dataset, Sampler # type: ignore
import torchvision.transforms as transforms # type: ignore
from preprocessing.config import OUTPUT_FFPP, OUTPUT_CELEB, OUTPUT_GAN, SEED, IMAGE_SIZE

random.seed(SEED)
np.random.seed(SEED)

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")

# ImageNet normalisation — matches EfficientNet-B3 pretraining
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


class SingleDomainDataset(Dataset):
   
    def __init__(self, root_dir: str, split: str, domain_name: str = "domain"):
        assert split in ("train", "val", "test"), f"Invalid split '{split}'."
        self.root_dir    = root_dir
        self.split       = split
        self.domain_name = domain_name
        self.is_train    = split == "train"
        self.samples:      list = []
        self.real_indices: list = []
        self.fake_indices: list = []
        split_dir = os.path.join(root_dir, split)
        self._collect_images(os.path.join(split_dir, "real"), label=0)
        self._collect_images(os.path.join(split_dir, "fake"), label=1)
        self._build_class_indices()
        self._shuffle()
        self._log_distribution()

        # ── v5 Train transform — 224×224, stronger augmentation ───────────
        self.train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.15,
                hue=0.05
            ),
            transforms.RandomGrayscale(p=0.1),
            # GaussianBlur simulates video recompression blur — new in v5
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.3),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE
        ])

        # Eval transform — 224×224 
        self.eval_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE
        ])

    def _collect_images(self, class_dir: str, label: int) -> None:
        if not os.path.exists(class_dir):
            print(f"  WARNING [{self.domain_name}][{self.split}]: "
                  f"Directory not found — {class_dir}")
            return
        collected = 0
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(SUPPORTED_EXTS):
                self.samples.append((os.path.join(class_dir, fname), label))
                collected += 1
        label_str = "real" if label == 0 else "fake"
        print(f"  [{self.domain_name}][{self.split}] "
              f"Collected {collected:>6} {label_str} images.")

    def _build_class_indices(self) -> None:
        self.real_indices = []
        self.fake_indices = []
        for i, (_, label) in enumerate(self.samples):
            if label == 0:
                self.real_indices.append(i)
            else:
                self.fake_indices.append(i)

    def _shuffle(self) -> None:
        rng = random.Random(SEED)
        rng.shuffle(self.samples)
        self._build_class_indices()

    def _log_distribution(self) -> None:
        real_count = len(self.real_indices)
        fake_count = len(self.fake_indices)
        total      = len(self.samples)
        print(f"  [{self.domain_name}][{self.split}] "
              f"Real: {real_count} | Fake: {fake_count} | Total: {total}")
        if real_count != fake_count:
            print(f"  WARNING [{self.domain_name}][{self.split}]: "
                  f"Class imbalance — diff: {abs(real_count - fake_count)}")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  WARNING [{self.domain_name}]: Failed to load {path}: {e}")
            image = Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8))
        if self.is_train:
            image = self.train_transform(image)
        else:
            image = self.eval_transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


class MultiDomainDataset(Dataset):
    """Combines FF++, Celeb-DF, and GAN domains into unified dataset."""

    DOMAIN_CONFIG = [
        ("FF++",     OUTPUT_FFPP),
        ("Celeb-DF", OUTPUT_CELEB),
        ("GAN",      OUTPUT_GAN),
    ]

    def __init__(self, split: str):
        assert split in ("train", "val", "test")

        self.split    = split
        self.is_train = split == "train"

        print(f"\n{'='*60}")
        print(f" FortifyAI v5 — MultiDomainDataset [{split}]")
        print(f" Resolution: {IMAGE_SIZE}×{IMAGE_SIZE}")
        print(f"{'='*60}")

        self.domains: list = [
            SingleDomainDataset(root, split, name)
            for name, root in self.DOMAIN_CONFIG
        ]

        self.index_map: list = [
            (domain_id, local_idx)
            for domain_id, domain in enumerate(self.domains)
            for local_idx in range(len(domain))
        ]

        if self.is_train:
            random.shuffle(self.index_map)

        self._log_summary()

    def _log_summary(self) -> None:
        print(f"\n  [MultiDomainDataset][{self.split}] Summary:")
        for domain in self.domains:
            print(f"    {domain.domain_name:<12}: {len(domain):>8} samples")
        print(f"    {'Total':<12}: {len(self.index_map):>8} samples")
        print(f"{'='*60}\n")

    def get_domain_class_indices(self) -> list:
        reverse_map = {
            (domain_id, local_idx): global_idx
            for global_idx, (domain_id, local_idx) in enumerate(self.index_map)
        }
        result = []
        for domain_id, domain in enumerate(self.domains):
            real_global = [
                reverse_map[(domain_id, li)]
                for li in domain.real_indices
                if (domain_id, li) in reverse_map
            ]
            fake_global = [
                reverse_map[(domain_id, li)]
                for li in domain.fake_indices
                if (domain_id, li) in reverse_map
            ]
            result.append({
                "name": domain.domain_name,
                "real": real_global,
                "fake": fake_global
            })
        return result

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        domain_id, local_idx = self.index_map[idx]
        return self.domains[domain_id][local_idx]


class BalancedDomainSampler(Sampler):
   
    def __init__(
        self,
        dataset:    MultiDomainDataset,
        batch_size: int  = 24,       # v5: 24 (was 48) — 224×224 VRAM constraint
        drop_last:  bool = True
    ):
        super().__init__(dataset)

        self.dataset     = dataset
        self.batch_size  = batch_size
        self.drop_last   = drop_last
        self.num_domains = len(dataset.domains)

        assert batch_size % (self.num_domains * 2) == 0, (
            f"batch_size ({batch_size}) must be divisible by "
            f"num_domains * 2 ({self.num_domains * 2} = 6).\n"
            f"Valid values: 6, 12, 18, 24, 30, 36, 48..."
        )

        self.samples_per_domain   = batch_size // self.num_domains    # 8
        self.samples_per_class    = self.samples_per_domain // 2      # 4
        self.domain_class_indices = dataset.get_domain_class_indices()

        print(f"\n[BalancedDomainSampler v5] Configured:")
        print(f"  batch_size        : {batch_size}  (effective 48 with grad_accum=2)")
        print(f"  domains           : {self.num_domains}")
        print(f"  samples/domain    : {self.samples_per_domain} "
              f"({self.samples_per_class} real + {self.samples_per_class} fake)")
        print(f"  estimated batches : {self.__len__()}\n")

    def _build_batches(self) -> list:
        domain_pools = []
        for info in self.domain_class_indices:
            real_pool = info["real"].copy()
            fake_pool = info["fake"].copy()
            random.shuffle(real_pool)
            random.shuffle(fake_pool)
            domain_pools.append({"real": real_pool, "fake": fake_pool})

        min_real    = min(len(p["real"]) for p in domain_pools)
        min_fake    = min(len(p["fake"]) for p in domain_pools)
        num_batches = min(
            min_real // self.samples_per_class,
            min_fake // self.samples_per_class
        )

        batches = []
        for b in range(num_batches):
            batch = []
            start = b * self.samples_per_class
            end   = start + self.samples_per_class
            for pool in domain_pools:
                real_s = pool["real"][start:end]
                fake_s = pool["fake"][start:end]
                if len(real_s) < self.samples_per_class:
                    real_s += pool["real"][:self.samples_per_class - len(real_s)]
                if len(fake_s) < self.samples_per_class:
                    fake_s += pool["fake"][:self.samples_per_class - len(fake_s)]
                batch.extend(real_s + fake_s)
            random.shuffle(batch)
            batches.append(batch)

        random.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self._build_batches():
            yield batch

    def __len__(self) -> int:
        min_real = min(len(info["real"]) for info in self.domain_class_indices)
        min_fake = min(len(info["fake"]) for info in self.domain_class_indices)
        return min(
            min_real // self.samples_per_class,
            min_fake // self.samples_per_class
        )