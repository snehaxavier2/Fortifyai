import os
import random
from PIL import Image # type: ignore
from tqdm import tqdm
from preprocessing.config import (
OUTPUT_FFPP,
OUTPUT_CELEB,
OUTPUT_GAN,
SEED
)

random.seed(SEED)
DATASET_ROOTS = [
("FF++", OUTPUT_FFPP),
("Celeb-DF", OUTPUT_CELEB),
("GAN", OUTPUT_GAN)
]
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")
JPEG_MIN_QUALITY = 30
JPEG_MAX_QUALITY = 95

def recompress_image(path: str) -> None:
    try:
        img = Image.open(path).convert("RGB")
        quality = random.randint(JPEG_MIN_QUALITY, JPEG_MAX_QUALITY)
        img.save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0
    )
    except Exception as e:
        print(f"WARNING: Failed to recompress {path}: {e}")


def process_domain(name: str, root: str) -> None:
    train_dir = os.path.join(root, "train")
    if not os.path.exists(train_dir):
        print(f"WARNING: Train directory not found for {name}: {train_dir}")
        return
    print("\n" + "=" * 60)
    print(f" FortifyAI JPEG Augmentation — {name} (train set only)")
    print("=" * 60)
    image_paths = []
    for root_dir, _, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTS):
                image_paths.append(os.path.join(root_dir, file))
    print(f"Total train images found: {len(image_paths)}")
    for path in tqdm(image_paths, desc=f"{name} train recompression"):
        recompress_image(path)
    print(f"{name} train JPEG augmentation completed.")


if __name__ == "__main__":
    print("=" * 60)
    print(" FortifyAI — JPEG Compression Augmentation")
    print("=" * 60)
    for name, root in DATASET_ROOTS:
        process_domain(name, root)
    print("\nAll domains processed successfully.")