import os
import random
import shutil
import numpy as np # type: ignore
from tqdm import tqdm
from PIL import Image # type: ignore
from preprocessing.config import (
    SEED, RAW_FFHQ, RAW_STYLEGAN, OUTPUT_GAN, TEMP_FACE_GAN,
    FRAMES_PER_CLASS, IMAGE_SIZE
)


# Reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Supported image extensions
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")


# Image Collection

def collect_images(root_dir: str, label: str) -> list:
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"{label.capitalize()} image directory not found: {root_dir}")
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(SUPPORTED_EXTS):
                image_paths.append(os.path.join(root, fname))
    print(f"[GAN] Found {len(image_paths)} {label} images in {root_dir}")
    return image_paths


# Image Processing

def process_images(image_paths: list, label: str, target_count: int) -> None:
    save_dir = os.path.join(TEMP_FACE_GAN, label)
    os.makedirs(save_dir, exist_ok=True)
    saved_count = 0
    with tqdm(total=target_count, desc=f"Processing {label} images") as pbar:
        for img_path in image_paths:
            if saved_count >= target_count:
                break
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  WARNING: Failed to open {img_path}: {e}")
                continue
            # Direct resize
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            filename = f"{label}_{saved_count:05d}.jpg"
            save_path = os.path.join(save_dir, filename)
            image.save(save_path, format="JPEG", quality=95)
            saved_count += 1
            pbar.update(1)
    print(f"[GAN] {label.capitalize()} images saved: {saved_count}")

# Dataset Splitting

def split_dataset() -> None:
    for label in ["real", "fake"]:
        label_dir = os.path.join(TEMP_FACE_GAN, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Temp face directory not found: {label_dir}")
        images = sorted([
            f for f in os.listdir(label_dir)
            if f.lower().endswith(SUPPORTED_EXTS)
        ])
        random.shuffle(images)
        total = len(images)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        splits = {
            "train": images[:train_end],
            "val":   images[train_end:val_end],
            "test":  images[val_end:]
        }
        print(f"\n[GAN] Splitting {label}: "
              f"train={len(splits['train'])} | "
              f"val={len(splits['val'])} | "
              f"test={len(splits['test'])}")
        for split_name, split_images in splits.items():
            for img in tqdm(split_images, desc=f"  Moving {label}/{split_name}"):
                src = os.path.join(label_dir, img)
                dst = os.path.join(OUTPUT_GAN, split_name, label, img)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
    print("\n[GAN] Dataset split completed.")


# Entry Point

if __name__ == "__main__":
    print("=" * 60)
    print("FortifyAI v3 — GAN Domain Preprocessing Pipeline")
    print("=" * 60)
    print("\n[Step 1/4] Collecting images...")
    real_images = collect_images(RAW_FFHQ, "real")
    fake_images = collect_images(RAW_STYLEGAN, "fake")
    if len(real_images) < FRAMES_PER_CLASS:
        raise ValueError(
            f"Not enough real images: found {len(real_images)}, "
            f"need {FRAMES_PER_CLASS}."
        )
    if len(fake_images) < FRAMES_PER_CLASS:
        raise ValueError(
            f"Not enough fake images: found {len(fake_images)}, "
            f"need {FRAMES_PER_CLASS}."
        )
    real_selected = random.sample(real_images, FRAMES_PER_CLASS)
    fake_selected = random.sample(fake_images, FRAMES_PER_CLASS)
    random.shuffle(real_selected)
    random.shuffle(fake_selected)
    print(f"[GAN] Real images selected: {len(real_selected)}")
    print(f"[GAN] Fake images selected: {len(fake_selected)}")
    print("\n[Step 2/4] Processing REAL images (FFHQ)...")
    process_images(real_selected, "real", FRAMES_PER_CLASS)
    print("\n[Step 3/4] Processing FAKE images (StyleGAN)...")
    process_images(fake_selected, "fake", FRAMES_PER_CLASS)
    print("\n[Step 4/4] Splitting dataset...")
    split_dataset()
    print("\n" + "=" * 60)
    print("GAN domain preprocessing completed successfully.")
    print("=" * 60)