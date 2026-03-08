import os
import random
import shutil
import numpy as np # type: ignore
from tqdm import tqdm
from PIL import Image # type: ignore
from preprocessing.config import (
    SEED, RAW_CELEBDF, OUTPUT_CELEB, TEMP_FACE_CELEB,
    VIDEOS_PER_CLASS, FRAMES_PER_CLASS,
    FRAME_PER_VIDEO, IMAGE_SIZE, DEVICE
)
from preprocessing.frame_extractor import extract_frames
from preprocessing.face_detection import FaceExtractor

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)


# Video Collection

def get_real_videos() -> list:
    real_dirs = [
        os.path.join(RAW_CELEBDF, "Celeb-real"),
        os.path.join(RAW_CELEBDF, "YouTube-real")
    ]
    videos = []
    for real_root in real_dirs:
        if not os.path.exists(real_root):
            continue
        videos.extend([
            os.path.join(real_root, v)
            for v in os.listdir(real_root)
            if v.endswith(".mp4")
        ])
    print(f"[Celeb-DF] Found {len(videos)} real videos.")
    return videos


def get_fake_videos() -> list:
    fake_root = os.path.join(RAW_CELEBDF, "Celeb-synthesis")
    if not os.path.exists(fake_root):
        raise FileNotFoundError(f"Fake video directory not found: {fake_root}")
    videos = [
        os.path.join(fake_root, v)
        for v in os.listdir(fake_root)
        if v.endswith(".mp4")
    ]
    print(f"[Celeb-DF] Found {len(videos)} fake videos.")
    return videos


# Face Saving Utility

def save_face(face_tensor, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    face_np = face_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    face_np = (face_np * 255).astype(np.uint8) if face_np.max() <= 1.0 else face_np.astype(np.uint8)
    Image.fromarray(face_np).save(save_path, format="JPEG", quality=95)


# Video Processing

def process_videos(video_paths: list, label: str, target_count: int) -> None:
    extractor = FaceExtractor(image_size=IMAGE_SIZE, device=DEVICE)
    save_dir = os.path.join(TEMP_FACE_CELEB, label)
    os.makedirs(save_dir, exist_ok=True)
    saved_count = 0
    video_idx = 0
    with tqdm(total=target_count, desc=f"Processing {label} videos") as pbar:
        while saved_count < target_count and video_idx < len(video_paths):
            video_path = video_paths[video_idx]
            video_idx += 1
            try:
                frames = extract_frames(video_path, FRAME_PER_VIDEO)
            except Exception as e:
                print(f"  WARNING: Failed to extract frames from {video_path}: {e}")
                continue
            for frame in frames:
                if saved_count >= target_count:
                    break
                try:
                    face = extractor.extract_face(frame)
                except Exception as e:
                    print(f"  WARNING: Face extraction error: {e}")
                    continue
                if face is not None:
                    filename = f"{label}_{saved_count:05d}.jpg"
                    save_path = os.path.join(save_dir, filename)
                    save_face(face, save_path)
                    saved_count += 1
                    pbar.update(1)
    if saved_count < target_count:
        print(
            f"  WARNING: Only saved {saved_count}/{target_count} {label} faces. "
            f"Consider expanding video pool."
        )
    else:
        print(f"[Celeb-DF] {label.capitalize()} faces saved: {saved_count}")


# Dataset Splitting

def split_dataset() -> None:
    for label in ["real", "fake"]:
        label_dir = os.path.join(TEMP_FACE_CELEB, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Temp face directory not found: {label_dir}")
        images = sorted([
            f for f in os.listdir(label_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
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
        print(f"\n[Celeb-DF] Splitting {label}: "
              f"train={len(splits['train'])} | "
              f"val={len(splits['val'])} | "
              f"test={len(splits['test'])}")
        for split_name, split_images in splits.items():
            for img in tqdm(split_images, desc=f"  Moving {label}/{split_name}"):
                src = os.path.join(label_dir, img)
                dst = os.path.join(OUTPUT_CELEB, split_name, label, img)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
    print("\n[Celeb-DF] Dataset split completed.")


# Entry Point

if __name__ == "__main__":
    print("=" * 60)
    print("FortifyAI — Celeb-DF Preprocessing Pipeline")
    print("=" * 60)
    print("\n[Step 1/4] Collecting videos...")
    real_videos = get_real_videos()
    fake_videos = get_fake_videos()
    if len(real_videos) < VIDEOS_PER_CLASS:
        print(f"  [Celeb-DF] WARNING: Only {len(real_videos)} real videos available "
          f"(requested {VIDEOS_PER_CLASS}). Using all {len(real_videos)}.")
        real_videos_selected = real_videos
    else:
        real_videos_selected = random.sample(real_videos, VIDEOS_PER_CLASS)
    if len(fake_videos) < VIDEOS_PER_CLASS:
        print(f"  [Celeb-DF] WARNING: Only {len(fake_videos)} fake videos available "
          f"(requested {VIDEOS_PER_CLASS}). Using all {len(fake_videos)}.")
        fake_videos_selected = fake_videos
    else:
        fake_videos_selected = random.sample(fake_videos, VIDEOS_PER_CLASS)
    real_selected = random.sample(real_videos, VIDEOS_PER_CLASS)
    fake_selected = random.sample(fake_videos, VIDEOS_PER_CLASS)
    random.shuffle(real_selected)
    random.shuffle(fake_selected)
    print(f"[Celeb-DF] Real videos selected: {len(real_selected)}")
    print(f"[Celeb-DF] Fake videos selected: {len(fake_selected)}")
    print("\n[Step 2/4] Processing REAL videos...")
    process_videos(real_selected, "real", FRAMES_PER_CLASS)
    print("\n[Step 3/4] Processing FAKE videos...")
    process_videos(fake_selected, "fake", FRAMES_PER_CLASS)
    print("\n[Step 4/4] Splitting dataset...")
    split_dataset()
    print("\n" + "=" * 60)
    print("Celeb-DF preprocessing completed successfully.")
    print("=" * 60)