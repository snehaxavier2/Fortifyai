import os
import random
import shutil
import numpy as np # type: ignore
from tqdm import tqdm
from PIL import Image # type: ignore

from preprocessing.config import (
    SEED, RAW_FFPP, OUTPUT_FFPP, TEMP_FACE,
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
    real_root = os.path.join(
        RAW_FFPP, "original_sequences", "youtube", "c23", "videos"
    )
    if not os.path.exists(real_root):
        raise FileNotFoundError(f"Real video directory not found: {real_root}")
    videos = [
        os.path.join(real_root, v)
        for v in os.listdir(real_root)
        if v.endswith(".mp4")
    ]
    print(f"[FF++] Found {len(videos)} real videos.")
    return videos


def get_fake_videos() -> list:
    fake_root = os.path.join(RAW_FFPP, "manipulated_sequences")
    if not os.path.exists(fake_root):
        raise FileNotFoundError(f"Fake video directory not found: {fake_root}")
    manipulation_types = [
        d for d in os.listdir(fake_root)
        if os.path.isdir(os.path.join(fake_root, d))
    ]
    print(f"[FF++] Found manipulation types: {manipulation_types}")
    samples_per_type = VIDEOS_PER_CLASS // len(manipulation_types)
    remainder = VIDEOS_PER_CLASS % len(manipulation_types)
    if remainder != 0:
        print(
            f"  WARNING: VIDEOS_PER_CLASS ({VIDEOS_PER_CLASS}) is not evenly divisible "
            f"by {len(manipulation_types)} manipulation types. "
            f"{remainder} videos will be dropped."
        )
    fake_videos = []
    for mtype in manipulation_types:
        mtype_path = os.path.join(fake_root, mtype, "c23", "videos")
        if not os.path.exists(mtype_path):
            print(f"  WARNING: Path not found, skipping: {mtype_path}")
            continue
        videos = [
            os.path.join(mtype_path, v)
            for v in os.listdir(mtype_path)
            if v.endswith(".mp4")
        ]
        if len(videos) < samples_per_type:
            print(
                f"  WARNING: {mtype} has only {len(videos)} videos, "
                f"requested {samples_per_type}. Using all available."
            )
            selected = videos
        else:
            selected = random.sample(videos, samples_per_type)
        fake_videos.extend(selected)
        print(f"  [{mtype}] Selected {len(selected)} videos.")

    print(f"[FF++] Total fake videos selected: {len(fake_videos)}")
    if len(fake_videos) < VIDEOS_PER_CLASS:
        print(
            f"  WARNING: Collected {len(fake_videos)} fake videos, "
            f"expected {VIDEOS_PER_CLASS}. Check manipulation type paths."
        )
    return fake_videos


# Face Saving Utility

def save_face(face_tensor, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    face_np = face_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    face_np = (face_np * 255).astype(np.uint8) if face_np.max() <= 1.0 else face_np.astype(np.uint8)
    Image.fromarray(face_np).save(save_path, format="JPEG", quality=95)


# Video Processing

def process_videos(video_paths: list, label: str, target_count: int) -> None:
    extractor = FaceExtractor(image_size=IMAGE_SIZE, device=DEVICE)
    save_dir = os.path.join(TEMP_FACE, label)
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
        print(f"[FF++] {label.capitalize()} faces saved: {saved_count}")


# Dataset Splitting

def split_dataset() -> None:
    for label in ["real", "fake"]:
        label_dir = os.path.join(TEMP_FACE, label)
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
        print(f"\n[FF++] Splitting {label}: "
              f"train={len(splits['train'])} | "
              f"val={len(splits['val'])} | "
              f"test={len(splits['test'])}")
        for split_name, split_images in splits.items():
            for img in tqdm(split_images, desc=f"  Moving {label}/{split_name}"):
                src = os.path.join(label_dir, img)
                dst = os.path.join(OUTPUT_FFPP, split_name, label, img)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
    print("\n[FF++] Dataset split completed.")


# Entry Point

if __name__ == "__main__":
    print("=" * 60)
    print("FortifyAI — FF++ Preprocessing Pipeline")
    print("=" * 60)
    print("\n[Step 1/4] Collecting videos...")
    real_videos = get_real_videos()
    fake_videos = get_fake_videos()
    random.shuffle(fake_videos)
    if len(real_videos) < VIDEOS_PER_CLASS:
        raise ValueError(
            f"Not enough real videos: found {len(real_videos)}, "
            f"need {VIDEOS_PER_CLASS}."
        )
    real_selected = random.sample(real_videos, VIDEOS_PER_CLASS)
    print(f"[FF++] Real videos selected: {len(real_selected)}")
    print("\n[Step 2/4] Processing REAL videos...")
    process_videos(real_selected, "real", FRAMES_PER_CLASS)
    print("\n[Step 3/4] Processing FAKE videos...")
    process_videos(fake_videos, "fake", FRAMES_PER_CLASS)
    print("\n[Step 4/4] Splitting dataset...")
    split_dataset()
    print("\n" + "=" * 60)
    print("FF++ preprocessing completed successfully.")
    print("=" * 60)