import os
import csv
import random
import yaml
from pathlib import Path


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collect_videos(raw_root, real_path, fake_paths):
    real_videos = []
    fake_videos = []

    # Collect real videos
    real_dir = Path(raw_root) / real_path
    for file in real_dir.glob("*.mp4"):
        real_videos.append(str(file.relative_to(raw_root)))

    # Collect fake videos from all manipulation folders
    for fake_subpath in fake_paths:
        fake_dir = Path(raw_root) / fake_subpath
        for file in fake_dir.glob("*.mp4"):
            fake_videos.append(str(file.relative_to(raw_root)))

    return real_videos, fake_videos


def stratified_split(real_videos, fake_videos, train_ratio, val_ratio, seed):
    random.seed(seed)

    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    def split_list(video_list):
        n = len(video_list)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        train = video_list[:train_end]
        val = video_list[train_end:val_end]
        test = video_list[val_end:]
        return train, val, test

    r_train, r_val, r_test = split_list(real_videos)
    f_train, f_val, f_test = split_list(fake_videos)

    train_set = [(v, 0) for v in r_train] + [(v, 1) for v in f_train]
    val_set = [(v, 0) for v in r_val] + [(v, 1) for v in f_val]
    test_set = [(v, 0) for v in r_test] + [(v, 1) for v in f_test]

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def save_split(split_data, output_path):
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        writer.writerows(split_data)


def main():
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config = load_config(config_path)

    raw_root = Path(config["paths"]["raw_dataset"])
    splits_root = Path(config["paths"]["splits"])

    real_path = config["dataset"]["real_path"]
    fake_paths = config["dataset"]["fake_paths"]

    train_ratio = config["splitting"]["train_ratio"]
    val_ratio = config["splitting"]["val_ratio"]
    seed = config["project"]["random_seed"]

    print("Collecting videos...")
    real_videos, fake_videos = collect_videos(raw_root, real_path, fake_paths)

    print(f"Total real videos: {len(real_videos)}")
    print(f"Total fake videos: {len(fake_videos)}")

    print("Performing stratified split...")
    train_set, val_set, test_set = stratified_split(
        real_videos, fake_videos, train_ratio, val_ratio, seed
    )

    print(f"Train size: {len(train_set)}")
    print(f"Val size: {len(val_set)}")
    print(f"Test size: {len(test_set)}")

    save_split(train_set, splits_root / "train.csv")
    save_split(val_set, splits_root / "val.csv")
    save_split(test_set, splits_root / "test.csv")

    print("Splits saved successfully.")


if __name__ == "__main__":
    main()
