import os
import csv
import random
import yaml
from pathlib import Path

# Load Configuration

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Collect Real Videos

def collect_real_videos(raw_root, real_path):
    real_dir = Path(raw_root) / real_path

    if not real_dir.exists():
        raise FileNotFoundError(f"Real path not found: {real_dir}")

    videos = list(real_dir.glob("*.mp4"))
    return [str(v.relative_to(raw_root)) for v in videos]

# Collect Fake Videos Per Type

def collect_fake_videos_by_type(raw_root, manipulated_root, compression_level):
    manipulated_dir = Path(raw_root) / manipulated_root

    if not manipulated_dir.exists():
        raise FileNotFoundError(f"Manipulated root not found: {manipulated_dir}")

    fake_dict = {}

    for manipulation_type in manipulated_dir.iterdir():
        if manipulation_type.is_dir():
            video_dir = manipulation_type / compression_level / "videos"
            if video_dir.exists():
                videos = list(video_dir.glob("*.mp4"))
                if len(videos) > 0:
                    fake_dict[manipulation_type.name] = [
                        str(v.relative_to(raw_root)) for v in videos
                    ]

    if len(fake_dict) == 0:
        raise ValueError("No fake manipulation folders detected.")

    return fake_dict

# Balanced Fake Sampling

def balanced_fake_sampling(fake_dict, total_fake, seed):
    random.seed(seed)

    manipulation_types = list(fake_dict.keys())
    num_types = len(manipulation_types)

    videos_per_type = total_fake // num_types
    remainder = total_fake % num_types

    balanced_fakes = []

    for idx, m_type in enumerate(manipulation_types):
        videos = fake_dict[m_type]

        if len(videos) < videos_per_type:
            raise ValueError(
                f"Not enough videos in {m_type}. "
                f"Required: {videos_per_type}, Available: {len(videos)}"
            )

        random.shuffle(videos)

        count = videos_per_type + (1 if idx < remainder else 0)
        selected = videos[:count]

        balanced_fakes.extend(selected)

    if len(balanced_fakes) != total_fake:
        raise ValueError("Balanced fake sampling failed. Incorrect total count.")

    return balanced_fakes

# Stratified Split

def stratified_split(real_list, fake_list, train_ratio, val_ratio, seed):
    random.seed(seed)

    random.shuffle(real_list)
    random.shuffle(fake_list)

    def split_class(data):
        n = len(data)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]
        return train, val, test

    r_train, r_val, r_test = split_class(real_list)
    f_train, f_val, f_test = split_class(fake_list)

    train = [(v, 0) for v in r_train] + [(v, 1) for v in f_train]
    val = [(v, 0) for v in r_val] + [(v, 1) for v in f_val]
    test = [(v, 0) for v in r_test] + [(v, 1) for v in f_test]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test

# Save CSV

def save_split(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        writer.writerows(data)

# Main Execution

def main():
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config = load_config(config_path)

    raw_root = config["paths"]["raw_dataset"]
    splits_root = Path(config["paths"]["splits"])

    real_path = config["dataset"]["real_path"]
    manipulated_root = config["dataset"]["manipulated_root"]
    compression_level = config["dataset"]["compression_level"]

    total_real = config["dataset"]["total_real_videos"]
    total_fake = config["dataset"]["total_fake_videos"]

    train_ratio = config["splitting"]["train_ratio"]
    val_ratio = config["splitting"]["val_ratio"]

    seed = config["project"]["random_seed"]

    print("Collecting real videos...")
    real_videos = collect_real_videos(raw_root, real_path)

    if len(real_videos) < total_real:
        raise ValueError(
            f"Not enough real videos. Required: {total_real}, Available: {len(real_videos)}"
        )

    random.seed(seed)
    random.shuffle(real_videos)
    real_videos = real_videos[:total_real]

    print(f"Selected real videos: {len(real_videos)}")

    print("Collecting fake videos by manipulation type...")
    fake_dict = collect_fake_videos_by_type(raw_root, manipulated_root, compression_level)

    for k, v in fake_dict.items():
        print(f"{k}: {len(v)} videos")

    print("Performing balanced fake sampling...")
    fake_videos = balanced_fake_sampling(fake_dict, total_fake, seed)
    print(f"Selected fake videos: {len(fake_videos)}")

    print("Performing stratified split...")
    train_set, val_set, test_set = stratified_split(
    real_videos, fake_videos, train_ratio, val_ratio, seed
)

    def simple_balance_check(data, name):
        real = sum(1 for _, label in data if label == 0)
        fake = sum(1 for _, label in data if label == 1)
        print(f"{name} -> Real: {real}, Fake: {fake}")
        if real != fake:
            raise ValueError(f"Class imbalance detected in {name} split.")

    simple_balance_check(train_set, "Train")
    simple_balance_check(val_set, "Validation")
    simple_balance_check(test_set, "Test")


    save_split(train_set, splits_root / "train.csv")
    save_split(val_set, splits_root / "val.csv")
    save_split(test_set, splits_root / "test.csv")

    print("Split generation completed successfully.")


if __name__ == "__main__":
    main()
