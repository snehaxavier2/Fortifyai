import os
import yaml
import random
import numpy as np # type: ignore
import pandas as pd # type: ignore
from tqdm import tqdm

from preprocessing.modules.frame_sampler import sample_frames_from_video
from preprocessing.modules.face_processor import FaceProcessor
from preprocessing.utils.logger import PreprocessLogger

# Utility Functions

def load_config():
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Core Processing Logic

def process_split(split_name, config, face_processor):

    print(f"\n=== Processing {split_name.upper()} Split ===")

    split_csv_path = os.path.join(
        config["paths"]["splits"],
        f"{split_name}.csv"
    )

    processed_root = config["paths"]["processed_dataset"]
    split_output_dir = os.path.join(processed_root, split_name)

    ensure_dir(split_output_dir)

    logger = PreprocessLogger(os.path.join("logs", f"{split_name}.log"))

    df = pd.read_csv(split_csv_path)
    total_videos = len(df)
    discarded_count = 0
    processed_count = 0

    logger.info(f"Total videos in {split_name}: {total_videos}")

    for _, row in tqdm(df.iterrows(), total=total_videos, desc=f"{split_name}"):

        relative_path = row["video_path"]
        video_path = os.path.join(
            config["paths"]["raw_dataset"],
            relative_path
        )
        label = row["label"]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        manipulation = None 

        try:
            # Sample frames
            sampled_frames = sample_frames_from_video(
                video_path,
                frames_per_video=config["sampling"]["frames_per_video"],
                seed=config["project"]["random_seed"]
            )

            # Generate boundaries
            total_frames = sampled_frames[-1][1] if sampled_frames else 0
            cap = None

            import cv2 # type: ignore
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            boundaries = np.linspace(
                0,
                total_frames,
                config["sampling"]["frames_per_video"] + 1,
                dtype=int
            )

            # Face detection + recovery
            results = face_processor.process_video_frames(
                video_path,
                sampled_frames,
                boundaries,
                recovery_range=config["sampling"]["recovery_range"]
            )

            if results is None:
                discarded_count += 1
                logger.info(f"[DISCARDED] {video_id}")
                continue

            # Prepare output directory (atomic creation)
            if label == 0:
                # Real
                output_dir = os.path.join(
                    split_output_dir,
                    "real",
                    video_id
                )
            else:
                # Fake
                output_dir = os.path.join(
                    split_output_dir,
                    "fake",
                    video_id
                )

            ensure_dir(output_dir)

            # Save faces
            for segment_id, frame_index, face in results:
                filename = f"frame_{segment_id:02d}_{frame_index}.jpg"
                save_path = os.path.join(output_dir, filename)

                import cv2 # type: ignore
                cv2.imwrite(
                    save_path,
                    face,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
            processed_count += 1

        except Exception as e:
            discarded_count += 1
            logger.error(f"[ERROR] {video_id} - {str(e)}")
            continue

    logger.summary(
        f"{split_name} Completed | Processed: {processed_count} | Discarded: {discarded_count}"
    )

    print(f"{split_name} Completed.")
    print(f"Processed: {processed_count}")
    print(f"Discarded: {discarded_count}")

# Entry Point

def main():
    config = load_config()
    # Determinism
    random.seed(config["project"]["random_seed"])
    np.random.seed(config["project"]["random_seed"])

    face_processor = FaceProcessor(
        min_valid_frames=config["sampling"]["min_valid_frames"],
        device=config["system"]["device"]
    )
    for split in ["test"]:  
        process_split(split, config, face_processor)


if __name__ == "__main__":
    main()
