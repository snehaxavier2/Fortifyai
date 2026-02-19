import cv2 # type: ignore
import numpy as np # pyright: ignore[reportMissingImports]
import random


def sample_frames_from_video(video_path, frames_per_video=20, seed=42):
    """
    Extract stratified temporal frames from a video.

    Returns:
        List of tuples:
        (segment_id, frame_index, frame_array)
    """

    random.seed(seed)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid frame count in video: {video_path}")

    if total_frames < frames_per_video:
        cap.release()
        raise ValueError(
            f"Video too short. Frames: {total_frames}, Required: {frames_per_video}"
        )

    # Generate segment boundaries
    boundaries = np.linspace(0, total_frames, frames_per_video + 1, dtype=int)

    sampled_frames = []

    for segment_id in range(frames_per_video):
        start = boundaries[segment_id]
        end = boundaries[segment_id + 1]

        if start >= end:
            frame_idx = start
        else:
            frame_idx = random.randint(start, end - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue  # Skip

        sampled_frames.append((segment_id, frame_idx, frame))

    cap.release()

    return sampled_frames
