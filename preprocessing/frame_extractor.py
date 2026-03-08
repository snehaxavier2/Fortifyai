import cv2 # type: ignore
import numpy as np # type: ignore

def extract_frames(video_path, frames_per_video=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < frames_per_video:
        cap.release()
        return[]
    segment_size = total_frames // frames_per_video
    selected_frames = []
    for i in range(frames_per_video):
        start = i * segment_size
        end = (i+1) * segment_size - 1
        frame_idx = np.random.randint(start, max(start + 1, end))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        selected_frames.append(frame)
    cap.release() 
    return selected_frames