import torch # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
from facenet_pytorch import MTCNN # type: ignore


class FaceExtractor:
    def __init__(self, image_size=224, device="cpu"):
        self.mtcnn = MTCNN(
            image_size = image_size,
            margin = 0,
            post_process = False,
            device = device
        )
    
    def extract_face(self, frame_rgb):
        img = Image.fromarray(frame_rgb)
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None:
            return None
        boxes = np.array(boxes)
        boxes = boxes.reshape(-1,4)
        probs = np.array(probs).flatten()
        # Filter detections by confidence
        valid_indices = [i for i, p in enumerate(probs)
           if p is not None and p > 0.90]
        if len(valid_indices) == 0:
           return None
        # Select largest face among valid detections
        areas = []
        for i in valid_indices:
            box = boxes[i]
            area = (box[2] - box[0]) * (box[3] - box[1])
            areas.append(area)
        largest_idx = valid_indices[np.argmax(areas)]
        largest_box = boxes[largest_idx].reshape(1,4)
        face = self.mtcnn.extract(img, largest_box, save_path=None)
        if face is None:
            return None
        return face