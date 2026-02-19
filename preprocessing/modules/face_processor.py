import cv2 # type: ignore
import numpy as np # type: ignore
from insightface.app import FaceAnalysis # type: ignore


class FaceProcessor:
    def __init__(self, min_valid_frames=15, device="cpu"):
        self.min_valid_frames = min_valid_frames

        # Initialize RetinaFace via InsightFace
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1 if device == "cpu" else 0)

    def _get_largest_face(self, faces):
        if len(faces) == 0:
            return None

        # Choose face with largest bounding box area
        largest = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
        )
        return largest

    def _crop_and_resize(self, frame, bbox, size=224):
        x1, y1, x2, y2 = bbox.astype(int)

        # Clip boundaries
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face_crop = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (size, size))

        return face_resized

    def process_video_frames(
        self,
        video_path,
        sampled_frames,
        segment_boundaries,
        recovery_range=5,
    ):
        """
        sampled_frames:
            List of (segment_id, frame_index, frame_array)

        segment_boundaries:
            Array of boundaries from np.linspace

        Returns:
            List of tuples:
            (segment_id, frame_index, face_crop)
        """

        cap = cv2.VideoCapture(video_path)
        valid_faces = []

        for segment_id, frame_index, frame in sampled_frames:

            face = self._detect_face(frame)

            if face is not None:
                valid_faces.append((segment_id, frame_index, face))
                continue

            # Recovery: symmetric outward search
            start = segment_boundaries[segment_id]
            end = segment_boundaries[segment_id + 1] - 1

            offsets = [0]
            for i in range(1, recovery_range + 1):
                offsets.extend([i, -i])

            recovered = False

            for offset in offsets:
                candidate_idx = frame_index + offset

                if candidate_idx < start or candidate_idx > end:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
                ret, new_frame = cap.read()

                if not ret or new_frame is None:
                    continue

                face = self._detect_face(new_frame)

                if face is not None:
                    valid_faces.append((segment_id, candidate_idx, face))
                    recovered = True
                    break

            if not recovered:
                continue  # Skip this segment

        cap.release()

        if len(valid_faces) >= self.min_valid_frames:
            return valid_faces
        else:
            return None  # Discard video

    def _detect_face(self, frame):
        faces = self.app.get(frame)

        if len(faces) == 0:
            return None

        largest_face = self._get_largest_face(faces)

        if largest_face is None:
            return None

        bbox = largest_face.bbox
        face_crop = self._crop_and_resize(frame, bbox)

        return face_crop
