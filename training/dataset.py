import os
import numpy as np # type: ignore
from PIL import Image # type: ignore
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
import torchvision.transforms as transforms # type: ignore
import cv2 # type: ignore


class FFPPDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir: E:/Datasets/FFPP_Processed
        split: train / val / test
        """
        self.root_dir = root_dir
        self.split = split

        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(root_dir, split)

        # Collect real images
        real_dir = os.path.join(split_dir, "real")
        fake_dir = os.path.join(split_dir, "fake")

        self._collect_images(real_dir, label=0)
        self._collect_images(fake_dir, label=1)

        # RGB transform for MobileNetV2
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _collect_images(self, base_dir, label):
        for video_id in os.listdir(base_dir):
            video_path = os.path.join(base_dir, video_id)
            if not os.path.isdir(video_path):
                continue

            for img_name in os.listdir(video_path):
                if img_name.endswith(".jpg"):
                    self.image_paths.append(os.path.join(video_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def _compute_fft(self, img_rgb):
        """
        img_rgb: numpy array (H, W, 3)
        returns: torch tensor (1, H, W)
        """

        # Convert to Y channel
        img_y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]

        # FFT
        f = np.fft.fft2(img_y)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Log magnitude
        magnitude = np.log1p(magnitude)

        # High-frequency emphasis mask
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        mask = np.ones_like(magnitude)
        radius = min(h, w) // 8

        y, x = np.ogrid[:h, :w]
        distance = (y - center_h)**2 + (x - center_w)**2
        mask[distance < radius**2] = 0.5  # suppress low freq slightly

        magnitude = magnitude * mask

        # Normalize to [0,1]
        magnitude = (magnitude - magnitude.min()) / (
            magnitude.max() - magnitude.min() + 1e-8
        )

        magnitude = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)

        return magnitude

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_rgb = np.array(image)

        # Spatial branch
        rgb_tensor = self.rgb_transform(image)

        # Frequency branch
        fft_tensor = self._compute_fft(img_rgb)

        return {
            "rgb": rgb_tensor,
            "fft": fft_tensor,
            "label": torch.tensor(label, dtype=torch.float32)
        }
