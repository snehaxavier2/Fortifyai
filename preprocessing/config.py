import torch # type: ignore
import os
from dotenv import load_dotenv # type: ignore

load_dotenv()                  

SEED             = 42
VIDEOS_PER_CLASS = 890
FRAME_PER_VIDEO  = 8
FRAMES_PER_CLASS = VIDEOS_PER_CLASS * FRAME_PER_VIDEO
IMAGE_SIZE       = 224
BLUR_THRESHOLD   = 50.0
TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.10
TEST_RATIO       = 0.10

# PATHS — now reads from .env instead of hardcoded
_DATASETS = os.getenv("FORTIFY_DATASETS", "E:/Datasets")  # ← CHANGE THIS
_OUTPUT   = os.getenv("FORTIFY_OUTPUT",   "E:/Datasets")  # ← CHANGE THIS

RAW_FFPP        = os.path.join(_DATASETS, "FaceForensicsPP")
OUTPUT_FFPP  = f"{_OUTPUT}/FaceForensicsPP_processed"
TEMP_FACE       = os.path.join(_DATASETS, "temp_faces", "FaceForensicsPP")
RAW_CELEBDF     = os.path.join(_DATASETS, "CelebDF", "Celeb-DF-v2")
OUTPUT_CELEB = f"{_OUTPUT}/CelebDF_processed"
TEMP_FACE_CELEB = os.path.join(_DATASETS, "temp_faces", "CelebDF")
RAW_GAN         = os.path.join(_DATASETS, "GANFaces")
RAW_FFHQ        = os.path.join(RAW_GAN,   "FFHQ",     "real")
RAW_STYLEGAN    = os.path.join(RAW_GAN,   "StyleGAN", "fake")
OUTPUT_GAN   = f"{_OUTPUT}/GANFaces_processed"
TEMP_FACE_GAN   = os.path.join(_DATASETS, "temp_faces", "GANFaces")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")