import torch # type: ignore
import os

# GLOBAL CONFIGURATION
SEED = 42
VIDEOS_PER_CLASS = 890
FRAME_PER_VIDEO = 8
FRAMES_PER_CLASS = VIDEOS_PER_CLASS * FRAME_PER_VIDEO
IMAGE_SIZE = 224
BLUR_THRESHOLD = 50.0
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# PATHS
RAW_FFPP = "E:/Datasets/FaceForensicsPP"
OUTPUT_FFPP = "E:/Datasets/FaceForensicsPP_processed"
TEMP_FACE = "E:/Datasets/temp_faces/FaceForensicsPP"
RAW_CELEBDF = "E:/Datasets/CelebDF/Celeb-DF-v2"
OUTPUT_CELEB = "E:/Datasets/CelebDF_processed"
TEMP_FACE_CELEB =  "E:/Datasets/temp_faces/CelebDF"
RAW_GAN = "E:/Datasets/GANFaces"
RAW_FFHQ = os.path.join(RAW_GAN, "FFHQ", "real")
RAW_STYLEGAN = os.path.join(RAW_GAN, "StyleGAN", "fake")
OUTPUT_GAN = "E:/Datasets/GANFaces_processed"
TEMP_FACE_GAN = "E:/Datasets/temp_faces/GANFaces"

# DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")