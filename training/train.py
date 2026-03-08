import os
import sys
import time
import torch # type: ignore
from training.trainer import train

# Project root on path 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def log_environment() -> None:
    print("\n" + "=" * 60)
    print(" FortifyAI v5 — Deepfake Detection Training")
    print("=" * 60)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device      : CUDA — {device}")
        print(f"  VRAM        : {vram:.1f} GB")
    else:
        print(f"  Device      : CPU (WARNING: training will be very slow)")
        print(f"  Recommended : CUDA GPU with 6GB+ VRAM")
    print(f"  Resolution  : 224×224")
    print(f"  Batch size  : 24 (effective 48 with grad accumulation)")
    print("=" * 60 + "\n")


def main() -> None:
    log_environment()
    start = time.time()
    train()
    elapsed = time.time() - start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"\n  Total training time: {h:02d}h {m:02d}m {s:02d}s")


if __name__ == "__main__":
    main()