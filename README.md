# Fortifyai
Hybrid Spatial-Frequency Deepfake Detection Framework

## Environment Setup

This project uses `.env` for path configuration.

### Steps

1. Install python-dotenv
```bash
pip install python-dotenv
```

2. Copy the example file
```bash
copy .env.example .env

3. Open `.env` and fill in your paths
```bash
notepad .env
```
```
FORTIFY_DATASETS=your_datasets_folder_path
FORTIFY_OUTPUT=your_processed_data_folder_path
```

4. Verify paths loaded correctly
```bash
python -c "
from preprocessing.config import OUTPUT_FFPP, OUTPUT_CELEB, OUTPUT_GAN
print('FF++  :', OUTPUT_FFPP)
print('Celeb :', OUTPUT_CELEB)
print('GAN   :', OUTPUT_GAN)
"
```