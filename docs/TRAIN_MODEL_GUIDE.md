# Train Model Guide - Step by Step

## Quick Start - Google Colab (Recommended!)

### Step 1: Open Colab Notebook

1. Go to: https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb
2. Or click "Open in Colab" button in the GitHub repository

### Step 2: Set Up GPU

1. **Runtime → Change runtime type**
2. **Hardware accelerator → GPU (T4)**
3. Click **Save**

### Step 3: Run All Cells

1. **Runtime → Run all**
2. Wait for training to complete (can take 10-30 minutes depending on dataset size)

### Step 4: Download the Model

After training completes, the model will be saved in `models/run_YYYYMMDD_HHMMSS/`

**To download to your computer:**

1. In Colab, run this in a new cell:
```python
from google.colab import files
import shutil
import glob

# Find the latest model
models_dir = sorted(glob.glob('models/run_*'))[-1]
print(f"Downloading: {models_dir}")

# Create a zip file
shutil.make_archive('trained_model', 'zip', models_dir)

# Download
files.download('trained_model.zip')
```

2. Or download manually:
   - Click the folder icon on the left (Files)
   - Navigate to `models/run_YYYYMMDD_HHMMSS/`
   - Right-click on `best_model.keras` → Download
   - Also download `label_mapping.json`

### Step 5: Upload Model to Railway

**Option 1: Add to GitHub (if model is small - <100MB)**

1. Extract the downloaded model
2. Add to your local repository:
   ```bash
   # Copy model to models/run_YYYYMMDD_HHMMSS/
   cp -r downloaded_model/* models/run_YYYYMMDD_HHMMSS/
   ```
3. Commit and push:
   ```bash
   git add models/
   git commit -m "Add trained model"
   git push
   ```

**Option 2: Upload via Railway (Recommended for large models)**

Railway will automatically use the model if it's in the GitHub repo. If the model is too large for GitHub:

1. Use Railway's file system (more complex - requires SSH)
2. Or use a storage service like S3/Google Drive and download on startup

**Option 3: Train Directly in Railway (Not Recommended)**

Railway doesn't have GPU, so training will be very slow.

---

## Training Parameters

The notebook uses these default parameters:
- Batch size: 32
- Epochs: 100
- GRU units: 128
- GRU layers: 2
- Dropout: 0.3
- Learning rate: 0.001
- Patience: 10 (early stopping)

You can modify these in the training cell if needed.

---

## Troubleshooting

### "No trained model found" in app

**Check:**
1. Model exists in `models/run_*/best_model.keras`
2. Model is committed to GitHub (if using GitHub deployment)
3. Model directory structure is correct:
   ```
   models/
   └── run_YYYYMMDD_HHMMSS/
       ├── best_model.keras
       └── label_mapping.json
   ```

### Training accuracy is low

**Possible causes:**
- Small dataset (<100 samples per class recommended)
- Need more data augmentation
- Need to adjust model parameters

**Solutions:**
- Collect more training videos
- Try different model parameters
- Check data quality

### Out of memory in Colab

**Solutions:**
- Reduce batch size (change to 16 or 8)
- Reduce GRU units (128 → 64)
- Use fewer GRU layers (2 → 1)

---

## After Training

Once you have a trained model:

1. ✅ Model is saved in `models/run_*/`
2. ✅ Download from Colab
3. ✅ Add to GitHub repo (if small) or upload to Railway
4. ✅ Redeploy on Railway
5. ✅ Test the app!

---

**Good luck training! 🚀**

