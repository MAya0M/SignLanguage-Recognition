# Train Your Model Now! 🚀

## Quick Steps

### 1. Open Google Colab

Click here: **[Open in Colab](https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb)**

Or go to GitHub and click the "Open in Colab" button.

### 2. Enable GPU

1. **Runtime → Change runtime type**
2. **Hardware accelerator → GPU (T4)**
3. **Save**

### 3. Run Training

1. **Runtime → Run all**
2. Wait 10-30 minutes (depending on your dataset size)

### 4. Download Model

After training, run this in a new Colab cell:

```python
from google.colab import files
import shutil
import glob

# Find latest model
models_dir = sorted(glob.glob('models/run_*'))[-1]
print(f"Downloading: {models_dir}")

# Create zip
shutil.make_archive('trained_model', 'zip', models_dir)

# Download
files.download('trained_model.zip')
```

### 5. Upload to GitHub

1. Extract the downloaded zip
2. Copy to your local repository:
   ```bash
   # Create directory
   mkdir -p models/run_YYYYMMDD_HHMMSS
   
   # Copy files
   cp downloaded_model/* models/run_YYYYMMDD_HHMMSS/
   ```

3. Commit and push:
   ```bash
   git add models/
   git commit -m "Add trained model"
   git push
   ```

### 6. Railway Will Auto-Deploy

Once pushed, Railway will automatically redeploy with your new model!

---

## That's It!

After Railway redeploys, refresh your app and the warning will be gone!

---

**See [docs/TRAIN_MODEL_GUIDE.md](docs/TRAIN_MODEL_GUIDE.md) for detailed instructions.**

