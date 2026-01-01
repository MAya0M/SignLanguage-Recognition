# Sign Language Recognition System

Sign language recognition system using GRU Neural Network.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb)

> **Fully Automatic!** Just click the button above, select GPU, and Run all. Everything will work automatically! 🚀

> **✅ Production Ready!** The app is tested and deployed on Railway. All dependencies are configured for cloud deployment.

## 🚀 Quick Start

### Option 1: Use Online App (Recommended!)

**The app is ready to deploy!** Follow these steps:

1. **Deploy to Railway (Free):**
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select this repository
   - Railway will auto-detect Flask app
   - Click "Deploy" - **That's it!**
   - Your app will be live at: `https://your-app-name.railway.app`

2. **Or Deploy to Render (Free):**
   - Go to [render.com](https://render.com)
   - Sign in with GitHub
   - Click "New" → "Web Service"
   - Select this repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Click "Create Web Service"

**See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for detailed instructions.**

### Option 2: Train Model in Google Colab

1. Click the "Open in Colab" button above ⬆️
2. Runtime → Change runtime type → Select **GPU**
3. Run all cells (Runtime → Run all)

**That's it!** The model will train automatically.

---

## Project Structure

```
SignLanguage-Recognition/
├── Data/                    # Data
│   ├── Keypoints/          # Extracted keypoints (.npy files)
│   ├── Labels/             # CSV files with dataset splits
│   ├── rawVideos/          # Original videos
│   └── Sessions/           # Session videos
├── scripts/                # Main scripts
│   ├── extract_keypoints.py      # Extract keypoints from videos
│   ├── create_dataset_csv.py     # Create CSV dataset
│   ├── train_model.py            # Train GRU model
│   ├── predict.py                # Predict from videos
│   ├── data_loader.py            # Data loading
│   └── model_gru.py              # Model architecture
├── notebooks/              # Jupyter notebooks
│   └── SignLanguage_Training.ipynb  # Automatic Colab notebook
├── docs/                   # Documentation
│   ├── README.md
│   ├── README_MODEL.md
│   ├── COLAB_UPLOAD_GUIDE.md
│   └── ...
├── models/                 # Trained models
├── output/                 # Outputs (annotated videos, etc.)
├── utils/                  # Utilities
└── requirements.txt        # Python dependencies
```

---

## Local Installation

### 1. Clone Repository

```bash
git clone https://github.com/MAya0M/SignLanguage-Recognition.git
cd SignLanguage-Recognition
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Google Colab (Recommended!) ⭐

**The easiest way:**
1. Click [Open in Colab](https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb)
2. Runtime → Change runtime type → **GPU**
3. Run all cells

**Or:**
1. Open [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub
3. Enter: `MAya0M/SignLanguage-Recognition`
4. Select: `notebooks/SignLanguage_Training.ipynb`

### Local (if you have GPU)

```bash
# 1. Extract keypoints
python scripts/extract_keypoints.py

# 2. Create dataset
python scripts/create_dataset_csv.py

# 3. Train model
python scripts/train_model.py --csv Data/Labels/dataset.csv

# 4. Run web app
python app.py
# Open http://localhost:5000 and upload a video!

# Or predict via command line
python scripts/predict.py \
    --model models/run_*/best_model.keras \
    --video your_video.mp4
```

---

## Features

✅ **Web Application** - Upload video and get translation through browser! 🎬 **[Deploy Online Now!](#-quick-start)**  
✅ **Advanced Normalization** - Invariant to hand position, size, and hand side (left/right)  
✅ **Google Colab** - Free GPU, automatic training  
✅ **GRU Model** - For recognizing sequences of hand movements  
✅ **Video Prediction** - Predict directly from videos or keypoints  
✅ **Automatic CI/CD** - GitHub Actions validates code on every push  
✅ **Automatic Deployment Ready** - One-click deploy to Railway/Render/Heroku  

---

## Documentation

- **[Model Guide](docs/README_MODEL.md)** - Model details and training
- **[Colab Guide](docs/COLAB_UPLOAD_GUIDE.md)** - How to upload data to Colab
- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Full implementation guide
- **[Model Explanation](docs/MODEL_EXPLANATION.md)** - How the model works
- **[Web App Guide](docs/APP_GUIDE.md)** - Web application for uploading videos
- **[App README](README_APP.md)** - Quick start for the app
- **[GitHub Actions Guide](docs/GITHUB_ACTIONS_EXPLAINED.md)** - What happens after workflow completes
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Deploy to production

---

## Workflow

```bash
# 1. Extract keypoints
python scripts/extract_keypoints.py

# 2. Create dataset
python scripts/create_dataset_csv.py

# 3. Train (Google Colab recommended!)
# Click "Open in Colab" above

# 4. Run web app
python app.py
# Open http://localhost:5000 and upload a video!

# Or predict via command line
python scripts/predict.py --model models/.../best_model.keras --video test.mp4
```

---

## Requirements

- Python 3.8+
- GPU (recommended for training) - Google Colab provides free GPU!
- ~10GB disk space
- MediaPipe Hand Landmarker model (downloaded automatically)

---

## License

This project is for educational purposes.

---

## Support

For questions and issues, see the guides in `docs/`.

---

## GitHub Actions

Every push to GitHub automatically triggers:
- ✅ Syntax checking
- ✅ Import checking
- ✅ Project structure validation

For more details: see [docs/GITHUB_ACTIONS_EXPLAINED.md](docs/GITHUB_ACTIONS_EXPLAINED.md)

## 🌐 View App Online

**Your app is ready to deploy!** Get it online in 5 minutes:

### Quick Deploy (Railway - Recommended):
1. Go to [railway.app](https://railway.app) → Sign in with GitHub
2. New Project → Deploy from GitHub repo
3. Select this repository → Deploy
4. **Done!** Your app will be live at `https://your-app.railway.app`

**See [docs/QUICK_DEPLOY.md](docs/QUICK_DEPLOY.md) for step-by-step instructions.**

### Other Options:
- **Render:** [render.com](https://render.com) - Free tier available
- **Heroku:** [heroku.com](https://heroku.com) - Free tier available

**Note:** Make sure to train a model first! See [Training Guide](docs/README_MODEL.md) or use [Google Colab](https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb).

---

**Good luck! 🚀**
