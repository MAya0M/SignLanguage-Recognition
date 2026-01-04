# Sign Language Recognition System

Sign language recognition system using **CNN + LSTM** Neural Network.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb)

> **Fully Automatic!** Just click the button above, select GPU, and Run all. Everything will work automatically!

## Quick Start

### Train Model in Google Colab (Recommended!)

1. Click the **"Open in Colab"** button above
2. **Runtime → Change runtime type → Select GPU**
3. **Runtime → Run all**

**That's it!** The model will train automatically.

---

## Project Structure

```
SignLanguage-Recognition/
├── Data/                    # Data directory
│   ├── Keypoints/          # Extracted keypoints (.npy files)
│   ├── Labels/             # CSV files with dataset splits
│   └── rawVideos/          # Original videos
├── scripts/                # Main scripts
│   ├── extract_keypoints.py      # Extract keypoints from videos
│   ├── create_dataset_csv.py     # Create CSV dataset
│   ├── prepare_for_training.py    # Prepare data for training
│   ├── train_model.py            # Train CNN + LSTM model
│   ├── predict.py                # Predict from videos
│   ├── data_loader.py            # Data loading
│   └── model_cnn_lstm.py         # CNN + LSTM architecture
├── notebooks/              # Jupyter notebooks
│   └── SignLanguage_Training.ipynb  # Automatic Colab notebook
├── models/                 # Trained models (saved here)
├── app.py                  # Flask web application
└── requirements.txt        # Python dependencies
```

---

## How It Works

### 1. Extract Keypoints
- Uses MediaPipe to extract hand keypoints from videos
- Normalizes keypoints (minimal normalization - only translation)
- Saves as `.npy` files

### 2. Create Dataset
- Creates CSV file with video paths and labels
- Splits into train/val/test sets

### 3. Train Model
- **CNN + LSTM architecture:**
  - **CNN** - Recognizes spatial patterns (how keypoints are arranged)
  - **LSTM** - Recognizes temporal patterns (how movement changes over time)
- Trains on Google Colab with free GPU

### 4. Predict
- Use live camera through web app
- Or use command line: `python scripts/predict.py --model models/.../best_model.keras --video test.mp4`

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

### 3. Prepare Data

```bash
# Extract keypoints from videos
python scripts/extract_keypoints.py

# Create dataset CSV
python scripts/create_dataset_csv.py

# Or use the all-in-one script
python scripts/prepare_for_training.py
```

### 4. Train Model

**Option A: Google Colab (Recommended!)**
- Click "Open in Colab" button above
- Select GPU and run all cells

**Option B: Local (if you have GPU)**

```bash
python scripts/train_model.py \
    --csv Data/Labels/dataset.csv \
    --keypoints-dir Data/Keypoints/rawVideos \
    --output-dir models \
    --batch-size 8 \
    --epochs 200 \
    --cnn-filters 64 \
    --lstm-units 128 \
    --num-cnn-layers 2 \
    --dropout 0.3 \
    --learning-rate 0.001
```

### 5. Run Web App

```bash
python app.py
# Open http://localhost:5000 and upload a video!
```

---

## Using the Web App

### Access from Computer

1. **Start the app:**
   ```bash
   python app.py
   ```

2. **Open browser:** `http://localhost:5000`

3. **Use live camera:**
   - Click "Start Recognition" for real-time sign language translation

### Access from Phone (for Live Camera)

**Important:** Modern browsers require **HTTPS** to access the camera. Use one of these solutions:

#### Option 1: Using ngrok (Recommended - Easiest!)

1. **Install ngrok:**
   - Download from: https://ngrok.com/download
   - Or use: `choco install ngrok` (Windows with Chocolatey)
   - Or: `winget install ngrok`

2. **Start the Flask app:**
   ```bash
   python app.py
   ```

3. **In a new terminal, start ngrok:**
   ```bash
   ngrok http 5000
   ```

4. **Copy the HTTPS URL** from ngrok (looks like: `https://abc123.ngrok.io`)

5. **Open that URL on your phone** - Camera will work!

#### Option 2: Using Mobile Hotspot

1. **Create hotspot on your computer:**
   - Windows: Settings → Network & Internet → Mobile hotspot → On

2. **Connect your phone to the hotspot**

3. **Start the app:**
   ```bash
   python app.py
   ```

4. **Use the IP address shown** (e.g., `http://192.168.137.1:5000`)

5. **Note:** You'll still need HTTPS for camera. Use ngrok with the hotspot IP:
   ```bash
   ngrok http 192.168.137.1:5000
   ```

#### Option 3: Using Phone's Hotspot

1. **Turn on hotspot on your phone**

2. **Connect your computer to the phone's hotspot**

3. **Start the app:**
   ```bash
   python app.py
   ```

4. **Use the IP address shown** on your phone's browser

5. **Note:** You'll still need HTTPS for camera. Use ngrok.

---

4. **Get predictions:**
   - The app will show all recognized words from the video

---

## Model Architecture

### CNN + LSTM Model

**Why CNN + LSTM?**
- **CNN** - Recognizes spatial patterns (how keypoints are arranged in each frame)
- **LSTM** - Recognizes temporal patterns (how movement changes over time)
- **Combined** - Better accuracy than GRU alone

**Architecture:**
```
Input (96 frames, 126 features)
  ↓
CNN Layers (spatial pattern recognition)
  ├── Conv1D (64 filters) → BatchNorm → MaxPool → Dropout
  └── Conv1D (128 filters) → BatchNorm → MaxPool → Dropout
  ↓
Global Average Pooling
  ↓
Bidirectional LSTM (128 units) - temporal pattern recognition
  ↓
Dense Layers
  ├── Dense (128) → Dropout
  └── Dense (64) → Dropout
  ↓
Output (8 classes)
```

---

## Adding New Videos

1. **Add videos to `Data/rawVideos/[WordName]/`:**
   ```
   Data/rawVideos/
   ├── Hello/
   │   ├── Hello01.mp4
   │   ├── Hello02.mp4
   │   └── ...
   ├── Yes/
   │   ├── Yes01.mp4
   │   └── ...
   └── ...
   ```

2. **Extract keypoints:**
   ```bash
   python scripts/extract_keypoints.py
   ```

3. **Update CSV:**
   ```bash
   python scripts/create_dataset_csv.py
   ```

4. **Or use all-in-one:**
   ```bash
   python scripts/prepare_for_training.py
   ```

5. **Retrain model** (in Colab or locally)

---

## Model Parameters

Default parameters (optimized for small datasets):

- **Batch size:** 8
- **Epochs:** 200
- **CNN filters:** 64 (first layer), 128 (second layer)
- **LSTM units:** 128
- **CNN layers:** 2
- **Dropout:** 0.3
- **Learning rate:** 0.001

**To change parameters:**
```bash
python scripts/train_model.py \
    --cnn-filters 128 \
    --lstm-units 256 \
    --num-cnn-layers 3 \
    --dropout 0.4 \
    --learning-rate 0.0005
```

---

## Adding Trained Model to Project

After training in Colab:

1. **Download model from Colab:**
   - Run the download cell in the notebook
   - Or download from Google Drive

2. **Extract model to project:**
   ```bash
   # Extract the zip file
   unzip run_YYYYMMDD_HHMMSS.zip
   
   # Move to models directory
   mv run_YYYYMMDD_HHMMSS models/
   ```

3. **Use in web app:**
   - The app will automatically find the latest model in `models/run_*/best_model.keras`

---

## Troubleshooting

### Model stuck at 12.5% accuracy?

**12.5% = 1/8 classes = random guessing**

**Possible causes:**
1. **Not enough data** - Need at least 20-30 videos per word
2. **Normalization too aggressive** - Try minimal normalization (only translation)
3. **Classes too similar** - Add more distinctive videos

**Solutions:**
1. Add more training videos (30-50 per word)
2. Try re-extracting keypoints with minimal normalization:
   ```bash
   python scripts/re_extract_with_minimal_normalization.py
   ```
3. Check data quality - make sure videos are clear and consistent

### Can't find model?

- Make sure model is in `models/run_*/best_model.keras`
- Check that `label_mapping.json` exists in the same directory

### GPU not available?

- Use Google Colab (free GPU!)
- Or train on CPU (will be slower, 2-4 hours)

---

## Requirements

- Python 3.8+
- GPU (recommended for training) - Google Colab provides free GPU!
- ~10GB disk space
- MediaPipe Hand Landmarker model (downloaded automatically)

---

## Deployment

### Deploy Web App to Railway (Free)

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select this repository
5. Railway will auto-detect Flask app
6. Click "Deploy" - **That's it!**

Your app will be live at: `https://your-app-name.railway.app`

**Note:** Make sure to train a model first and add it to the repository!

---

## License

This project is for educational purposes.

---

## Support

For questions and issues:
1. Check the troubleshooting section above
2. Review the code comments
3. Check Google Colab notebook for detailed steps

---

**Good luck!**
