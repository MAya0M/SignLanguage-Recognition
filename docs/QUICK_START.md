# Quick Start Guide - Sign Language Recognition

## הדרך המהירה ביותר להתחיל 🚀

### Google Colab (מומלץ! ⭐)

**כל מה שצריך:**

1. לחץ על [Open in Colab](https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb)
2. **Runtime → Change runtime type → GPU**
3. **Runtime → Run all**

**זה הכל!** המודל יתאמן אוטומטית.

---

### מקומי (אם יש GPU)

```bash
# 1. Clone repository
git clone https://github.com/MAya0M/SignLanguage-Recognition.git
cd SignLanguage-Recognition

# 2. Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Extract keypoints (if not done)
python scripts/extract_keypoints.py

# 5. Create dataset
python scripts/create_dataset_csv.py

# 6. Train model
python scripts/train_model.py --csv Data/Labels/dataset.csv

# 7. Predict
python scripts/predict.py \
    --model models/run_*/best_model.keras \
    --video your_video.mp4
```

---

## מה הלאה?

- **[מדריך יישום מלא](IMPLEMENTATION_GUIDE.md)** - כל השלבים בפירוט
- **[מדריך Colab](COLAB_UPLOAD_GUIDE.md)** - איך להעלות נתונים
- **[הסבר מודל](MODEL_EXPLANATION.md)** - איך המודל עובד

