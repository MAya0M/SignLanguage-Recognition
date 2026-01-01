# Sign Language Recognition System

מערכת לזיהוי שפת סימנים באמצעות GRU Neural Network.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb)

> **אוטומטי לחלוטין!** פשוט לחץ על הכפתור למעלה, בחר GPU, ו-Run all. הכל יעבוד אוטומטית! 🚀

## התחלה מהירה - Google Colab

**הדרך הקלה ביותר להתחיל:**

1. לחץ על הכפתור "Open in Colab" למעלה ⬆️
2. Runtime → Change runtime type → Select **GPU**
3. Run all cells (Runtime → Run all)

**זה הכל!** המודל יתאמן אוטומטית.

---

## מבנה הפרויקט

```
SignLanguage-Recognition/
├── Data/                    # נתונים
│   ├── Keypoints/          # Keypoints מופקים (.npy files)
│   ├── Labels/             # CSV files עם dataset splits
│   ├── rawVideos/          # סרטונים מקוריים
│   └── Sessions/           # סרטוני sessions
├── scripts/                # סקריפטים עיקריים
│   ├── extract_keypoints.py      # חילוץ keypoints מסרטונים
│   ├── create_dataset_csv.py     # יצירת CSV dataset
│   ├── train_model.py            # אימון מודל GRU
│   ├── predict.py                # חיזוי מסרטונים
│   ├── data_loader.py            # טעינת נתונים
│   └── model_gru.py              # ארכיטקטורת מודל
├── notebooks/              # Jupyter notebooks
│   └── SignLanguage_Training.ipynb  # Colab notebook אוטומטי
├── docs/                   # תיעוד
│   ├── README.md
│   ├── README_MODEL.md
│   ├── COLAB_UPLOAD_GUIDE.md
│   └── ...
├── models/                 # מודלים מאומנים
├── output/                 # פלטים (annotated videos, etc.)
├── utils/                  # כלי עזר
└── requirements.txt        # תלויות Python
```

---

## התקנה מקומית

### 1. Clone Repository

```bash
git clone https://github.com/MAya0M/SignLanguage-Recognition.git
cd SignLanguage-Recognition
```

### 2. התקן תלויות

```bash
# יצירת virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# התקנת תלויות
pip install -r requirements.txt
```

---

## שימוש

### Google Colab (מומלץ!) ⭐

**הדרך הכי קלה:**
1. לחץ על [Open in Colab](https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb)
2. Runtime → Change runtime type → **GPU**
3. Run all cells

**או:**
1. פתח [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub
3. הזן: `MAya0M/SignLanguage-Recognition`
4. בחר: `notebooks/SignLanguage_Training.ipynb`

### מקומי (אם יש GPU)

```bash
# 1. חילוץ keypoints
python scripts/extract_keypoints.py

# 2. יצירת dataset
python scripts/create_dataset_csv.py

# 3. אימון המודל
python scripts/train_model.py --csv Data/Labels/dataset.csv

# 4. חיזוי
python scripts/predict.py \
    --model models/run_*/best_model.keras \
    --video your_video.mp4
```

---

## תכונות עיקריות

✅ **נרמול מתקדם** - בלתי תלוי במיקום היד, גודל היד, וצד היד (שמאל/ימין)  
✅ **Google Colab** - GPU חינם, אימון אוטומטי  
✅ **מודל GRU** - לזיהוי sequences של תנועות יד  
✅ **חיזוי מסרטונים** - חיזוי ישירות מסרטונים או keypoints  
✅ **אפליקציית Web** - העלה סרטון וקבל תרגום דרך דפדפן! 🎬  

---

## תיעוד

- **[מדריך מודל](docs/README_MODEL.md)** - פרטים על המודל והאימון
- **[מדריך Colab](docs/COLAB_UPLOAD_GUIDE.md)** - איך להעלות נתונים ל-Colab
- **[מדריך יישום](docs/IMPLEMENTATION_GUIDE.md)** - מדריך יישום מלא
- **[הסבר מודל](docs/MODEL_EXPLANATION.md)** - איך המודל עובד
- **[מדריך אפליקציה](docs/APP_GUIDE.md)** - אפליקציית Web להעלאת סרטונים
- **[README אפליקציה](README_APP.md)** - התחלה מהירה לאפליקציה

---

## Workflow מלא

```bash
# 1. חילוץ keypoints
python scripts/extract_keypoints.py

# 2. יצירת dataset
python scripts/create_dataset_csv.py

# 3. אימון (Google Colab מומלץ!)
# לחץ על "Open in Colab" למעלה

# 4. הרצת אפליקציית Web
python app.py
# פתח http://localhost:5000 והעלה סרטון!

# או חיזוי דרך command line
python scripts/predict.py --model models/.../best_model.keras --video test.mp4
```

---

## דרישות

- Python 3.8+
- GPU (מומלץ לאימון) - Google Colab מספק GPU חינם!
- ~10GB disk space
- MediaPipe Hand Landmarker model (מורד אוטומטית)

---

## רישיון

פרויקט זה הוא למטרות לימוד.

---

## תמיכה

לשאלות ובעיות, ראה את המדריכים ב-`docs/`.

---

**בהצלחה! 🚀**
