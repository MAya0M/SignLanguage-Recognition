# מדריך יישום מלא - Sign Language Recognition

מדריך שלב-אחר-שלב ליישום הפרויקט ואימון המודל.

## תוכן עניינים

1. [הכנת הסביבה](#1-הכנת-הסביבה)
2. [חילוץ Keypoints](#2-חילוץ-keypoints)
3. [יצירת Dataset](#3-יצירת-dataset)
4. [אימון המודל](#4-אימון-המודל)
5. [שימוש במודל](#5-שימוש-במודל)
6. [אימון ב-Google Colab](#6-אימון-ב-google-colab)

---

## 1. הכנת הסביבה

### 1.1 התקנת Python

```bash
# בדוק שיש Python 3.8+
python --version

# אם לא, הורד מ-python.org
```

### 1.2 יצירת Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 1.3 התקנת תלויות

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**תלויות עיקריות:**
- `opencv-python` - עיבוד וידאו
- `mediapipe` - זיהוי ידיים
- `tensorflow` - אימון מודל
- `numpy`, `pandas`, `scikit-learn` - עיבוד נתונים

---

## 2. חילוץ Keypoints

### 2.1 הכנת סרטונים

הניח את הסרטונים בתיקייה:
```
Data/rawVideos/
├── Hello/
│   ├── Hello01.mp4
│   ├── Hello02.mp4
│   └── ...
├── Yes/
└── ...
```

### 2.2 הרצת חילוץ

```bash
python scripts/extract_keypoints.py
```

**מה קורה:**
1. הסקריפט עובר על כל הסרטונים
2. מפיק keypoints באמצעות MediaPipe
3. מנרמל את ה-keypoints (מיקום, גודל, צד היד)
4. שומר כ-`.npy` files ב-`Data/Keypoints/rawVideos/`

**פורמט נתונים:**
- כל סרטון → קובץ `.npy`
- צורה: `(num_frames, 2, 21, 3)`
  - `num_frames`: מספר frames
  - `2`: מספר ידיים (תמיד 2 slots)
  - `21`: keypoints לכל יד
  - `3`: קואורדינטות (x, y, z)

**נרמול:**
- ✅ Wrist ב-(0,0,0) - לא תלוי במיקום
- ✅ Scale לפי גודל היד - לא תלוי בגודל
- ✅ Mirror left/right - לא תלוי בצד היד
- ✅ Rotation alignment - לא תלוי בכיוון

### 2.3 בדיקת תוצאות

```bash
# בדוק כמה קבצים נוצרו
python -c "from pathlib import Path; files = list(Path('Data/Keypoints/rawVideos').rglob('*.npy')); print(f'Total files: {len(files)}')"
```

---

## 3. יצירת Dataset

### 3.1 יצירת CSV

```bash
python scripts/create_dataset_csv.py
```

**מה קורה:**
1. מוצא את כל קבצי `.npy`
2. מחלק ל-train/val/test (60%/20%/20%)
3. יוצר `Data/Labels/dataset.csv`

**פורמט CSV:**
```csv
path,label,split
keypoints/GoodBye/goodbye02.npy,GOODBYE,train
keypoints/Hello/hello01.npy,HELLO,test
...
```

### 3.2 בדיקת Dataset

```bash
# בדוק כמה samples בכל split
python -c "import pandas as pd; df = pd.read_csv('Data/Labels/dataset.csv'); print(df.groupby(['label', 'split']).size())"
```

---

## 4. אימון המודל

### 4.1 אימון מקומי (אם יש GPU)

```bash
python scripts/train_model.py \
    --csv Data/Labels/dataset.csv \
    --keypoints-dir Data/Keypoints/rawVideos \
    --output-dir models \
    --batch-size 32 \
    --epochs 100 \
    --gru-units 128 \
    --num-gru-layers 2 \
    --dropout 0.3 \
    --learning-rate 0.001 \
    --patience 10
```

**פרמטרים:**
- `--batch-size`: גודל batch (32 מומלץ)
- `--epochs`: מספר מקסימלי של epochs
- `--gru-units`: מספר יחידות ב-GRU (128 מומלץ)
- `--num-gru-layers`: מספר שכבות GRU (2 מומלץ)
- `--dropout`: Dropout rate (0.3 מומלץ)
- `--learning-rate`: Learning rate (0.001 מומלץ)
- `--patience`: Early stopping patience (10 מומלץ)

### 4.2 מה קורה באימון

1. **טעינת נתונים**: טוען keypoints מה-CSV
2. **Preprocessing**: Padding sequences לאותו אורך
3. **אימון**: GRU model עם callbacks:
   - ModelCheckpoint - שומר את המודל הטוב ביותר
   - EarlyStopping - עוצר אם אין שיפור
   - ReduceLROnPlateau - מקטין learning rate

### 4.3 תוצאות

לאחר אימון, בתיקיית `models/run_TIMESTAMP/`:
- `best_model.keras` - המודל הטוב ביותר
- `final_model.keras` - המודל מהאימון האחרון
- `label_mapping.json` - מיפוי labels
- `training_history.json` - היסטוריית אימון
- `test_results.json` - תוצאות על test set

---

## 5. שימוש במודל

### 5.1 חיזוי מסרטון

```bash
python scripts/predict.py \
    --model models/run_YYYYMMDD_HHMMSS/best_model.keras \
    --video path/to/video.mp4
```

### 5.2 חיזוי מ-keypoints

```bash
python scripts/predict.py \
    --model models/run_YYYYMMDD_HHMMSS/best_model.keras \
    --keypoints Data/Keypoints/rawVideos/Hello/Hello01.npy
```

### 5.3 שמירת תוצאות

```bash
python scripts/predict.py \
    --model models/run_YYYYMMDD_HHMMSS/best_model.keras \
    --video video.mp4 \
    --output results.json
```

---

## 6. אימון ב-Google Colab

### 6.1 פתיחת Notebook ב-Colab

**הדרך הקלה ביותר:**

1. לך ל-[Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub
3. הזן: `MAya0M/SignLanguage-Recognition`
4. בחר: `notebooks/SignLanguage_Training.ipynb`

**או פשוט לחץ על הכפתור "Open in Colab" ב-README!**

ראה [COLAB_AUTOMATIC_SETUP.md](COLAB_AUTOMATIC_SETUP.md) למדריך מפורט.

### 6.2 הגדרת GPU

1. **Runtime → Change runtime type**
2. **Hardware accelerator → GPU (T4)**
3. **Save**

### 6.3 העלאת נתונים

**דרך 1: Google Drive (מומלץ)**
- העלה את `sign_language_data.tar.gz` ל-Google Drive
- Mount Drive ב-Colab
- העתק את הקובץ לתיקיית הפרויקט

**דרך 2: ישירות ב-Colab**
- Files → Upload to session storage
- העלה את הקבצים הנדרשים

למדריך מפורט, ראה [COLAB_UPLOAD_GUIDE.md](COLAB_UPLOAD_GUIDE.md)

### 6.4 אימון

פשוט **Runtime → Run all** - הכל אוטומטי!

המודל יתאמן ויישמר בתיקיית `models/`.

---

## טיפים לשיפור

### 1. הגדלת Dataset

- אוסף יותר סרטונים לכל מילה
- וריאציות: זוויות שונות, אנשים שונים
- Data Augmentation: rotations, scaling

### 2. שיפור המודל

- הגדל `gru-units` (256, 512)
- הוסף שכבות GRU (`--num-gru-layers 3`)
- נסה Attention mechanisms
- נסה Transformer במקום GRU

### 3. אופטימיזציה

- Mixed Precision Training
- Gradient Accumulation
- Learning Rate Scheduling

---

## Troubleshooting

### בעיות נפוצות

**Out of Memory:**
```bash
# הקטן batch size
--batch-size 16
```

**Overfitting:**
```bash
# הגדל dropout
--dropout 0.5

# הוסף regularization
```

**Underfitting:**
```bash
# הגדל מספר layers/units
--gru-units 256
--num-gru-layers 3
```

**קבצים לא נמצאים:**
```bash
# בדוק נתיבים
python utils/verify_csv_files.py
```

---

## Workflow מלא

```bash
# 1. חילוץ keypoints
python scripts/extract_keypoints.py

# 2. יצירת dataset
python scripts/create_dataset_csv.py

# 3. אימון (מקומי או Google Colab)
python scripts/train_model.py --csv Data/Labels/dataset.csv

# 4. חיזוי
python scripts/predict.py --model models/.../best_model.keras --video test.mp4
```

---

## Next Steps

1. ✅ הכן את הסביבה
2. ✅ חלץ keypoints מהסרטונים
3. ✅ צור dataset CSV
4. ✅ אמן את המודל (מקומי או Google Colab)
5. ✅ בדוק את המודל על סרטונים חדשים
6. 🔄 שפר את המודל לפי הצורך

**בהצלחה! 🚀**

