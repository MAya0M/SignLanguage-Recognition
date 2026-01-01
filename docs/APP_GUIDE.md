# מדריך אפליקציית Web - Sign Language Recognition

אפליקציית Web פשוטה להעלאת סרטונים ולקבלת תחזיות מהמודל המאומן.

## התקנה

### 1. התקן תלויות

```bash
pip install -r requirements.txt
```

### 2. אמן מודל (אם עדיין לא)

```bash
python scripts/train_model.py --csv Data/Labels/dataset.csv
```

או השתמש ב-Google Colab (ראה `notebooks/SignLanguage_Training.ipynb`)

### 3. הרץ את האפליקציה

```bash
python app.py
```

האפליקציה תרוץ על `http://localhost:5000`

---

## שימוש

### דרך הדפדפן

1. פתח את `http://localhost:5000` בדפדפן
2. גרור וזרוק סרטון (או לחץ כדי לבחור)
3. לחץ על "זהה שפת סימנים"
4. קבל את התוצאה!

### דרך API

```bash
# POST request
curl -X POST -F "video=@your_video.mp4" http://localhost:5000/predict
```

**Response:**
```json
{
  "success": true,
  "prediction": "HELLO",
  "confidence": 0.95,
  "all_predictions": [
    {"word": "HELLO", "confidence": 0.95},
    {"word": "YES", "confidence": 0.03},
    ...
  ]
}
```

---

## מבנה הקבצים

```
├── app.py                    # Flask application
├── templates/
│   └── index.html           # Web UI
├── scripts/
│   └── predict.py           # Prediction logic
└── models/
    └── run_*/               # Trained models
```

---

## הפעלה ב-Production

### עם Gunicorn (מומלץ)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### עם Docker

צור `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

בנה והרץ:
```bash
docker build -t sign-language-app .
docker run -p 5000:5000 sign-language-app
```

---

## הגדרות

### גודל קובץ מקסימלי

בקובץ `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

### תיקיית העלאות

```python
app.config['UPLOAD_FOLDER'] = 'temp/uploads'
```

### תיקיית מודלים

```python
app.config['MODEL_DIR'] = 'models'
```

---

## Troubleshooting

### "No trained model found"

אמן מודל תחילה:
```bash
python scripts/train_model.py --csv Data/Labels/dataset.csv
```

### שגיאת Upload

בדוק שגודל הקובץ לא עולה על המגבלה (100MB כברירת מחדל).

### שגיאת Memory

אם יש שגיאת זיכרון, נסה:
- הקטן את גודל הסרטון
- השתמש במודל קטן יותר
- הגדל את הזיכרון הזמין

---

**בהצלחה! 🚀**

