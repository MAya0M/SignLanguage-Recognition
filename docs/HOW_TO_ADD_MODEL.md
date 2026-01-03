# איך להביא את המודל המאומן לפרויקט

מדריך זה מסביר איך להוריד את המודל מ-Google Colab ולהעביר אותו לפרויקט המקומי.

## שלב 1: הורדת המודל מ-Colab

### אופציה A: הורדה ישירה (מומלץ)

1. **בנוטבוק Colab**, אחרי שהמודל סיים להתאמן, הרץ את התא הזה:

```python
# Download model to your computer
from google.colab import files
import shutil
import glob
from pathlib import Path

models_dir = sorted(glob.glob('models/run_*'))
if models_dir:
    latest_run = models_dir[-1]  # Latest run
    print(f"📦 Preparing model: {Path(latest_run).name}")
    
    # Create a zip file
    zip_name = f"{Path(latest_run).name}"
    shutil.make_archive(zip_name, 'zip', latest_run)
    
    # Download
    print(f"⬇️ Downloading {zip_name}.zip...")
    files.download(f'{zip_name}.zip')
    print("✅ Model downloaded! Extract and add to your project.")
else:
    print("❌ No models found - train the model first!")
```

2. הקובץ `run_XXXXX.zip` יורד אוטומטית למחשב שלך.

### אופציה B: העברה ל-Google Drive

אם אתה רוצה לשמור ב-Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
import glob
from pathlib import Path

models_dir = sorted(glob.glob('models/run_*'))
if models_dir:
    latest_run = models_dir[-1]
    dest = f'/content/drive/MyDrive/{Path(latest_run).name}'
    shutil.copytree(latest_run, dest, dirs_exist_ok=True)
    print(f"✅ Model saved to Google Drive: {Path(latest_run).name}")
```

## שלב 2: העברת המודל לפרויקט

### צעד 1: פתח את הקובץ שהורדת

1. מצא את הקובץ `run_XXXXX.zip` שהורדת
2. חלץ את הקובץ (Extract) - תקבל תיקייה בשם `run_XXXXX`

### צעד 2: העתק את התיקייה לפרויקט

1. פתח את תיקיית הפרויקט: `C:\Users\madar\Documents\Ai-course\signlanguage`
2. פתח את התיקייה `models` (אם היא לא קיימת, צור אותה)
3. העתק את התיקייה `run_XXXXX` לתוך `models/`

**המבנה הסופי צריך להיות:**
```
signlanguage/
├── models/
│   ├── hand_landmarker.task
│   └── run_XXXXX/          ← התיקייה החדשה
│       ├── best_model.keras
│       ├── label_mapping.json
│       ├── model_architecture.json
│       └── ... (קבצים נוספים)
├── app.py
└── ...
```

### צעד 3: בדוק שהמודל נמצא

הרץ את הפקודה הזו בטרמינל:

```bash
python -c "from pathlib import Path; models = list(Path('models').glob('run_*/best_model.keras')); print('✅ Found models:' if models else '❌ No models found'); [print(f'  - {m}') for m in models]"
```

או פשוט בדוק ידנית:
- פתח את `models/run_XXXXX/`
- ודא שיש את הקבצים:
  - ✅ `best_model.keras`
  - ✅ `label_mapping.json`

## שלב 3: הפעל את האפליקציה

1. הפעל את האפליקציה:
   ```bash
   python app.py
   ```

2. פתח בדפדפן: `http://localhost:5000`

3. **האזהרה "No trained model found" אמורה להיעלם!** ✅

## פתרון בעיות

### המודל לא נמצא

**בעיה:** האפליקציה עדיין מציגה "No trained model found"

**פתרון:**
1. ודא שהתיקייה `run_XXXXX` נמצאת בתוך `models/`
2. ודא שיש קובץ `best_model.keras` בתוך התיקייה
3. בדוק את שם התיקייה - צריך להתחיל ב-`run_`

**בדיקה מהירה:**
```bash
dir models\run_*\best_model.keras
```

### שגיאת טעינת מודל

**בעיה:** שגיאה בעת טעינת המודל

**פתרון:**
1. ודא שהתקנת את כל התלויות: `pip install -r requirements.txt`
2. ודא שיש קובץ `label_mapping.json` באותה תיקייה
3. בדוק את גרסת TensorFlow - צריך להיות תואם לגרסה שבה אימנת

### המודל לא עובד

**בעיה:** המודל נטען אבל לא עושה תחזיות נכונות

**פתרון:**
1. ודא שהמודל אומן על אותו סוג נתונים
2. בדוק את `label_mapping.json` - התוויות צריכות להתאים
3. נסה לאמן מודל חדש עם יותר epochs

## טיפים

- **שמור גיבוי:** לפני שאתה מחליף מודל, שמור עותק של המודל הישן
- **מספר מודלים:** אתה יכול להשאיר כמה תיקיות `run_*` - האפליקציה תבחר את החדש ביותר
- **Git:** אם אתה משתמש ב-Git, הוסף את `models/run_*/` ל-`.gitignore` (המודלים גדולים מדי)

## מבנה קבצים נדרש

```
models/
└── run_20240101_120000/     ← תיקיית המודל
    ├── best_model.keras     ← המודל (חובה!)
    ├── label_mapping.json   ← מיפוי תוויות (חובה!)
    ├── model_architecture.json
    ├── training_history.json
    └── ...
```

**רק `best_model.keras` ו-`label_mapping.json` הם חובה!** הקבצים האחרים הם אופציונליים.

---

✅ **אחרי שתעשה את זה, האפליקציה תוכל לזהות שפת סימנים!**

