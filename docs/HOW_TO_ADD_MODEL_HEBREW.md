# 🎯 איך להביא את המודל המאומן לפרויקט - מדריך קצר

## שלבים מהירים:

### 1️⃣ הורד את המודל מ-Colab

בנוטבוק Colab, אחרי שהמודל סיים להתאמן, הרץ:

```python
from google.colab import files
import shutil
import glob
from pathlib import Path

models_dir = sorted(glob.glob('models/run_*'))
if models_dir:
    latest_run = models_dir[-1]
    zip_name = f"{Path(latest_run).name}"
    shutil.make_archive(zip_name, 'zip', latest_run)
    files.download(f'{zip_name}.zip')
    print("✅ הורדה הושלמה!")
```

### 2️⃣ חלץ את הקובץ

1. מצא את הקובץ `run_XXXXX.zip` שהורדת
2. לחץ עליו פעמיים לחילוץ
3. תקבל תיקייה בשם `run_XXXXX`

### 3️⃣ העתק לפרויקט

1. פתח את התיקייה: `C:\Users\madar\Documents\Ai-course\signlanguage\models`
2. העתק את התיקייה `run_XXXXX` לתוך `models/`

**המבנה צריך להיות:**
```
signlanguage/
└── models/
    ├── hand_landmarker.task
    └── run_XXXXX/          ← התיקייה החדשה
        ├── best_model.keras
        └── label_mapping.json
```

### 4️⃣ בדוק שהכל עובד

הרץ בטרמינל:
```bash
python utils/check_model.py
```

אם הכל תקין, תראה:
```
✅ Found 1 model run(s):
📁 run_XXXXX
   ✅ best_model.keras (XX.X MB)
   ✅ label_mapping.json
```

### 5️⃣ הפעל את האפליקציה

```bash
python app.py
```

פתח בדפדפן: `http://localhost:5000`

**האזהרה "No trained model found" אמורה להיעלם!** ✅

---

## 🔍 פתרון בעיות

### המודל לא נמצא?
- ודא שהתיקייה `run_XXXXX` נמצאת בתוך `models/`
- ודא שיש קובץ `best_model.keras` בתוך התיקייה

### בדיקה מהירה:
```bash
dir models\run_*\best_model.keras
```

אם אתה רואה את הקובץ - הכל תקין! ✅

---

**עזרה נוספת?** ראה `docs/HOW_TO_ADD_MODEL.md` למדריך מפורט יותר.

