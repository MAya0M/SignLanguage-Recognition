# איך לבדוק ולנהל מקום ב-Google Colab

## בדיקת מקום פנוי

### 1. בדוק כמה מקום יש

```python
# בדוק מקום פנוי
import shutil

# בדוק מקום בדיסק
total, used, free = shutil.disk_usage("/")
print(f"Total: {total // (1024**3)} GB")
print(f"Used: {used // (1024**3)} GB")
print(f"Free: {free // (1024**3)} GB")

# בדוק מקום ב-Google Drive (אם מחובר)
try:
    drive_total, drive_used, drive_free = shutil.disk_usage("/content/drive/MyDrive")
    print(f"\nGoogle Drive:")
    print(f"Total: {drive_total // (1024**3)} GB")
    print(f"Used: {drive_used // (1024**3)} GB")
    print(f"Free: {drive_free // (1024**3)} GB")
except:
    print("\nGoogle Drive not mounted")
```

### 2. בדוק מה תופס מקום

```python
# מצא קבצים גדולים
import os
from pathlib import Path

def get_size(path):
    """Get size of file or directory in MB"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total / (1024 * 1024)
    return 0

# בדוק גודל תיקיות חשובות
paths_to_check = [
    "/content",
    "/content/Data",
    "/content/models",
    "/content/SignLanguage-Recognition"
]

print("Directory sizes:")
for path in paths_to_check:
    if os.path.exists(path):
        size_mb = get_size(path)
        print(f"  {path}: {size_mb:.1f} MB")
```

## ניקוי מקום

### 1. מחק קבצים זמניים

```python
# מחק קבצים זמניים
import shutil
import os

# מחק __pycache__
!find /content -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true

# מחק .ipynb_checkpoints
!find /content -type d -name .ipynb_checkpoints -exec rm -r {} + 2>/dev/null || true

# מחק קבצי .pyc
!find /content -name "*.pyc" -delete 2>/dev/null || true

print("✅ Temporary files cleaned")
```

### 2. מחק מודלים ישנים

```python
# מחק מודלים ישנים (שמור רק את האחרון)
import glob
import shutil
from pathlib import Path

models = sorted(glob.glob('/content/SignLanguage-Recognition/models/run_*'))
if len(models) > 1:
    print(f"Found {len(models)} model runs")
    print("Keeping latest, deleting others...")
    for model_path in models[:-1]:  # Keep last one
        shutil.rmtree(model_path)
        print(f"  Deleted: {Path(model_path).name}")
    print(f"✅ Kept: {Path(models[-1]).name}")
else:
    print("Only one model found, nothing to delete")
```

### 3. מחק את ה-repository הישן

```python
# אם אתה רוצה להתחיל מחדש
import shutil

# מחק את ה-repository הישן
if os.path.exists("/content/SignLanguage-Recognition"):
    shutil.rmtree("/content/SignLanguage-Recognition")
    print("✅ Old repository deleted")

# Clone מחדש
!git clone https://github.com/MAya0M/SignLanguage-Recognition.git
%cd SignLanguage-Recognition
```

## הגדלת מקום ב-Colab

### 1. Colab Pro/Pro+ (מומלץ)

- **Colab Pro:** $10/חודש - יותר RAM, יותר GPU time
- **Colab Pro+:** $50/חודש - עוד יותר resources
- **קישור:** https://colab.research.google.com/signup

### 2. Google Drive (חינם)

- **15GB חינם** - מספיק לנתונים
- **100GB:** $2/חודש
- **200GB:** $3/חודש
- **2TB:** $10/חודש

### 3. אחסון נתונים ב-Google Drive

```python
# העבר נתונים ל-Google Drive
from google.colab import drive
import shutil

drive.mount('/content/drive')

# העתק נתונים ל-Drive
if os.path.exists("/content/Data"):
    dest = "/content/drive/MyDrive/SignLanguage_Data"
    shutil.copytree("/content/Data", dest, dirs_exist_ok=True)
    print("✅ Data copied to Google Drive")
    
    # מחק מהדיסק המקומי
    shutil.rmtree("/content/Data")
    print("✅ Local data deleted")
    
    # צור symbolic link
    os.symlink(dest, "/content/Data")
    print("✅ Created symbolic link")
```

## טיפים לחיסכון במקום

1. **אל תשמור מודלים ב-Colab** - הורד אותם למחשב שלך
2. **מחק מודלים ישנים** - שמור רק את הטוב ביותר
3. **השתמש ב-Google Drive** - לנתונים גדולים
4. **נקה קבצים זמניים** - אחרי כל אימון

## אם אין מספיק מקום

1. **מחק מודלים ישנים** - זה תופס הכי הרבה מקום
2. **העבר נתונים ל-Google Drive** - אם יש לך מקום שם
3. **קנה Colab Pro** - אם אתה צריך יותר מקום
4. **השתמש ב-Google Drive Storage** - זול יותר מ-Colab Pro

## בדיקה מהירה

```python
# הרץ את זה כדי לבדוק מקום
import shutil
total, used, free = shutil.disk_usage("/")
free_gb = free // (1024**3)
print(f"Free space: {free_gb} GB")

if free_gb < 5:
    print("⚠️  WARNING: Low disk space!")
    print("   Consider cleaning up old models or using Google Drive")
else:
    print("✅ Enough space available")
```

