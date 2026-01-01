# איך להעלות קבצים ל-Google Colab - מדריך מפורט

## 3 דרכים להעלות קבצים ל-Colab

---

## דרך 1: Google Drive (מומלץ ביותר!) ⭐

### למה זה הכי טוב:
- ✅ **Storage גדול** - 15GB חינם
- ✅ **מהיר** - חיבור מהיר
- ✅ **נשמר** - הקבצים נשארים גם אחרי שה-session נסגר
- ✅ **קל** - פשוט להעלות

### שלב 1: העלה ל-Google Drive

1. **לך ל-Google Drive:** https://drive.google.com
2. **התחבר** עם Google Account
3. **New** → **File upload**
4. **בחר את הקובץ:** `sign_language_data.tar.gz`
5. **חכה** עד שההעלאה מסתיימת

### שלב 2: התחבר ל-Drive ב-Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

**מה יקרה:**
- תראה הודעה - לחץ על הלינק
- בחר את ה-Google Account
- העתק את הקוד שמופיע
- הדבק בקוד ב-Colab
- לחץ Enter

**אם זה עובד:** תראה "Mounted at /content/drive" ✅

### שלב 3: העתק את הקבצים

```python
# העתק את הארכיון
!cp /content/drive/MyDrive/sign_language_data.tar.gz ./

# פתח את הארכיון
!tar -xzf sign_language_data.tar.gz

# בדוק שהנתונים שם
!ls -la Data/
```

**אם תראה את תיקיית Data - הכל עובד!** ✅

---

## דרך 2: Upload ישיר ל-Colab

### למה זה טוב:
- ✅ **מהיר** - לא צריך Drive
- ✅ **פשוט** - רק Upload

### חסרונות:
- ⚠️ **נמחק** - הקבצים נמחקים כשה-session נסגר
- ⚠️ **מוגבל** - עד כמה GB

### איך לעשות:

```python
from google.colab import files

# Upload קובץ אחד
uploaded = files.upload()

# אחרי שתבחר את הקובץ, הוא יופיע
# הקובץ יישמר ב-/content/ עם אותו שם
```

**אחרי Upload:**
```python
# פתח את הארכיון
!tar -xzf sign_language_data.tar.gz

# או אם יש לך מספר קבצים:
for filename in uploaded.keys():
    print(f'Uploaded: {filename}')
```

---

## דרך 3: מ-S3 (אם כבר העלית ל-S3)

### למה זה טוב:
- ✅ **אם כבר יש** - אם כבר העלית ל-S3

### איך לעשות:

```python
# התקן boto3
!pip install boto3

import boto3
import os

# הגדר credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_SECRET_ACCESS_KEY'

# הורד מ-S3
s3 = boto3.client('s3')

# הורד את הארכיון
s3.download_file(
    'sign-language-project-yourname',  # bucket name
    'data/sign_language_data.tar.gz',  # path in S3
    'sign_language_data.tar.gz'        # local filename
)

# פתח את הארכיון
!tar -xzf sign_language_data.tar.gz

# בדוק
!ls -la Data/
```

---

## איך להעלות את הקוד (scripts)

### אפשרות 1: מ-Google Drive (מומלץ)

**1. העלה את תיקיית scripts ל-Google Drive:**
- לך ל-Google Drive
- Upload → Folder upload
- בחר את תיקיית `scripts`
- העלה

**2. ב-Colab:**
```python
# Mount Drive (אם עדיין לא עשית)
from google.colab import drive
drive.mount('/content/drive')

# העתק את scripts
!cp -r /content/drive/MyDrive/scripts ./

# או אם העלית את כל הפרויקט:
!cp -r /content/drive/MyDrive/signlanguage/scripts ./
!cp /content/drive/MyDrive/signlanguage/requirements.txt ./
```

### אפשרות 2: מ-GitHub (אם יש repository)

```python
!git clone https://github.com/YOUR_USERNAME/signlanguage.git
!cd signlanguage
```

### אפשרות 3: העלה ישירות (לקבצים קטנים)

```python
from google.colab import files

# Upload כל קובץ בנפרד
uploaded = files.upload()

# אחרי Upload, צור תיקיות
!mkdir -p scripts

# העבר את הקבצים
import shutil
for filename in uploaded.keys():
    shutil.move(filename, f'scripts/{filename}')
```

### אפשרות 4: יצירה ישירה ב-Colab (אם הקוד קצר)

```python
# צור את הקובץ ישירות
%%writefile train_model.py
# כאן תעתיק את התוכן של train_model.py
```

---

## Workflow מומלץ - סיכום

### 1. הכנה (על המחשב המקומי):

```bash
# צור ארכיון
tar -czf sign_language_data.tar.gz Data/

# העלה ל-Google Drive (דרך הדפדפן)
# https://drive.google.com → Upload → sign_language_data.tar.gz
```

### 2. ב-Google Colab:

```python
# שלב 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# שלב 2: העתק נתונים
!cp /content/drive/MyDrive/sign_language_data.tar.gz ./
!tar -xzf sign_language_data.tar.gz

# שלב 3: העתק scripts (אם העלית)
!cp -r /content/drive/MyDrive/signlanguage/scripts ./
!cp /content/drive/MyDrive/signlanguage/requirements.txt ./

# שלב 4: התקן תלויות
!pip install -r requirements.txt

# שלב 5: בדוק שהכל עובד
!ls -la Data/
!ls -la scripts/

# שלב 6: הרץ אימון
!python scripts/train_model.py --csv Data/Labels/dataset.csv
```

---

## טיפים חשובים

### 1. בדוק שהקבצים נשמרו:
```python
!ls -la /content/drive/MyDrive/  # רשימת קבצים ב-Drive
!ls -la Data/                    # רשימת נתונים
!ls -la scripts/                 # רשימת scripts
```

### 2. אם יש שגיאות בנתיבים:
```python
# בדוק איפה אתה
!pwd

# בדוק מה יש בתיקייה
!ls -la

# אם צריך, צור תיקיות
!mkdir -p Data scripts models
```

### 3. אם ההעלאה איטית:
- Google Drive לפעמים איטי
- נסה Upload ישיר (דרך 2)
- או S3 אם כבר יש

---

## שאלות נפוצות

**Q: כמה זמן לוקח להעלות?**  
A: תלוי בגודל:
- 50MB: ~1-2 דקות
- 500MB: ~10-20 דקות
- 1GB+: ~20-40 דקות

**Q: מה אם ההעלאה נכשלה?**  
A: נסה שוב, או חלק את הקבצים לחלקים קטנים יותר.

**Q: מה אם יש שגיאת permissions?**  
A: ודא שהתחברת ל-Drive נכון (drive.mount).

---

## המלצה סופית

**הדרך הכי טובה: Google Drive** ⭐

1. העלה ל-Google Drive (דרך הדפדפן)
2. Mount Drive ב-Colab
3. העתק את הקבצים
4. זה נשאר גם אחרי שה-session נסגר!

---

**בהצלחה! 🚀**

